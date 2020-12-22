import torch.nn as nn
import torch
import random
import copy
import pickle

from graphs.models.deeplab_multi import DeeplabMulti

def get_model(args):
    if args.backbone == "deeplabv2_multi":
        model = DeeplabMulti(num_classes=args.num_classes,
                            pretrained=args.imagenet_pretrained)
        params = model.optim_parameters(args) # 分类层是十倍的学习率，其他是正常学习率
        args.numpy_transform = True
    return model, params


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)), #0
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),#3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),                       #6
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),                       #9
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),                       #12
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),     #14
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'), #16
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),                        #19
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'), #23
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),                                 #26
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),                   #28
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        vgg.load_state_dict(torch.load("/data/Projects/pytorch-AdaIN/models/vgg_normalised.pth"))
        enc_layers = list(vgg.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])#.to('cuda:2')  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])#.to('cuda:2')  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])#.to('cuda:2')  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])#.to('cuda:1')  # relu3_1 -> relu4_1
        self.encoder = nn.Sequential(*list(vgg.children())[:31]).to('cuda:2')

        pretrained_decoder = '/data/Projects/pytorch-AdaIN/experiments/gta5pcity_ambulance_alpha1wts1/decoder_iter_160000.pth.tar'
        # pretrained_decoder = '/data/Projects/pytorch-AdaIN/experiments/gta5pcity_ambulance_alpha1wts1awts1e-3_affineloss_pretrain11/decoder_iter_57000.pth.tar'

        decoder.load_state_dict(torch.load(pretrained_decoder))
        print('###################################')
        print('the decoder is from {}'.format(pretrained_decoder))
        dec_layers = list(decoder.children())

        # self.dec_1 = nn.Sequential(*dec_layers[:6]).to('cuda:1')
        # self.dec_2 = nn.Sequential(*dec_layers[6:13]).to('cuda:4')
        # self.dec_4 = nn.Sequential(*dec_layers[13:20]).to('cuda:5')
        # self.dec_5 = nn.Sequential(*dec_layers[19:25]).to('cuda:6')
        # self.dec_6 = nn.Sequential(*dec_layers[25:27]).to('cuda:7')
        # self.dec_7 = nn.Sequential(*dec_layers[27:]).to('cuda:3')
        self.decoder = decoder.to("cuda:1")
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward_with_losses(self, content, batch_style, alpha=1.0,
                            weights=(1,1,1,1)):

        assert 0 <= alpha <= 1
        if torch.cuda.is_available():
            batch_style = batch_style.cuda()
        content = content.expand_as(batch_style)
        if torch.cuda.is_available():
            content = content.cuda()

        interpolation_weights = [i / sum(weights) for i in weights]

        # fusion style
        _, C, H, W = batch_style.size()
        style_gather = torch.FloatTensor(1, C, H, W).zero_()
        if torch.cuda.is_available():
            style_gather = style_gather.cuda()
        for j,wt in enumerate(interpolation_weights):
            style_gather += wt * batch_style[j:j+1]
        style_gather = self.encode_with_intermediate(style_gather)

        style_f = self.encoder(batch_style)
        content_f = self.encoder(content)

        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_()
        if torch.cuda.is_available():
            feat = feat.cuda()
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        t = alpha * feat+ (1 - alpha) * content_f[0:1]

        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1],t)
        loss_s = self.calc_style_loss(g_t_feats[-1],style_gather[-1])
        for i in range(len(interpolation_weights)-1):
            loss_s += self.calc_style_loss(g_t_feats[i],style_gather[i])

        return loss_s, loss_c, g_t


    def forward(self, content, batch_style, alpha=1.0,
                        weights=(1,1,1,1),save_path = None):
        assert 0 <= alpha <= 1
        if torch.cuda.is_available():
            batch_style = batch_style.to('cuda:2')
        # print('batch_style.size inside', style.size())

        content = content.expand_as(batch_style)
        if torch.cuda.is_available():
            content = content.to('cuda:2')
        interpolation_weights = [i / sum(weights) for i in weights]

        # content_f = self.enc_4(self.enc_3(self.enc_2(self.enc_1(content))).to('cuda:1'))
        # style_f = self.enc_4(self.enc_3(self.enc_2(self.enc_1(content))).to('cuda:1'))
        content_f = self.encoder(content)
        torch.cuda.empty_cache()
        style_f = self.encoder(batch_style)

        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_()
        if torch.cuda.is_available():
            feat = feat.to('cuda:2')
        base_feat = adaptive_instance_normalization(content_f,style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]

        t = alpha * feat + (1 - alpha) * content_f[0:1]

        # g_t1 = self.dec_1(t.to('cuda:1'))
        # g_t2 = self.dec_2(g_t1.to('cuda:4'))
        # g_t4 = self.dec_4(g_t2.to('cuda:5'))
        # g_t5 = self.dec_5(g_t4.to('cuda:6'))
        # g_t6 = self.dec_6(g_t5.to('cuda:7'))
        # g_t7 = self.dec_7(g_t6.to('cuda:3'))
        g_t7 = self.decoder(t.to('cuda:1'))

        return g_t7


class STNet_refer(nn.Module):
    def __init__(self):
        super(STNet_refer, self).__init__()
        vgg.load_state_dict(torch.load("/data/Projects/pytorch-AdaIN/models/vgg_normalised.pth"))
        enc_layers = list(vgg.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4]).to('cuda:2')  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11]).to('cuda:2')  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:30]).to('cuda:2')  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[30:31]).to('cuda:2')  # relu3_1 -> relu4_1
        self.encoder = nn.Sequential(*list(vgg.children())[:31])

        # pretrained_decoder = 'gta5pcity_ambulance_alpha1wts1awts1e-3_affineloss_pretrain11/decoder_iter_57000.pth.tar'
        # pretrained_decoder = 'gta5pcity_ambulance_alpha1wts1/decoder_iter_160000.pth.tar'
        # pretrained_decoder = 'gta5pcity_ambulance_alpha1wts0p5/decoder_iter_160000.pth.tar'
        # pretrained_decoder = '/data/Projects/pytorch-AdaIN/experiments_stylewt5/decoder_iter_160000.pth.tar'
        pretrained_decoder = '/data/Projects/MaxSquareLoss/experiments/gta5pcity_ambulance_alpha1wts1_classifier_512r1024/decoder_iter_99000.pth.tar'
        decoder.load_state_dict(torch.load(pretrained_decoder))
                # torch.load('/data/Projects/pytorch-AdaIN/experiments/{}'.format(pretrained_decoder)))
        print('###################################')
        print('the decoder is from {}'.format(pretrained_decoder))
        dec_layers = list(decoder.children())
        self.dec_1 = nn.Sequential(*dec_layers[:10]).to('cuda:1')
        self.dec_4 = nn.Sequential(*dec_layers[10:24]).to('cuda:1')
        self.dec_7 = nn.Sequential(*dec_layers[24:]).to('cuda:1')

        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4','dec_1','dec_4','dec_7']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, batch_style, alpha=1.0,
                        weights=(1,1,1,1),save_path = None):
        assert 0 <= alpha <= 1
        if torch.cuda.is_available():
            batch_style = batch_style.to('cuda:2')

        content = content.expand_as(batch_style)
        if torch.cuda.is_available():
            content = content.to('cuda:2')
        interpolation_weights = [i / sum(weights) for i in weights]

        content_f = self.enc_1(content)
        content_f = self.enc_2(content_f)#.to('cuda:3'))
        content_f = self.enc_3(content_f)
        content_f = self.enc_4(content_f)
        torch.cuda.empty_cache()
        style_f = self.enc_4(self.enc_3(self.enc_2(self.enc_1(batch_style))))#.to('cuda:4'))

        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_()
        if torch.cuda.is_available():
            feat = feat.to('cuda:2')
        base_feat = adaptive_instance_normalization(content_f,style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]

        t = alpha * feat+ (1 - alpha) * content_f[0:1]

        g_t1 = self.dec_1(t.to('cuda:1'))
        g_t4 = self.dec_4(g_t1.to('cuda:1'))
        g_t7 = self.dec_7(g_t4.to('cuda:1'))

        return  g_t7



def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    import math
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid



class STNet(nn.Module):
    def __init__(self, encoder, decoder):
        super(STNet, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def gram_matrix(self,tensor):
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        return torch.mm(tensor, tensor.t())  # gram

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)


    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)

        t = adaptive_instance_normalization(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)

        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])

        return loss_c, loss_s, g_t