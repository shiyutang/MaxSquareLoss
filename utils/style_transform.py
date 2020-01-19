from pathlib import Path
from PIL import Image
import os
import random
import logging

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

style_dirs = [f for f in Path("/data/Projects/MaxSquareLoss/imagenet_style").glob("*")]


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


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),)

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


class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
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
        return loss_c, loss_s


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())






def style_transfer_AdaIN(content = None, content_dir= None, style=None, style_dir=None,
                   vgg_pretrain = "/data/Projects/pytorch-AdaIN/models/vgg_normalised.pth",vgg = vgg,
                   decoder_pretrain="/data/Projects/pytorch-AdaIN/models/decoder.pth",decoder=decoder,
                   content_size=(1052, 1914),style_size=(1052,1914),crop=None,save_ext=".jpg",
                   output_path="", preserve_color=None, alpha=1.0,
                   style_interpolation_weight=None,exp_tag = "", do_interpolation=False):


    def test_transform(size, crop):
        transform_list = []
        if size != 0:
            transform_list.append(transforms.Resize(size))
        if size != 0 and crop:
            transform_list.append(transforms.CenterCrop(size))
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
        return transform


    def style_transfer(vgg, decoder, content, style, alpha=1.0,
                       device=None, interpolation_weights=None):
        assert (0.0 <= alpha <= 1.0)
        content_f = vgg(content)
        style_f = vgg(style)
        if interpolation_weights:
            _, C, H, W = content_f.size()
            feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
            base_feat = adaptive_instance_normalization(content_f, style_f)
            for i, w in enumerate(interpolation_weights):  # 四个style根据比例合并
                feat = feat + w * base_feat[i:i + 1]
            content_f = content_f[0:1]  # 内容特征每一层都是一样的，因此取一层即可
        else:
            feat = adaptive_instance_normalization(content_f, style_f)
        feat = feat * alpha + content_f * (1 - alpha)
        return decoder(feat)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)


    assert (content or content_dir)
    if content:
        content_paths = [Path(content)]
    else:
        content_dir = Path(content_dir)
        content_paths = [f for f in content_dir.glob('*')]
        # find all files that satisfy this pattern


    # Either --style or --styleDir should be given.
    assert (style or style_dir),"please specify style dir or style"
    if style:
        style_paths = style
        if len(style_paths)==1:
            style_paths = [Path(style)]
        else:
            do_interpolation = True
            assert (style_interpolation_weight!=""), \
                "please specify interpolation weights"
            weights = [int(i) for i in style_interpolation_weight.split(",")]
            interpolation_weight = [i/sum(weights) for i in weights]
    else:
        style_paths = [p for p in Path(style_dir).glob("*")]


    decoder.eval()
    vgg.eval()
    decoder.load_state_dict(torch.load(decoder_pretrain))
    vgg.load_state_dict(torch.load(vgg_pretrain))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)


    content_tf = test_transform(content_size,crop)
    style_tf = test_transform(style_size,crop)

    for i,content_path in enumerate(tqdm(content_paths)):
        out_name = Path.joinpath(output_dir,str(content_path.stem)+".{}".format(save_ext))
        if out_name.exists():
            continue
        if do_interpolation:
            style = torch.stack([style_tf(Image.open(file)) for file in style_paths])
            content = content_tf(Image.open(content_path)).unsqueeze(0).expand_as(style)
            style = style.to(device)
            content = content.to(device)

            with torch.no_grad():
                output_Tensor = style_transfer(vgg,decoder,content,style,alpha,device,
                                            interpolation_weight)

            output_Tensor.cpu()
            out_name = Path.joinpath(output_dir,str(content_path.stem)+".{}".format(save_ext))
            save_image(output_Tensor,out_name)

        else:
            for style_path in style_paths:
                content = content_tf(Image.open(content_path))
                style = style_tf(Image.open(style_path))

                if preserve_color:
                    style = coral(style,content)
                style = style.to(device).unsqueeze(0)
                content = content.to(device).unsqueeze(0)
                with torch.no_grad():
                    output_Tensor = style_transfer(vgg,decoder,content,style,alpha,device)
                output_Tensor = output_Tensor.cpu()
                out_name = os.path.join(output_dir,"{}_stylized_{}.{}"\
                                        .format(content_path.stem,style_path.stem,save_ext))
                save_image(output_Tensor,out_name)



if __name__ == '__main__':
    content_dirs = [f for f in Path("/data/Projects/ADVENT/data/Cityscapes/leftImg8bit/train").glob("*")][9:]
    # content_dirs = [f for f in Path("/data/Projects/ADVENT/data/GTA5/images").glob("*")][12500:]
    exp_tag = "cityscape"
    style_interpolation_weight = "1,1,1,1"

    style_dir= Path("/data/Projects/MaxSquareLoss/imagenet_style/ambulance") #style_dirs[4]
    print("style_dir",style_dir)
    style = random.sample([p for p in style_dir.glob("*")],4)
    for content_dir in tqdm(content_dirs):
        style_transfer_AdaIN(content=None, content_dir=content_dir, style=style, style_dir=None,
                             vgg_pretrain="/data/Projects/pytorch-AdaIN/models/vgg_normalised.pth",
                             decoder_pretrain="/data/Projects/pytorch-AdaIN/models/decoder.pth",
                             vgg=vgg,decoder=decoder,do_interpolation=False,
                             content_size=(1052, 1914), style_size=(1052, 1914), crop=None, save_ext="png",
                             output_path="/data/Projects/ADVENT/data/Cityscapes_{}_newsize/leftImg8bit/train/{}".\
                                            format(style_dir.stem,content_dir.stem),
                             preserve_color=None, alpha=1.0,
                             style_interpolation_weight=style_interpolation_weight, exp_tag=exp_tag)
