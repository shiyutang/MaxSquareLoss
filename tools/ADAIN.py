import os
import sys

import torch.backends.cudnn as cudnn
from PIL import Image

sys.path.append(os.path.abspath('.'))
from utils.train_helper import Net
from utils.loss import *
from datasets.gta5_Dataset import GTA5_Dataset, FlatFolderDataset
from torch.optim import lr_scheduler

from tools.train_source import *

cudnn.benchmark = True  # network speedup
# Disable DecompressionBombError
Image.MAX_IMAGE_PIXELS = None


class UDATrainer(Trainer):
    def __init__(self):
        super().__init__(args, train_id, logger)  # 调用父类的初始化，这样不用重写函数，同时初始化了应有的参数
        self.scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                        mode='min', factor=0.9, patience=3, verbose=True)

        # source train loader
        self.source_dataset_train = GTA5_Dataset(args,
                                                 data_root_path=self.datasets_path["gta5"]['data_root_path'],
                                                 list_path=self.datasets_path['gta5']['list_path'],
                                                 gt_path=self.datasets_path['gta5']['gt_path'],
                                                 split='train_style')
        self.source_dataloader = data.DataLoader(self.source_dataset_train,
                                                 batch_size=self.args.batch_size,
                                                 shuffle=False,
                                                 num_workers=self.args.data_loader_workers,
                                                 pin_memory=self.args.pin_memory,
                                                 drop_last=True)

        # source validation loader
        self.source_dataset_val = GTA5_Dataset(args,
                                               data_root_path=self.datasets_path['gta5']['data_root_path'],
                                               list_path=self.datasets_path['gta5']['list_path'],
                                               gt_path=self.datasets_path['gta5']['gt_path'],
                                               split='val_style')

        self.source_dataloader_val = data.DataLoader(self.source_dataset_val,
                                                     batch_size=self.args.batch_size,
                                                     shuffle=False,
                                                     num_workers=self.args.data_loader_workers,
                                                     pin_memory=self.args.pin_memory,
                                                     drop_last=True)

        ## target dataset train and validation
        self.target_dataset_train = City_Dataset(args,
                                                 data_root_path=self.datasets_path['cityscapes']['data_root_path'],
                                                 list_path=self.datasets_path['cityscapes']['list_path'],
                                                 gt_path=self.datasets_path['cityscapes']['gt_path'],
                                                 split='train_style',
                                                 class_16=args.class_16)

        self.target_dataloader = data.DataLoader(self.target_dataset_train,
                                                 batch_size=self.args.batch_size,
                                                 shuffle=True,
                                                 num_workers=self.args.data_loader_workers,
                                                 pin_memory=self.args.pin_memory,
                                                 drop_last=True)
        self.target_dataset_val = \
            City_Dataset(args,
                         data_root_path=self.datasets_path['cityscapes']['data_root_path'],
                         list_path=self.datasets_path['cityscapes']['list_path'],
                         gt_path=self.datasets_path['cityscapes']['gt_path'],
                         split='val_style',
                         class_16=args.class_16)

        self.target_val_dataloader = data.DataLoader(self.target_dataset_val,
                                                     batch_size=self.args.batch_size,
                                                     shuffle=False,
                                                     num_workers=self.args.data_loader_workers,
                                                     pin_memory=self.args.pin_memory,
                                                     drop_last=True)

        style_dataset = FlatFolderDataset(
            root="/data/Projects/MaxSquareLoss/imagenet_style/ambulance",
            transform=self.source_dataset_val.adain_transform(size=(self.args.base_size[0], self.args.base_size[1])))

        self.style_dataloader = iter(data.DataLoader(style_dataset, batch_size=4, shuffle=False,
                                                     num_workers=self.args.data_loader_workers,
                                                     pin_memory=self.args.pin_memory,
                                                     drop_last=True))

        self.batch_style = next(self.style_dataloader)
        self.dataloader.val_loader = self.target_val_dataloader
        self.dataloader.valid_iterations = (len(self.target_dataset_val) + self.args.batch_size) // self.args.batch_size

        self.network = Net()
        self.network.train()

        # define target loss
        self.ignore_index = -1
        if self.args.target_mode == "hard":
            self.target_loss = nn.CrossEntropyLoss(ignore_index=-1)
        elif self.args.target_mode == "entropy":
            self.target_loss = softCrossEntropy(ignore_index=-1)
        elif self.args.target_mode == "IW_entropy":
            self.target_loss = IWsoftCrossEntropy(ignore_index=-1, num_class=self.args.num_classes,
                                                  ratio=self.args.IW_ratio)
        elif self.args.target_mode == "maxsquare":
            self.target_loss = MaxSquareloss(ignore_index=-1, num_class=self.args.num_classes)
        elif self.args.target_mode == "IW_maxsquare":
            self.target_loss = IW_MaxSquareloss(ignore_index=-1, num_class=self.args.num_classes,
                                                ratio=self.args.IW_ratio)

        self.current_round = self.args.init_round
        self.round_num = self.args.round_num

        self.target_loss.to(self.device)

        self.target_hard_loss = nn.CrossEntropyLoss(ignore_index=-1)

        # initially add network parameters into

        # self.adain_optimizer = torch.optim.Adam(self.network.parameters(),
        #                         lr=self.args.adain_lr)
        # self.adain_optimizer.zero_grad()

        self.optimizer.zero_grad()

    def save_tensor_as_Image(self, tensor, path, cnt=0, nrow=8, padding=2,
                             normalize=False, range=None, scale_each=False, pad_value=0):
        from PIL import Image
        from utils.train_helper import make_grid
        image = tensor.cpu().clone()
        grid = make_grid(image, nrow=nrow, padding=padding, pad_value=pad_value,
                         normalize=normalize, range=range, scale_each=scale_each)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        # print('ndarr.max(),x.min(),x.shape,x[0,0]', ndarr.max(), ndarr.min(), ndarr.shape, ndarr[0, 0, 0])
        im = Image.fromarray(ndarr)
        if cnt % 10000 == 0:
            im.save(path, format='png')
        return im

    def train_target(self, pred):
        if isinstance(pred, tuple):
            pred_2 = pred[1]
            pred = pred[0]
            pred_P_2 = F.softmax(pred_2, dim=1)
        pred_P = F.softmax(pred, dim=1)

        ## hard or soft labels
        if self.args.target_mode == "hard":
            label = torch.argmax(pred_P.detach(), dim=1)
            if self.args.multi: label_2 = torch.argmax(pred_P_2.detach(), dim=1)
        else:  ## 软标签，不要求求出最大
            label = pred_P
            if self.args.multi: label_2 = pred_P_2

        maxpred, argpred = torch.max(pred_P.detach(), dim=1)
        if self.args.multi:
            maxpred_2, argpred_2 = torch.max(pred_P_2.detach(), dim=1)

        if self.args.target_mode == "hard":
            mask = (maxpred > self.threshold)
            label = torch.where(mask, label, torch.ones(1).to(self.device, dtype=torch.long) * self.ignore_index)

        self.loss_target = self.args.lambda_target * self.target_loss(pred, label)

        loss_target_ = self.loss_target

        ######################################
        # Multi-level Self-produced Guidance #
        ######################################
        if self.args.multi:
            pred_c = (pred_P + pred_P_2) / 2
            maxpred_c, argpred_c = torch.max(pred_c, dim=1)
            self.mask = (maxpred > self.threshold) | (maxpred_2 > self.threshold)

            label_2 = torch.where(self.mask, argpred_c,
                                  torch.ones(1).to(self.device, dtype=torch.long) * self.ignore_index)
            self.loss_target_2 = self.args.lambda_seg * self.args.lambda_target * self.target_hard_loss(pred_2, label_2)
            loss_target_ += self.loss_target_2
            self.loss_target_value_2 += self.loss_target_2 / self.iter_num

        loss_target_.backward()
        self.loss_target_value += self.loss_target / self.iter_num

    def train_source(self, pred, y):
        if isinstance(pred, tuple):  # multi has 2 prediction
            pred_2 = pred[1]
            pred = pred[0]

        y = torch.squeeze(y, 1)
        self.loss_val = self.loss(pred, y)

        loss_ = self.loss_val
        if self.args.multi:
            loss_2 = self.args.lambda_seg * self.loss(pred_2, y)
            loss_ += loss_2
            self.loss_seg_value_2 += loss_2.cpu().item() / self.iter_num  #

        loss_.backward()
        self.loss_seg_value += self.loss_val.cpu().item() / self.iter_num

    def main(self):
        # load pretrained checkpoint
        if self.args.checkpoint_dir is not None:
            self.load_checkpoint(self.args.checkpoint_dir)

        if not self.args.continue_training:
            self.best_MIou = 0
            self.best_iter = 0
            self.current_iter = 0
            self.current_epoch = 0
        else:
            self.load_checkpoint(self.args.checkpoint_dir)
            self.best_iter = self.current_iter  # the best iteration for target
            self.best_source_iter = self.current_iter  # the best iteration for source

        self.args.iter_max = self.current_iter + self.dataloader.num_iterations * self.args.epoch_each_round * self.round_num

        # train
        self.train_round()

        self.writer.close()

    def train_round(self):
        for r in range(self.current_round, self.round_num):
            print("\n############## Begin {}/{} Round! #################\n".format(self.current_round + 1,
                                                                                   self.round_num))
            print("epoch_each_round:", self.args.epoch_each_round)

            self.epoch_num = self.current_epoch + (self.current_round + 1) * self.args.epoch_each_round

            # generate threshold
            self.threshold = self.args.threshold
            self.train()  ## it was using the method in the trainer

            self.current_round += 1

    def save_images(self, pred, label, image_tensor, name, epoch):
        pred = decode_labels(pred, 1)
        label = decode_labels(label, 1)
        image_tensor = inv_preprocess(image_tensor, 1)
        self.writer.add_image('{}/pred_{}'.format(epoch, name), pred, epoch)
        self.writer.add_image('{}/label_{}'.format(epoch, name), label, epoch)
        self.writer.add_image('{}/image_{}'.format(epoch, name), image_tensor, epoch)

    def train_one_epoch(self, epoch=0):
        tqdm_epoch = tqdm(zip(self.source_dataloader, self.target_dataloader),  # , self.source_dataloader_trans),
                          total=self.dataloader.num_iterations,
                          desc="Train Round-{}-Epoch-{}-total-{}". \
                          format(self.current_round, self.current_epoch + 1, self.epoch_num))
        self.logger.info("Training one epoch... in the method defined by ADAIN")
        self.Eval.reset()

        # Initialize your average meters
        self.loss_seg_value = 0
        self.loss_target_value = 0
        self.loss_seg_value_2 = 0
        self.loss_target_value_2 = 0
        self.std = None
        self.iter_num = self.dataloader.num_iterations

        # Set the model to be in training mode (for batchnorm and dropout)
        self.model.train()

        self.style_loss_weight = 1
        for batch_s, batch_t in tqdm_epoch:
            # self.adain_lr_scheduler(optimizer=self.adain_optimizer,
            #                         iteration_count=self.current_iter,lr = args.adain_lr)

            ############################################
            # get source and target picture from ADAIN #
            ############################################
            # source
            content, source_label_tf, source_id = batch_s
            with torch.no_grad():
                if self.args.crop_trans:
                    trans_source = self.cropTrans(content)
                else:
                    trans_source = self.network(content, self.batch_style)

            # target
            content, target_label, target_id = batch_t
            with torch.no_grad():
                if self.args.crop_trans:
                    trans_target = self.cropTrans(content)
                else:
                    trans_target = self.network(content, self.batch_style)
            if self.current_iter == 0:
                self.save_tensor_as_Image(trans_source, '/data/result/source_crop_trans.png')
                self.save_tensor_as_Image(trans_target, '/data/result/target_crop_trans.png')

            ##########################
            # source supervised loss #
            ##########################
            # transform pic as two-stage and resize label
            trans_source_tensor, source_label_tf = self.seg_transform(trans_source, source_label_tf)

            if self.cuda:
                x, y = Variable(trans_source_tensor).to('cuda:0'), \
                       Variable(source_label_tf).to(device='cuda:0', dtype=torch.long)

            pred = self.model(x)
            self.train_source(pred, y)
            if self.current_iter % self.dataloader.num_iterations == 0:
                self.save_images(np.argmax(pred[0].cpu().detach(), axis=1), source_label_tf, trans_source,
                                 'source_train', epoch)

            #####################
            # train with target #
            #####################
            trans_target_tensor, target_label = self.seg_transform(trans_target, target_label)

            if self.cuda:
                x = Variable(trans_target_tensor).to('cuda:0')

            pred = self.model(x)
            self.train_target(pred)
            if self.current_iter % self.dataloader.num_iterations == 0:
                self.save_images(np.argmax(pred[0].cpu().detach(), axis=1), target_label, trans_target, 'target_train',
                                 epoch)

            self.optimizer.step()
            self.optimizer.zero_grad()

            # self.adain_optimizer.step()
            # self.adain_optimizer.zero_grad()

            self.current_iter += 1
            # if self.current_iter == 1:
            #     break

        self.scheduler.step(self.loss_seg_value)
        self.writer.add_scalar('train_loss', self.loss_seg_value, self.current_epoch)
        tqdm.write("The average loss of train epoch-{}-:{}".format(self.current_epoch, self.loss_seg_value))
        self.writer.add_scalar('target_loss', self.loss_target_value, self.current_epoch)
        tqdm.write(
            "The average target_loss of train epoch-{}-:{:.3f}".format(self.current_epoch, self.loss_target_value))

        if self.args.multi:
            self.writer.add_scalar('train_loss_2', self.loss_seg_value_2, self.current_epoch)
            tqdm.write("The average loss_2 of train epoch-{}-:{}"
                       .format(self.current_epoch, self.loss_seg_value_2))
            self.writer.add_scalar('target_loss_2', self.loss_target_value_2, self.current_epoch)
            tqdm.write("The average target_loss_2 of train epoch-{}-:{:.3f}"
                       .format(self.current_epoch, self.loss_target_value_2))
        tqdm_epoch.close()

        # eval on source domain
        self.validate_source(epoch)

    def cropT(self, tensor):
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        c, h, w = tensor.shape
        a1 = tensor[:, 0:h // 2, 0:w // 2]
        a2 = tensor[:, 0:h // 2, w // 2:w]
        a3 = tensor[:, h // 2:h, 0:w // 2]
        a4 = tensor[:, h // 2:h, w // 2:w]
        return a1, a2, a3, a4

    def catT(self, a1, a2, a3, a4):
        aup = torch.cat((a1, a2), dim=2)
        adown = torch.cat((a3, a4), dim=2)
        a = torch.cat((aup, adown), dim=1)
        return a

    def cropTrans(self, content):
        contents = self.cropT(content)
        trans_results = []
        for c in contents:
            assert c.shape[1:] == (self.args.crop_size[1], self.args.crop_size[0]), \
                'the shape of croped pic is {} but should be crop_size'.format(c.shape[1:])
            c = F.interpolate(c.unsqueeze(0), (self.args.base_size[1], self.args.base_size[0]))

            assert len(c.shape) == 4, 'the dim of pic input to network need to be 4'
            ct = self.network(c, self.batch_style)  # 输出正确

            ct = F.interpolate(ct, (self.args.crop_size[1], self.args.crop_size[0]))
            trans_results.append(ct.squeeze(0))
        result = self.catT(trans_results[0], trans_results[1], trans_results[2], trans_results[3])
        return result.unsqueeze(0)


def add_UDA_train_args(arg_parser):
    arg_parser.add_argument('--source_dataset', default='gta5', type=str,
                            choices=['gta5', 'synthia'],
                            help='source dataset choice')
    arg_parser.add_argument('--source_split', default='train', type=str,
                            help='source datasets split')
    arg_parser.add_argument('--init_round', type=int, default=0,
                            help='init_round')
    arg_parser.add_argument('--round_num', type=int, default=40,
                            help="num round")
    arg_parser.add_argument('--epoch_each_round', type=int, default=2,
                            help="epoch each round")
    arg_parser.add_argument('--target_mode', type=str, default="IW_maxsquare",
                            choices=['maxsquare', 'IW_maxsquare', 'entropy', 'IW_entropy', 'hard'],
                            help="the loss function on target domain")
    arg_parser.add_argument('--lambda_target', type=float, default=0.09,
                            help="lambda of target loss")
    arg_parser.add_argument('--gamma', type=float, default=0,
                            help='parameter for scaled entorpy')
    arg_parser.add_argument('--IW_ratio', type=float, default=0.2,
                            help='the ratio of image-wise weighting factor')
    arg_parser.add_argument('--threshold', type=float, default=0.95,
                            help="threshold for Self-produced guidance")
    arg_parser.add_argument('--adain_lr', type=float, default=1e-4,
                            help='learning rate for adain')
    return arg_parser


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    arg_parser = argparse.ArgumentParser()
    arg_parser = add_train_args(arg_parser)
    arg_parser = add_UDA_train_args(arg_parser)

    args = arg_parser.parse_args()
    args, _, logger = init_args(args)

    train_id = "GTA52Cityscapes_" + args.target_mode

    agent = UDATrainer()
    agent.main()
