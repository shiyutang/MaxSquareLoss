import os

import sys
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import transforms as ttransforms
import torch.utils.data as data
from PIL import Image
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy


sys.path.append(os.path.abspath('.'))
from utils.train_helper import Net  ##
from utils.loss import *
from datasets.cityscapes_Dataset import City_Dataset, City_DataLoader, inv_preprocess, decode_labels
from datasets.gta5_Dataset import GTA5_Dataset, FlatFolderDataset
from torch.optim import lr_scheduler

from tools.train_source import *

cudnn.benchmark = True # network speedup
# Disable DecompressionBombError
Image.MAX_IMAGE_PIXELS = None

class UDATrainer(Trainer):
    def __init__(self, args, cuda=True, train_id="None", logger=None):
        super().__init__(args, cuda, train_id, logger)  # 调用父类的初始化，这样不用重写函数，同时初始化了应有的参数
        self.scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                     mode='min', factor=0.9, patience=3, verbose=True)

        ## source train loader
        self.source_dataset_train = GTA5_Dataset(args,
                               data_root_path=self.datasets_path["gta5"]['data_root_path'],
                               list_path=self.datasets_path['gta5']['list_path'],
                               gt_path=self.datasets_path['gta5']['gt_path'],
                               split='train_style',
                               base_size=args.base_size,
                               crop_size=args.crop_size)
        self.source_dataloader = \
            data.DataLoader(self.source_dataset_train,
                           batch_size=self.args.batch_size,
                           shuffle=True,
                           num_workers=self.args.data_loader_workers,
                           pin_memory=self.args.pin_memory,
                           drop_last=True)

        ## source validation loader
        self.source_dataset_val = GTA5_Dataset(args,
                                       data_root_path=self.datasets_path['gta5']['data_root_path'],
                                       list_path=self.datasets_path['gta5']['list_path'],
                                       gt_path=self.datasets_path['gta5']['gt_path'],
                                       split='val_style',
                                       base_size=args.base_size,
                                       crop_size=args.crop_size)
        self.source_dataloader_val = data.DataLoader\
                                    (self.source_dataset_val,
                                     batch_size=self.args.batch_size,
                                     shuffle=False,
                                     num_workers=self.args.data_loader_workers,
                                     pin_memory=self.args.pin_memory,
                                     drop_last=True)

        ## target dataset train and validation
        self.target_dataset_train =\
            City_Dataset(args,
                       data_root_path=self.datasets_path['cityscapes']['data_root_path'],
                       list_path=self.datasets_path['cityscapes']['list_path'],
                       gt_path=self.datasets_path['cityscapes']['gt_path'],
                       split='train_style',
                       base_size=args.base_size,
                       crop_size=args.crop_size,
                       class_16=args.class_16)
        self.target_dataloader = data.DataLoader\
                                       (self.target_dataset_train,
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
                       base_size=args.base_size,
                       crop_size=args.crop_size,
                       class_16=args.class_16)

        self.target_val_dataloader = data.DataLoader(self.target_dataset_val,
                                                     batch_size=self.args.batch_size,
                                                     shuffle=False,
                                                     num_workers=self.args.data_loader_workers,
                                                     pin_memory=self.args.pin_memory,
                                                     drop_last=True)

        style_dataset = FlatFolderDataset(
            root="/data/Projects/MaxSquareLoss/imagenet_style/ambulance",
            transform=self.source_dataset_val.adain_transform())

        self.style_dataloader=iter(data.DataLoader(style_dataset,
                              batch_size=4,
                              shuffle=False,
                              num_workers=self.args.data_loader_workers,
                              pin_memory=self.args.pin_memory,
                              drop_last=True))

        self.batch_style = next(self.style_dataloader)
        self.dataloader.val_loader = self.target_val_dataloader
        self.dataloader.valid_iterations = (len(self.target_dataset_val) + self.args.batch_size) // self.args.batch_size

        self.network = Net()
        if torch.cuda.device_count()>=1:
            self.network = nn.DataParallel(self.network)
        self.network.train()
        if self.cuda:
            self.network.cuda()

        ## define target loss
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

        ## initially add network parameters into
        # allparams = self.params+[{'params':self.network.parameters(),'lr':self.args.lr}]
        #
        # self.all_optimizer = torch.optim.Adam(
        #      allparams,
        #      betas=(0.9, 0.99),
        #      weight_decay=self.args.weight_decay)
        if self.cuda:
            self.adain_optimizer = torch.optim.Adam(
                self.network.module.parameters(),
                lr=self.args.adain_lr)
        else:
            self.adain_optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.args.adain_lr)

        self.optimizer.zero_grad()
        self.adain_optimizer.zero_grad()

    def save_tensor_as_Image(self, tensor, path, filename, cnt, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
        if not os.path.exists(path):
            os.makedirs(path)

        from PIL import Image
        from utils.train_helper import make_grid
        image = tensor.cpu().clone()
        grid = make_grid(image, nrow=nrow, padding=padding, pad_value=pad_value,
                         normalize=normalize, range=range, scale_each=scale_each)
        # print('grid.max(),x.min(),x.shape,x[0,0]', grid.max(), grid.min(), grid.shape, grid[0, 0,])
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        # print('ndarr.max(),x.min(),x.shape,x[0,0]', ndarr.max(), ndarr.min(), ndarr.shape, ndarr[0, 0, 0])
        im = Image.fromarray(ndarr)
        outpath = os.path.join(path,filename)
        if cnt%10000 == 0:
            im.save(outpath)
        return im


    def train_target(self, pred):
        if isinstance(pred, tuple):
            pred_2 = pred[1]
            pred = pred[0]
            pred_P_2 = F.softmax(pred_2, dim=1)
        pred_P = F.softmax(pred, dim=1)

        if self.args.target_mode == "hard":
            label = torch.argmax(pred_P.detach(), dim=1)
            if self.args.multi: label_2 = torch.argmax(pred_P_2.detach(), dim=1)
        else:  ## 软标签，不要求求出最大
            label = pred_P
            if self.args.multi: label_2 = pred_P_2

        maxpred, argpred = torch.max(pred_P.detach(), dim=1)
        if self.args.multi: maxpred_2, argpred_2 = torch.max(pred_P_2.detach(), dim=1)

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
        # if self.args.checkpoint_dir is not None:
        #     self.args.pretrained_ckpt_file = os.path.join(self.args.checkpoint_dir, self.restore_id + 'best.pth')
            # self.load_checkpoint(self.args.checkpoint_dir)

        if not self.args.continue_training:
            self.best_MIou = 0
            self.best_iter = 0
            self.current_iter = 0
            self.current_epoch = 0
        else:
            self.load_checkpoint(os.path.join(self.args.checkpoint_dir, self.restore_id + 'final.pth'))
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

    def train_one_epoch(self,epoch=0):
        tqdm_epoch = tqdm(zip(self.source_dataloader, self.target_dataloader),#, self.source_dataloader_trans),
                          total=self.dataloader.num_iterations,
                          desc="Train Round-{}-Epoch-{}-total-{}". \
                          format(self.current_round,self.current_epoch + 1, self.epoch_num))
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
            # self.poly_lr_scheduler(optimizer=self.optimizer, init_lr=self.args.lr)
            self.adain_lr_scheduler(optimizer=self.adain_optimizer,
                                    iteration_count=self.current_iter,lr = args.adain_lr)

            ############################################
            # get source and target picture from ADAIN #
            ############################################
            ## source ##
            content, source_label_tf, source_id = batch_s
            # loss_s_source, loss_c_source, trans_source = self.network(content, self.batch_style)  #输出正确
            # self.save_tensor_as_Image(trans_source,path='/data/result/',filename='test311train.png',cnt=0)

            tmp = Image.open('/data/result/test311.png')
            tmp = self.source_dataset_train._img_transform(tmp)

            # torch.save(trans_source,'/data/result/trans_source.pt')
            trans_source = torch.load('/data/result/trans_source.pt')
            trans_source = trans_source.mul(255).add(0.5).clamp(0, 255)
            trans_source_tf= self.seg_transform(trans_source)

            cmp = torch.load('/data/result/train_trans_pic.pt')
            cmp_tf = torch.load('/data/result/train_trans_1.pt')

            ## target
            content, _, target_id = batch_t
            loss_s_target, loss_c_target, trans_target= self.network(content,self.batch_style)

            ##########################
            # source supervised loss #
            ##########################
            trans_source_tensor = self.seg_transform(trans_source)

            if self.cuda:
                x, y = Variable(trans_source_tensor).to(self.device), \
                       Variable(source_label_tf).to(device=self.device, dtype=torch.long)

            pred = self.model(x)
            self.train_source(pred, y)

            #####################
            # train with target #
            #####################
            trans_target_tensor = self.seg_transform(trans_target)

            if self.cuda:
                x = Variable(trans_target_tensor).to(self.device)

            pred = self.model(x)
            self.train_target(pred)

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.adain_optimizer.step()
            self.adain_optimizer.zero_grad()

            self.current_iter += 1
            # if self.current_iter == 1:
            #     break

        self.scheduler.step(self.loss_seg_value)
        self.writer.add_scalar('train_loss', self.loss_seg_value, self.current_epoch)
        tqdm.write("The average loss of train epoch-{}-:{}".format(self.current_epoch, self.loss_seg_value))
        self.writer.add_scalar('target_loss', self.loss_target_value, self.current_epoch)
        tqdm.write("The average target_loss of train epoch-{}-:{:.3f}".format(self.current_epoch, self.loss_target_value))

        if self.args.multi:
            self.writer.add_scalar('train_loss_2', self.loss_seg_value_2, self.current_epoch)
            tqdm.write("The average loss_2 of train epoch-{}-:{}"
                       .format(self.current_epoch, self.loss_seg_value_2))
            self.writer.add_scalar('target_loss_2', self.loss_target_value_2, self.current_epoch)
            tqdm.write("The average target_loss_2 of train epoch-{}-:{:.3f}"
                       .format(self.current_epoch, self.loss_target_value_2))
        tqdm_epoch.close()

        # eval on source domain
        self.validate_source()


def add_UDA_train_args(arg_parser):
    arg_parser.add_argument('--source_dataset', default='gta5', type=str,
                            choices=['gta5', 'synthia'],
                            help='source dataset choice')
    arg_parser.add_argument('--source_split', default='train', type=str,
                            help='source datasets split')
    arg_parser.add_argument('--init_round', type=int, default=0,
                            help='init_round')
    arg_parser.add_argument('--round_num', type=int, default=1,
                            help="num round")
    arg_parser.add_argument('--epoch_each_round', type=int, default=2,
                            help="epoch each round")
    arg_parser.add_argument('--target_mode', type=str, default="maxsquare",
                            choices=['maxsquare', 'IW_maxsquare', 'entropy', 'IW_entropy', 'hard'],
                            help="the loss function on target domain")
    arg_parser.add_argument('--lambda_target', type=float, default=1,
                            help="lambda of target loss")
    arg_parser.add_argument('--gamma', type=float, default=0,
                            help='parameter for scaled entorpy')
    arg_parser.add_argument('--IW_ratio', type=float, default=0.2,
                            help='the ratio of image-wise weighting factor')
    arg_parser.add_argument('--threshold', type=float, default=0.95,
                            help="threshold for Self-produced guidance")
    arg_parser.add_argument('--adain_lr',type=float, default=1e-4,
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

    agent = UDATrainer(args=args, train_id=train_id,
                       logger=logger)
    agent.main()
