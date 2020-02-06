import os

import sys
from pathlib import Path
import random

sys.path.append(os.path.abspath('.'))
from utils.eval import Eval
from utils.loss import *
from datasets.cityscapes_Dataset import City_Dataset, City_DataLoader, inv_preprocess, decode_labels
from datasets.gta5_Dataset import GTA5_DataLoader, GTA5_Dataset
from datasets.synthia_Dataset import SYNTHIA_Dataset

from tools.train_source import *


class UDATrainer(Trainer):
    def __init__(self, args, cuda=None, train_id="None", logger=None, \
                 datasets_path=None, styles_source=None, styles_target=None
                 ):
        super().__init__(args, cuda, train_id, logger)  # 调用父类的初始化，这样不用重写函数，同时初始化了应有的参数

        self.datasets_path = datasets_path
        self.styles_source = styles_source
        self.styles_target = styles_target
        ## source train loader

        source_trans_data_set = GTA5_Dataset(args,
                               data_root_path=self.datasets_path[self.styles_source[0]]['data_root_path'],
                               list_path=self.datasets_path[self.styles_source[0]]['list_path'],
                               gt_path=self.datasets_path[self.styles_source[0]]['gt_path'],
                               split='train',
                               base_size=args.base_size,
                               crop_size=args.crop_size)
        self.source_trans_dataloader = \
            data.DataLoader(source_trans_data_set,
                           batch_size=self.args.batch_size,
                           shuffle=True,
                           num_workers=self.args.data_loader_workers,
                           pin_memory=self.args.pin_memory,
                           drop_last=True)

        ## source validation loader
        source_trans_data_set = GTA5_Dataset(args,
                                       data_root_path=self.datasets_path[self.styles_source[0]]['data_root_path'],
                                       list_path=self.datasets_path[self.styles_source[0]]['list_path'],
                                       gt_path=self.datasets_path[self.styles_source[0]]['gt_path'],
                                       split='val',
                                       base_size=args.base_size,
                                       crop_size=args.crop_size)
        self.source_trans_val_dataloader = data.DataLoader(source_trans_data_set,
                                                           batch_size=self.args.batch_size,
                                                           shuffle=False,
                                                           num_workers=self.args.data_loader_workers,
                                                           pin_memory=self.args.pin_memory,
                                                           drop_last=True)

        ## target dataset train and validation
        target_trans_data_set =\
            City_Dataset(args,
                       data_root_path=self.datasets_path[self.styles_target[0]]['data_root_path'],
                       list_path=self.datasets_path[self.styles_target[0]]['list_path'],
                       gt_path=self.datasets_path[self.styles_target[0]]['gt_path'],
                       split='train',
                       base_size=args.target_base_size,
                       crop_size=args.target_crop_size,
                       class_16=args.class_16)
        self.target_trans_dataloader = data.DataLoader(target_trans_data_set,
                                                       batch_size=self.args.batch_size,
                                                       shuffle=True,
                                                       num_workers=self.args.data_loader_workers,
                                                       pin_memory=self.args.pin_memory,
                                                       drop_last=True)
        target_trans_data_set =             \
            City_Dataset(args,
                       data_root_path=self.datasets_path[self.styles_target[0]]['data_root_path'],
                       list_path=self.datasets_path[self.styles_target[0]]['list_path'],
                       gt_path=self.datasets_path[self.styles_target[0]]['gt_path'],
                       split='val',
                       base_size=args.target_base_size,
                       crop_size=args.target_crop_size,
                       class_16=args.class_16)
        self.target_val_dataloader = data.DataLoader(target_trans_data_set,
                                                     batch_size=self.args.batch_size,
                                                     shuffle=False,
                                                     num_workers=self.args.data_loader_workers,
                                                     pin_memory=self.args.pin_memory,
                                                     drop_last=True)

        self.dataloader.val_loader = self.target_val_dataloader

        self.dataloader.valid_iterations = (len(target_trans_data_set) + self.args.batch_size) // self.args.batch_size

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

    def save_shuffled_list(self):
        with open(os.path.join(self.datasets_path[self.styles_source[0]]['list_path'], "train.txt"), "r") as f:
            content = f.readlines()
            random.shuffle(content)

        with open(os.path.join(self.datasets_path[self.styles_source[0]]['list_path'], "train.txt"), "w+") as f:
            for line in content:
                f.write(line)

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
        # display args details
        self.logger.info("Global configuration as follows:")
        for key, val in vars(self.args).items():
            self.logger.info("{:16} {}".format(key, val))

        # load pretrained checkpoint
        if self.args.checkpoint_dir is not None:
            self.args.pretrained_ckpt_file = os.path.join(self.args.checkpoint_dir, self.restore_id + 'best.pth')
            self.load_checkpoint(self.args.pretrained_ckpt_file)

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
        print(self.args.iter_max, self.dataloader.num_iterations)

        # train
        # self.validate() # check image summary
        self.train_round()

        self.writer.close()

    def train_round(self):
        if "target_aug" in self.args.exp_tag:
            self.logger.info("#########target_aug_begin##############")
        for r in range(self.current_round, self.round_num):
            print("\n############## Begin {}/{} Round! #################\n".format(self.current_round + 1,
                                                                                   self.round_num))
            print("epoch_each_round:", self.args.epoch_each_round)

            self.epoch_num = self.current_epoch + (self.current_round + 1) * self.args.epoch_each_round

            # generate threshold
            self.threshold = self.args.threshold
            print("self.epoch_num", self.epoch_num)
            self.train()  ## it was using the method in the trainer

            self.current_round += 1

    def train_one_epoch(self,epoch=0):
        # self.save_shuffled_list()  # shuffle the id.txt every epoch

        tqdm_epoch = tqdm(zip(self.source_trans_dataloader, self.target_trans_dataloader),
                          total=self.dataloader.num_iterations,
                          desc="Train Round-{}-Epoch-{}-total-{}".\
                          format(self.current_round,self.current_epoch + 1, self.epoch_num))
        self.logger.info("Training one epoch... in the method defined by solve_gta5")
        self.Eval.reset()

        # Initialize your average meters
        self.loss_seg_value = 0
        # loss with self and init outside of epoch will accumulate for all the aux domains
        self.loss_target_value = 0
        self.loss_seg_value_2 = 0
        self.loss_target_value_2 = 0
        self.iter_num = self.dataloader.num_iterations

        # Set the model to be in training mode (for batchnorm and dropout)
        if self.args.freeze_bn:
            self.model.eval()
            self.logger.info("freeze batch normalization successfully!")
        else:
            self.model.train()

        batch_idx = 0  # iter in the data


        for batch_s, batch_t in tqdm_epoch:
            self.poly_lr_scheduler(optimizer=self.optimizer, init_lr=self.args.lr)
            # self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], self.current_iter)

            ##########################
            # source supervised loss #
            ##########################
            # train with source
            x, y, _ = batch_s
            if self.cuda:
                x, y = Variable(x).to(self.device), Variable(y).to(device=self.device, dtype=torch.long)

            pred = self.model(x)
            self.train_source(pred, y)

            #####################
            # train with target #
            #####################
            x, _, _ = batch_t
            if self.cuda:
                x = Variable(x).to(self.device)
            pred = self.model(x)

            self.train_target(pred)

            self.optimizer.step()
            self.optimizer.zero_grad()

            batch_idx += 1

            self.current_iter += 1

        self.writer.add_scalar('train_loss', self.loss_seg_value, self.current_epoch)
        tqdm.write("The average loss of train epoch-{}-:{}".format(self.current_epoch, self.loss_seg_value))
        self.writer.add_scalar('target_loss', self.loss_target_value, self.current_epoch)
        tqdm.write("The average target_loss of train epoch-{}-:{:.3f}".format(self.current_epoch, self.loss_target_value))
        if self.args.multi:
            self.writer.add_scalar('train_loss_2', self.loss_seg_value_2, self.current_epoch)
            self.writer.add_scalar('train_loss_2', self.loss_seg_value_2, self.current_epoch)
            tqdm.write("The average loss_2 of train epoch-{}-:{}".format(self.current_epoch, self.loss_seg_value_2))
            self.writer.add_scalar('target_loss_2', self.loss_target_value_2, self.current_epoch)
            tqdm.write(
                "The average target_loss_2 of train epoch-{}-:{:.3f}".format(self.current_epoch, self.loss_target_value_2))
        tqdm_epoch.close()

        # eval on source domain
        self.validate_source()
        self.logger.info("learning rate for epoch {} is  {}".
                         format(self.current_epoch, self.optimizer.param_groups[0]["lr"]))


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
    arg_parser.add_argument('--target_solo_epoch',type=int, default=0,
                            help='the epoch to train solely on target')
    return arg_parser


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    arg_parser = argparse.ArgumentParser()
    arg_parser = add_train_args(arg_parser)
    arg_parser = add_UDA_train_args(arg_parser)

    args = arg_parser.parse_args()
    args, _, logger = init_args(args)
    args.source_data_path = datasets_path[args.source_dataset]['data_root_path']
    args.source_list_path = datasets_path[args.source_dataset]['list_path']

    args.target_dataset = args.dataset

    train_id = str(args.source_dataset) + "2" + str(args.target_dataset) + "_" + args.target_mode
    styles_source, styles_target = [], []
    for f in Path("/data/Projects/ADVENT/data").glob("*"):
        if "GTA5_" in str(f):
            styles_source.append(f.stem)
        elif "Cityscapes_" in str(f):
            styles_target.append(f.stem)

    styles_source=['GTA5_ambulance_styleRetrain']
    styles_target = ['Cityscapes_ambulance_styleRetrain']

    logger.info("styles_souce,style_target", styles_source, styles_target)

    agent = UDATrainer(args=args, cuda=True, train_id=train_id,
                       logger=logger, datasets_path=datasets_path,
                       styles_source=styles_source, styles_target=styles_target)
    agent.main()