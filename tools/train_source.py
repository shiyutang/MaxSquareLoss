import os
import random
import logging
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from tqdm import tqdm
from math import ceil
import numpy as np
from distutils.version import LooseVersion
from tensorboardX import SummaryWriter
from torchvision import  transforms as ttransforms

import sys
import random
sys.path.append(os.path.abspath('.'))
from utils.eval import Eval
from utils.train_helper import get_model

from datasets.cityscapes_Dataset import City_Dataset, City_DataLoader, inv_preprocess, decode_labels
from datasets.gta5_Dataset import GTA5_DataLoader
from datasets.synthia_Dataset import SYNTHIA_DataLoader


datasets_path={
    'cityscapes': {'data_root_path': '/data/Projects/ADVENT/data/Cityscapes',
                   'list_path': '/data/Projects/ADVENT/data/Cityscapes/leftImg8bit',
                    'image_path':'/data/Projects/ADVENT/data/Cityscapes/leftImg8bit',
                    'gt_path': '/data/Projects/ADVENT/data/Cityscapes/gtFine'},
    'Cityscapes_accordion': {'data_root_path': '/data/Projects/ADVENT/data/Cityscapes_accordion',
                   'list_path': '/data/Projects/ADVENT/data/Cityscapes/leftImg8bit',
                   'image_path': '/data/Projects/ADVENT/data/Cityscapes_accordion/leftImg8bit',
                   'gt_path': '/data/Projects/ADVENT/data/Cityscapes/gtFine'},
    'Cityscapes_ambulance_styleRetrain': {'data_root_path': '/data/Projects/ADVENT/data/Cityscapes_ambulance_styleRetrain',
                   'list_path': '/data/Projects/ADVENT/data/Cityscapes/leftImg8bit',
                   'image_path': '/data/Projects/ADVENT/data/Cityscapes_ambulance_styleRetrain/leftImg8bit',
                   'gt_path': '/data/Projects/ADVENT/data/Cityscapes/gtFine'},
    'Cityscapes_ambulance_retrain_alpha0p5wts10':
        {'data_root_path': '/data/Projects/ADVENT/data/Cityscapes_ambulance_retrain_alpha0p5wts10',
                   'list_path': '/data/Projects/ADVENT/data/Cityscapes/leftImg8bit',
                   'image_path': '/data/Projects/ADVENT/data/Cityscapes_ambulance_retrain_alpha0p5wts10/leftImg8bit',
                   'gt_path': '/data/Projects/ADVENT/data/Cityscapes/gtFine'},
    'Cityscapes_ambulance_retrain_alpha1stylewt5':
        {'data_root_path': '/data/Projects/ADVENT/data/Cityscapes_ambulance_retrain_alpha1stylewt5',
         'list_path': '/data/Projects/ADVENT/data/Cityscapes/leftImg8bit',
         'image_path': '/data/Projects/ADVENT/data/Cityscapes_ambulance_retrain_alpha1stylewt5/leftImg8bit',
         'gt_path': '/data/Projects/ADVENT/data/Cityscapes/gtFine'},
    'Cityscapes_ambulance_gta5pcity_retrain_alpha1stylewt1':
        {'data_root_path': '/data/Projects/ADVENT/data/Cityscapes_ambulance_gta5pcity_retrain_alpha1stylewt1',
         'list_path': '/data/Projects/ADVENT/data/Cityscapes/leftImg8bit',
         'image_path': '/data/Projects/ADVENT/data/Cityscapes_ambulance_gta5pcity_retrain_alpha1stylewt1/leftImg8bit',
         'gt_path': '/data/Projects/ADVENT/data/Cityscapes/gtFine'},
    'Cityscapes_cityscapes_standard':
        {'data_root_path': '/data/Projects/ADVENT/data/Cityscapes_cityscapes_standard',
         'list_path': '/data/Projects/ADVENT/data/Cityscapes/leftImg8bit',
         'image_path': '/data/Projects/ADVENT/data/Cityscapes_cityscapes_standard/leftImg8bit',
         'gt_path': '/data/Projects/ADVENT/data/Cityscapes/gtFine'},
    'gta5': {'data_root_path': '/data/Projects/ADVENT/data/GTA5',
             'list_path': '/data/Projects/ADVENT/data/GTA5',
             'image_path':'/data/Projects/ADVENT/data/GTA5/images',
             'gt_path': '/data/Projects/ADVENT/data/GTA5/labels'},
    'GTA5_accordion': {'data_root_path': '/data/Projects/ADVENT/data/GTA5_accordion',
                       'list_path': '/data/Projects/ADVENT/data/GTA5',
                       'image_path':'/data/Projects/ADVENT/data/GTA5_accordion/images',
                       'gt_path': '/data/Projects/ADVENT/data/GTA5/labels'},
    'GTA5_ambulance_styleRetrain': {'data_root_path': '/data/Projects/ADVENT/data/GTA5_ambulance_styleRetrain',
                        'list_path': '/data/Projects/ADVENT/data/GTA5',
                       'image_path': '/data/Projects/ADVENT/data/GTA5_ambulance_styleRetrain/images',
                       'gt_path': '/data/Projects/ADVENT/data/GTA5/labels'},
    'GTA5_church': {'data_root_path': '/data/Projects/ADVENT/data/GTA5_church',
                       'list_path': '/data/Projects/ADVENT/data/GTA5',
                       'image_path': '/data/Projects/ADVENT/data/GTA5_church/images',
                       'gt_path': '/data/Projects/ADVENT/data/GTA5/labels'},
    'GTA5_elephant': {'data_root_path': '/data/Projects/ADVENT/data/GTA5_elephant',
                       'list_path': '/data/Projects/ADVENT/data/GTA5',
                       'image_path': '/data/Projects/ADVENT/data/GTA5_elephant/images',
                       'gt_path': '/data/Projects/ADVENT/data/GTA5/labels'},
    'GTA5_ambulance_gta5pcity_retrain_alpha1stylewt1':
        {'data_root_path': '/data/Projects/ADVENT/data/GTA5_ambulance_gta5pcity_retrain_alpha1stylewt1',
         'list_path': '/data/Projects/ADVENT/data/GTA5',
         'image_path': '/data/Projects/ADVENT/data/GTA5_ambulance_gta5pcity_retrain_alpha1stylewt1/images',
         'gt_path': '/data/Projects/ADVENT/data/GTA5/labels'},
    'GTA5_ambulance_gta5pcity_retrain_alpha1stylewt0p5':
        {'data_root_path': '/data/Projects/ADVENT/data/GTA5_ambulance_gta5pcity_retrain_alpha1stylewt0p5',
         'list_path': '/data/Projects/ADVENT/data/GTA5',
         'image_path': '/data/Projects/ADVENT/data/GTA5_ambulance_gta5pcity_retrain_alpha1stylewt0p5/images',
         'gt_path': '/data/Projects/ADVENT/data/GTA5/labels'},
    'GTA5_cityscapes_standard':
        {'data_root_path': '/data/Projects/ADVENT/data/GTA5_cityscapes_standard',
         'list_path': '/data/Projects/ADVENT/data/GTA5',
         'image_path': '/data/Projects/ADVENT/data/GTA5_cityscapes_standard/images',
         'gt_path': '/data/Projects/ADVENT/data/GTA5/labels'},

    'synthia': {'data_root_path': '/data/Projects/ADVENT/data/SYNTHIA', 'list_path': '/data/Projects/ADVENT/data/SYNTHIA/list',
                    'image_path':'/data/Projects/ADVENT/data/SYNTHIA/RGB',
                    'gt_path': '/data/Projects/ADVENT/data/GT/LABELS'},
    'NTHU': {'data_root_path': './datasets/NTHU_Datasets', 'list_path': './datasets/NTHU_list'}
    }

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

ITER_MAX = 5000

class Trainer():
    def __init__(self, args, cuda=None, train_id="None", logger=None):
        self.args = args
        self.datasets_path = datasets_path
        if torch.cuda.device_count()==1:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu # else 0,1,2,3
        self.cuda = cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.train_id = train_id
        self.restore_id = args.restore_id
        self.logger = logger

        self.current_MIoU = 0
        self.best_MIou = 0
        self.best_FWIou=0
        self.best_source_MIou = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.second_best_MIou = 0

        # set TensorboardX
        self.writer = SummaryWriter(self.args.save_dir)

        # Metric definition
        self.Eval = Eval(self.args.num_classes)

        # loss definition
        self.loss = nn.CrossEntropyLoss(weight=None, ignore_index= -1)
        self.loss.to(self.device) # loss 也需要去一个device

        # model
        self.model, self.params = get_model(self.args)
        if torch.cuda.device_count()>1:
            print("let us use {} GPUs".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)
        elif torch.cuda.device_count()==1:
            self.model = nn.DataParallel(self.model, device_ids=[0])

        self.model.to(self.device)

        # optimizer
        if self.args.optim == "SGD":
            self.optimizer = torch.optim.SGD(lr=self.args.lr,
                params=self.params,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optim == "Adam":
            self.optimizer = torch.optim.Adam(self.params, betas=(0.9, 0.99), weight_decay=self.args.weight_decay)

        # dataloader
        if self.args.dataset=="cityscapes" or 'Cityscapes' in self.args.dataset:
            self.dataloader = City_DataLoader(self.args, datasets_path=datasets_path[self.args.dataset])
        elif self.args.dataset=="gta5" or 'GTA5' in self.args.dataset:
            self.dataloader = GTA5_DataLoader(self.args, datasets_path=datasets_path[self.args.dataset])
        elif self.args.dataset=='synthia':
            self.dataloader = SYNTHIA_DataLoader(self.args,datasets_path['synthia'])
        self.dataloader.num_iterations = min(self.dataloader.num_iterations, ITER_MAX)
        print(self.args.iter_max, self.dataloader.num_iterations)
        self.epoch_num = ceil(self.args.iter_max / self.dataloader.num_iterations) if self.args.iter_stop is None else \
                            ceil(self.args.iter_stop / self.dataloader.num_iterations)

        # self.logger.info('I am loading training data and validation data from {}'.\
        #                  format(datasets_path[self.args.dataset]['data_root_path']))

    def main(self):
        # display args details
        # self.logger.info("Global configuration as follows:")
        # for key, val in vars(self.args).items():
        #     self.logger.info("{:16} {}".format(key, val))


        # load pretrained checkpoint
        if self.args.checkpoint_dir is not None:  # restore from trained GTA
            self.args.pretrained_ckpt_file = os.path.join(self.args.checkpoint_dir, self.restore_id + 'best.pth')
            self.load_checkpoint(self.args.pretrained_ckpt_file)

        if self.args.continue_training:
            self.load_checkpoint(os.path.join(self.args.checkpoint_dir, self.restore_id + 'best.pth'))
            self.best_iter = self.current_iter         # the best iteration for target
            self.best_source_iter = self.current_iter  # the best iteration for source
        else:
            self.current_epoch = 0
        # train
        self.train()

        self.writer.close()

    def train(self):
        # self.validate() # check image summary

        for epoch in tqdm(range(self.current_epoch, self.epoch_num),
                          desc="Total {} epochs".format(self.epoch_num)):
            self.train_one_epoch(epoch)

            # validate
            PA, MPA, MIoU, FWIoU = self.validate()
            self.writer.add_scalar('PA', PA, self.current_epoch)
            self.writer.add_scalar('MPA', MPA, self.current_epoch)
            self.writer.add_scalar('MIoU', MIoU, self.current_epoch)
            self.writer.add_scalar('FWIoU', FWIoU, self.current_epoch)

            self.current_MIoU = MIoU
            is_best = MIoU > self.best_MIou
            if is_best:
                self.best_MIou = MIoU
                self.best_iter = self.current_iter
                self.logger.info("=>saving a new best checkpoint...")
                self.save_checkpoint(self.train_id+'best.pth')
            else:
                self.logger.info("=> The MIoU of val does't improve.")
                self.logger.info("=> The best MIoU of val is {} at {}".format(self.best_MIou, self.best_iter))
            
            self.current_epoch += 1

        self.logger.info("=>best_MIou {} at {}".format(self.best_MIou, self.best_iter))
        self.logger.info("=>saving the final checkpoint to " + os.path.join(self.args.save_dir, self.train_id+'final.pth'))
        self.save_checkpoint(self.train_id+'final.pth')

    def train_one_epoch(self,epoch=None):
        tqdm_epoch = tqdm(self.dataloader.data_loader, total=self.dataloader.num_iterations,
                          desc="Train Epoch-{}-total-{}".format(self.current_epoch+1, self.epoch_num))
        self.logger.info("Training one epoch...")
        self.Eval.reset()  # set confusion matrix to zeros

        train_loss = []
        loss_seg_value_2 = 0
        iter_num = self.dataloader.num_iterations
        
        if self.args.freeze_bn:
            self.model.eval()
            self.logger.info("freeze bacth normalization successfully!")
        else:
            self.model.train()
        # Initialize your average meters

        batch_idx = 0
        for x, y, _ in tqdm_epoch:
            self.poly_lr_scheduler(
                optimizer=self.optimizer,
                init_lr=self.args.lr,
                iter=self.current_iter,
                max_iter=self.args.iter_max,
                power=self.args.poly_power,
            )
            if self.args.iter_stop is not None and self.current_iter >= self.args.iter_stop:
                self.logger.info("iteration arrive {}(early stop)/{}(total step)!".format(self.args.iter_stop, self.args.iter_max))
                break
            if self.current_iter >= self.args.iter_max:
                self.logger.info("iteration arrive {}!".format(self.args.iter_max))
                break
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], self.current_iter)

            if self.cuda:
                x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)
            y = torch.squeeze(y, 1)
            self.optimizer.zero_grad()

            # model
            pred = self.model(x)
            if isinstance(pred, tuple):
                pred_2 = pred[1]
                pred = pred[0]
            
            # loss
            cur_loss = self.loss(pred, y)
            
            if self.args.multi:
                loss_2 = self.args.lambda_seg * self.loss(pred_2, y)
                cur_loss += loss_2
                loss_seg_value_2 += loss_2.cpu().item() / iter_num

            # optimizer
            cur_loss.backward()
            self.optimizer.step()

            train_loss.append(cur_loss.item())

            if batch_idx % 1000 == 0:
                if self.args.multi:
                    self.logger.info("The train loss of epoch{}-batch-{}:{};{}".format(self.current_epoch,
                                                                            batch_idx, cur_loss.item(), loss_2.item()))
                else:
                    self.logger.info("The train loss of epoch{}-batch-{}:{}".format(self.current_epoch,
                                                                            batch_idx, cur_loss.item()))
                
            batch_idx += 1

            self.current_iter += 1

            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            pred = pred.data.cpu().numpy()
            label = y.cpu().numpy()
            argpred = np.argmax(pred, axis=1)
            self.Eval.add_batch(label, argpred)

            if batch_idx==self.dataloader.num_iterations:
                break
        
        self.log_one_train_epoch(x, label, argpred, train_loss)
        tqdm_epoch.close()

    def log_one_train_epoch(self, x, label, argpred, train_loss):
        #show train image on tensorboard
        images_inv = inv_preprocess(x.clone().cpu(), self.args.show_num_images, numpy_transform=self.args.numpy_transform)
        labels_colors = decode_labels(label, self.args.show_num_images)
        preds_colors = decode_labels(argpred, self.args.show_num_images)
        for index, (img, lab, color_pred) in enumerate(zip(images_inv, labels_colors, preds_colors)):
            self.writer.add_image('train/'+ str(index)+'/Images', img, self.current_epoch)
            self.writer.add_image('train/'+ str(index)+'/Labels', lab, self.current_epoch)
            self.writer.add_image('train/'+ str(index)+'/preds', color_pred, self.current_epoch)

        if self.args.class_16:
            PA = self.Eval.Pixel_Accuracy()
            MPA_16, MPA = self.Eval.Mean_Pixel_Accuracy()
            MIoU_16, MIoU = self.Eval.Mean_Intersection_over_Union()
            FWIoU_16, FWIoU = self.Eval.Frequency_Weighted_Intersection_over_Union()
        else:
            PA = self.Eval.Pixel_Accuracy()
            MPA = self.Eval.Mean_Pixel_Accuracy()
            MIoU = self.Eval.Mean_Intersection_over_Union()
            FWIoU = self.Eval.Frequency_Weighted_Intersection_over_Union()

        self.logger.info('\nEpoch:{}, train PA1:{}, MPA1:{}, MIoU1:{}, FWIoU1:{}'.format(self.current_epoch, PA, MPA,
                                                                                       MIoU, FWIoU))
        self.writer.add_scalar('train_PA', PA, self.current_epoch)
        self.writer.add_scalar('train_MPA', MPA, self.current_epoch)
        self.writer.add_scalar('train_MIoU', MIoU, self.current_epoch)
        self.writer.add_scalar('train_FWIoU', FWIoU, self.current_epoch)

        tr_loss = sum(train_loss)/len(train_loss) if isinstance(train_loss, list) else train_loss
        self.writer.add_scalar('train_loss', tr_loss, self.current_epoch)
        tqdm.write("The average loss of train epoch-{}-:{}".format(self.current_epoch, tr_loss))

    def seg_transform(self,tensor):
        trans_source_tensor = tensor.mul(255).add(0.5).clamp(0, 255)

        d = torch.Tensor([122.67891434, 104.00698793, 116.66876762]).reshape(-1, 1, 1).expand(-1, 512, 1024).cuda()
        trans_source_tensor = trans_source_tensor.squeeze(0) - d
        r = trans_source_tensor[0, :, :]
        g = trans_source_tensor[1, :, :]
        b = trans_source_tensor[2, :, :]
        result = torch.stack([g, b, r], dim=0).unsqueeze(0)

        # if self.current_iter == 0:
        #     self.std, self.mean = torch.std_mean(trans_source,[1,2])
        # ttransforms.Normalize(self.mean, self.std, inplace=True)(trans_source)
        # trans_source_tensor = trans_source.unsqueeze(0)
        return result

    def validate(self, mode='val'):
        self.logger.info('\nvalidating one epoch...')
        self.Eval.reset()
        with torch.no_grad():
            tqdm_batch = tqdm(self.dataloader.val_loader, total=self.dataloader.valid_iterations,
                              desc="Val Epoch-{}-".format(self.current_epoch + 1))
            if mode == 'val':
                self.model.eval()
            for x, y, id in tqdm_batch:
                _,_,x = self.network(x,self.batch_style)
                x = self.seg_transform(x)

                if self.cuda:
                    x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)
                # model
                pred = self.model(x)                   
                if isinstance(pred, tuple):
                    pred_2 = pred[1]
                    pred = pred[0]
                y = torch.squeeze(y, 1)

                pred = pred.data.cpu().numpy()
                label = y.cpu().numpy()
                argpred = np.argmax(pred, axis=1)

                self.Eval.add_batch(label, argpred)

            #show val result on tensorboard
            images_inv = inv_preprocess(x.clone().cpu(), self.args.show_num_images, numpy_transform=self.args.numpy_transform)
            labels_colors = decode_labels(label, self.args.show_num_images)
            preds_colors = decode_labels(argpred, self.args.show_num_images)
            for index, (img, lab, color_pred) in enumerate(zip(images_inv, labels_colors, preds_colors)):
                self.writer.add_image(str(index)+'/Images', img, self.current_epoch)
                self.writer.add_image(str(index)+'/Labels', lab, self.current_epoch)
                self.writer.add_image(str(index)+'/preds', color_pred, self.current_epoch)

            if self.args.class_16:
                def val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA_16, MPA_13 = Eval.Mean_Pixel_Accuracy()
                    MIoU_16, MIoU_13 = Eval.Mean_Intersection_over_Union()
                    FWIoU_16, FWIoU_13 = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC_16, PC_13 = Eval.Mean_Precision()
                    print("########## Eval{} ############".format(name))

                    self.logger.info('\nEpoch:{:.3f}, {} PA:{:.3f}, MPA_16:{:.3f}, MIoU_16:{:.3f}, FWIoU_16:{:.3f}, PC_16:{:.3f}'.format(self.current_epoch, name, PA, MPA_16,
                                                                                                MIoU_16, FWIoU_16, PC_16))
                    self.logger.info('\nEpoch:{:.3f}, {} PA:{:.3f}, MPA_13:{:.3f}, MIoU_13:{:.3f}, FWIoU_13:{:.3f}, PC_13:{:.3f}'.format(self.current_epoch, name, PA, MPA_13,
                                                                                                MIoU_13, FWIoU_13, PC_13))
                    self.writer.add_scalar('PA'+name, PA, self.current_epoch)
                    self.writer.add_scalar('MPA_16'+name, MPA_16, self.current_epoch)
                    self.writer.add_scalar('MIoU_16'+name, MIoU_16, self.current_epoch)
                    self.writer.add_scalar('FWIoU_16'+name, FWIoU_16, self.current_epoch)
                    self.writer.add_scalar('MPA_13'+name, MPA_13, self.current_epoch)
                    self.writer.add_scalar('MIoU_13'+name, MIoU_13, self.current_epoch)
                    self.writer.add_scalar('FWIoU_13'+name, FWIoU_13, self.current_epoch)
                    return PA, MPA_13, MIoU_13, FWIoU_13
            else:
                def val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA = Eval.Mean_Pixel_Accuracy()
                    MIoU = Eval.Mean_Intersection_over_Union()
                    FWIoU = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC = Eval.Mean_Precision()
                    print("########## Eval{} ############".format(name))

                    self.logger.info('\nEpoch:{:.3f}, {} PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(self.current_epoch, name, PA, MPA,
                                                                                                MIoU, FWIoU, PC))
                    self.writer.add_scalar('PA'+name, PA, self.current_epoch)
                    self.writer.add_scalar('MPA'+name, MPA, self.current_epoch)
                    self.writer.add_scalar('MIoU'+name, MIoU, self.current_epoch)
                    self.writer.add_scalar('FWIoU'+name, FWIoU, self.current_epoch)
                    return PA, MPA, MIoU, FWIoU

            PA, MPA, MIoU, FWIoU = val_info(self.Eval, "")
            tqdm_batch.close()

        return PA, MPA, MIoU, FWIoU

    def validate_source(self):
        self.logger.info('\nvalidating source domain...')
        self.Eval.reset()
        with torch.no_grad():
            tqdm_batch = tqdm(self.source_dataloader_val, total=self.dataloader.valid_iterations,
                              desc="Source Val Epoch-{}-".format(self.current_epoch + 1))
            self.model.eval()
            i = 0
            for x, y, id in tqdm_batch:
                _,_,x = self.network(x,self.batch_style)
                x = self.seg_transform(x)

                if self.cuda:
                    x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)
                # model
                pred = self.model(x)

                if isinstance(pred, tuple):
                    pred_2 = pred[1]
                    pred = pred[0]
                    pred_P = F.softmax(pred, dim=1)
                    pred_P_2 = F.softmax(pred_2, dim=1)
                y = torch.squeeze(y, 1)
                pred = pred.data.cpu().numpy()
                label = y.cpu().numpy()
                argpred = np.argmax(pred, axis=1)

                self.Eval.add_batch(label, argpred)
                i += 1
                if i == self.dataloader.valid_iterations:
                    break

            #show val result on tensorboard
            images_inv = inv_preprocess(x.clone().cpu(), self.args.show_num_images, numpy_transform=self.args.numpy_transform)
            labels_colors = decode_labels(label, self.args.show_num_images)
            preds_colors = decode_labels(argpred, self.args.show_num_images)
            for index, (img, lab, color_pred) in enumerate(zip(images_inv, labels_colors, preds_colors)):
                self.writer.add_image('source_eval/'+str(index)+'/Images', img, self.current_epoch)
                self.writer.add_image('source_eval/'+str(index)+'/Labels', lab, self.current_epoch)
                self.writer.add_image('source_eval/'+str(index)+'/preds', color_pred, self.current_epoch)

            if self.args.class_16:
                def source_val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA_16, MPA_13 = Eval.Mean_Pixel_Accuracy()
                    MIoU_16, MIoU_13 = Eval.Mean_Intersection_over_Union()
                    FWIoU_16, FWIoU_13 = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC_16, PC_13 = Eval.Mean_Precision()
                    print("########## Source Eval{} ############".format(name))

                    self.logger.info('\nEpoch:{:.3f}, source {} PA:{:.3f}, MPA_16:{:.3f}, MIoU_16:{:.3f}, FWIoU_16:{:.3f}, PC_16:{:.3f}'.format(self.current_epoch, name, PA, MPA_16,
                                                                                                MIoU_16, FWIoU_16, PC_16))
                    self.logger.info('\nEpoch:{:.3f}, source {} PA:{:.3f}, MPA_13:{:.3f}, MIoU_13:{:.3f}, FWIoU_13:{:.3f}, PC_13:{:.3f}'.format(self.current_epoch, name, PA, MPA_13,
                                                                                                MIoU_13, FWIoU_13, PC_13))
                    self.writer.add_scalar('source_PA'+name, PA, self.current_epoch)
                    self.writer.add_scalar('source_MPA_16'+name, MPA_16, self.current_epoch)
                    self.writer.add_scalar('source_MIoU_16'+name, MIoU_16, self.current_epoch)
                    self.writer.add_scalar('source_FWIoU_16'+name, FWIoU_16, self.current_epoch)
                    self.writer.add_scalar('source_MPA_13'+name, MPA_13, self.current_epoch)
                    self.writer.add_scalar('source_MIoU_13'+name, MIoU_13, self.current_epoch)
                    self.writer.add_scalar('source_FWIoU_13'+name, FWIoU_13, self.current_epoch)
                    return PA, MPA_13, MIoU_13, FWIoU_13
            else:
                def source_val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA = Eval.Mean_Pixel_Accuracy()
                    MIoU = Eval.Mean_Intersection_over_Union()
                    FWIoU = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC = Eval.Mean_Precision()

                    self.writer.add_scalar('source_PA'+name, PA, self.current_epoch)
                    self.writer.add_scalar('source_MPA'+name, MPA, self.current_epoch)
                    self.writer.add_scalar('source_MIoU'+name, MIoU, self.current_epoch)
                    self.writer.add_scalar('source_FWIoU'+name, FWIoU, self.current_epoch)
                    print("########## Source Eval{} ############".format(name))

                    self.logger.info('\nEpoch:{:.3f}, source {} PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(self.current_epoch, name, PA, MPA,
                                                                                                MIoU, FWIoU, PC))
                    return PA, MPA, MIoU, FWIoU
        
            PA, MPA, MIoU, FWIoU = source_val_info(self.Eval, "")
            tqdm_batch.close()

        is_best = MIoU > self.best_source_MIou
        if is_best:
            self.best_source_MIou = MIoU
            self.best_source_iter = self.current_iter
            self.logger.info("=>saving a new best source checkpoint...")
            self.save_checkpoint(self.train_id+'source_best.pth')
        else:
            self.logger.info("=> The source MIoU of val does't improve.")
            self.logger.info("=> The best source MIoU of val is {} at {}".format(self.best_source_MIou, self.best_source_iter))

        return PA, MPA, MIoU, FWIoU

    def save_checkpoint(self, filename=None):
        """
        Save checkpoint if a new best is achieved
        :param state:
        :param is_best:
        :param filepath:
        :return:
        """
        filename = os.path.join(self.args.save_dir, filename)
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_MIou':self.best_MIou
        }
        if self.network:
            state['network'] = self.network.state_dict(),
        torch.save(state, filename)

    def load_checkpoint(self, filename):
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            if self.cuda:
                checkpoint = torch.load(filename)
            else:
                checkpoint = torch.load(filename,map_location=torch.device('cpu'))

            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.module.load_state_dict(checkpoint)
            self.logger.info("Checkpoint loaded successfully from "+filename)

            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.current_epoch = checkpoint["epoch"]
                self.current_iter = checkpoint["iteration"]
                self.best_MIou = checkpoint["best_MIou"]

        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.args.checkpoint_dir))
            self.logger.info("**First time to train**")

    def poly_lr_scheduler(self, optimizer, init_lr=None, iter=None, 
                            max_iter=None, power=None):
        init_lr = self.args.lr if init_lr is None else init_lr
        iter = self.current_iter if iter is None else iter
        max_iter = self.args.iter_max if max_iter is None else max_iter
        power = self.args.poly_power if power is None else power
        new_lr = init_lr * (1 - float(iter) / max_iter) ** power
        optimizer.param_groups[0]["lr"] = new_lr
        if len(optimizer.param_groups) == 2:
            optimizer.param_groups[1]["lr"] = 10 * new_lr
        if len(optimizer.param_groups) == 3:
            optimizer.param_groups[2]["lr"] = new_lr

    def adain_lr_scheduler(self, optimizer, iteration_count, lr):
        """Imitating the original implementation"""
        lr = lr / (1.0 + 5e-5 * iteration_count)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def add_train_args(arg_parser):
    # Path related arguments
    arg_parser.add_argument('--data_root_path', type=str, default=None,
                            help="the root path of dataset")
    arg_parser.add_argument('--list_path', type=str, default=None,
                            help="the root path of dataset")
    arg_parser.add_argument('--checkpoint_dir', default="./log/train",
                            help="the path of ckpt file")
    arg_parser.add_argument("--save_dir",default="./log/train",
                            help="the path that you want to save all the output")
    arg_parser.add_argument("--restore_id",type=str, default="add_multi",
                            help="the id that help find load model")

    # Model related arguments
    arg_parser.add_argument('--backbone', default='deeplabv2_multi',
                            help="backbone of encoder")
    arg_parser.add_argument('--bn_momentum', type=float, default=0.1,
                            help="batch normalization momentum")
    arg_parser.add_argument('--imagenet_pretrained', type=str2bool, default=True,
                            help="whether apply imagenet pretrained weights")
    arg_parser.add_argument('--pretrained_ckpt_file', type=str, default=None,
                            help="whether apply pretrained checkpoint")
    arg_parser.add_argument('--continue_training', type=str2bool, default=False,
                            help="whether to continue training ")
    arg_parser.add_argument('--show_num_images', type=int, default=2,
                        help="show how many images during validate")

    # train related arguments
    arg_parser.add_argument('--seed', default=12345, type=int,
                            help='random seed')
    arg_parser.add_argument('--gpu', type=str, default="0",
                            help=" the num of gpu")
    arg_parser.add_argument("--batch_size",default=1,type=int,
                            help="directly set batch size")
    arg_parser.add_argument("--exp_tag",type=str,default="test",
                            help="Set tag for each experiment")

    # dataset related arguments
    arg_parser.add_argument('--dataset', default='cityscapes', type=str,
                            help='dataset choice')
    arg_parser.add_argument('--base_size', default="1024,512", type=str, # for random crop
                            help='crop size of image')
    arg_parser.add_argument('--crop_size', default="1024,512", type=str,
                            help='base size of image')
    arg_parser.add_argument('--target_base_size', default="1024,512", type=str, # for random crop
                            help='crop size of target image')
    arg_parser.add_argument('--target_crop_size', default="1024,512", type=str,
                            help='base size of target image')
    arg_parser.add_argument('--num_classes', default=19, type=int,
                            help='num class of mask')
    arg_parser.add_argument('--data_loader_workers', default=1, type=int,
                            help='num_workers of Dataloader')
    arg_parser.add_argument('--pin_memory', default=2, type=int,
                            help='pin_memory of Dataloader')
    arg_parser.add_argument('--split', type=str, default='train',
                            help="choose from train/val/test/trainval/all")
    arg_parser.add_argument('--random_mirror', default=True, type=str2bool,
                            help='add random_mirror')
    arg_parser.add_argument('--random_crop', default=False, type=str2bool,
                        help='add random_crop')
    arg_parser.add_argument('--resize', default=True, type=str2bool,
                        help='resize')
    arg_parser.add_argument('--gaussian_blur', default=False, type=str2bool,
                        help='add gaussian_blur')
    arg_parser.add_argument('--numpy_transform', default=True, type=str2bool,
                        help='transform pic using numpy with Means and BGR CHW')

    # optimization related arguments

    arg_parser.add_argument('--freeze_bn', type=str2bool, default=False,
                            help="whether freeze BatchNormalization")
    arg_parser.add_argument('--optim', default="SGD", type=str,
                            help='optimizer')
    arg_parser.add_argument('--momentum', type=float, default=0.9)
    arg_parser.add_argument('--weight_decay', type=float, default=5e-4)

    arg_parser.add_argument('--lr', type=float, default=2.5e-4,
                            help="init learning rate ")
    arg_parser.add_argument('--iter_max', type=int, default=200000,
                            help="the maxinum of iteration")
    arg_parser.add_argument('--iter_stop', type=int, default=80000,
                            help="the early stop step")
    arg_parser.add_argument('--poly_power', type=float, default=0.95,
                            help="poly_power")

    # multi-level output

    arg_parser.add_argument('--multi', default=False, type=str2bool,
                        help='output model middle feature')
    arg_parser.add_argument('--lambda_seg', type=float, default=0.1,
                        help="lambda_seg of middle output")
    return arg_parser

def init_args(args):
    # if torch.cuda.device_count()==1:
    #     args.batch_size = 1

    print("batch size: ", args.batch_size)

    train_id = args.exp_tag
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    crop_size = args.crop_size.split(',')
    base_size = args.base_size.split(',')
    if len(crop_size)==1:
        args.crop_size = int(crop_size[0])
        args.base_size = int(base_size[0])
    else:
        args.crop_size = (int(crop_size[0]), int(crop_size[1]))
        args.base_size = (int(base_size[0]), int(base_size[1]))

    target_crop_size = args.target_crop_size.split(',')
    target_base_size = args.target_base_size.split(',')
    if len(target_crop_size)==1:
        args.target_crop_size = int(target_crop_size[0])
        args.target_base_size = int(target_base_size[0])
    else:
        args.target_crop_size = (int(target_crop_size[0]), int(target_crop_size[1]))
        args.target_base_size = (int(target_base_size[0]), int(target_base_size[1]))

    if args.data_root_path is None:
        args.data_root_path = datasets_path[args.dataset]['data_root_path']
        args.list_path = datasets_path[args.dataset]['list_path']
        args.image_filepath = datasets_path[args.dataset]['image_path']
        args.gt_filepath = datasets_path[args.dataset]['gt_path']
    
    args.class_16 = True if args.num_classes == 16 else False
    args.class_13 = True if args.num_classes == 13 else False

    # logger configure
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.save_dir, train_id+'_train_log.txt'))
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    #set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=True

    return args, train_id, logger
    
if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    arg_parser = argparse.ArgumentParser()
    arg_parser = add_train_args(arg_parser)

    args = arg_parser.parse_args()
    args, train_id, logger = init_args(args)

    agent = Trainer(args=args, cuda=True, train_id=train_id, logger=logger)
    agent.main()