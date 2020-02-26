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
import copy
from PIL import Image

from tools.train_source import *

datasets_path={
    'cityscapes': {'data_root_path': '/data/Projects/ADVENT/data/Cityscapes',
                   'list_path': '/data/Projects/ADVENT/data/Cityscapes/leftImg8bit',
                    'image_path':'/data/Projects/ADVENT/data/Cityscapes/leftImg8bit',
                    'gt_path': '/data/Projects/ADVENT/data/Cityscapes/gtFine'},
    'Cityscapes_ambulance_styleRetrain': {'data_root_path': '/data/Projects/ADVENT/data/Cityscapes_ambulance_styleRetrain',
                   'list_path': '/data/Projects/ADVENT/data/Cityscapes/leftImg8bit',
                   'image_path': '/data/Projects/ADVENT/data/Cityscapes_ambulance_styleRetrain/leftImg8bit',
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
    'Cityscapes_ambulance_gta5pcity_retrain_alpha1wts0p5':
        {'data_root_path': '/data/Projects/ADVENT/data/Cityscapes_ambulance_gta5pcity_retrain_alpha1wts0p5',
         'list_path': '/data/Projects/ADVENT/data/Cityscapes/leftImg8bit',
         'image_path': '/data/Projects/ADVENT/data/Cityscapes_ambulance_gta5pcity_retrain_alpha1wts0p5/leftImg8bit',
         'gt_path': '/data/Projects/ADVENT/data/Cityscapes/gtFine'},
    'gta5': {'data_root_path': '/data/Projects/ADVENT/data/GTA5', 'list_path': '/data/Projects/ADVENT/data/GTA5',
                    'image_path':'/data/Projects/ADVENT/data/GTA5/images',
                    'gt_path': '/data/Projects/ADVENT/data/GTA5/labels'},
    'GTA5_ambulance_styleRetrain': {'data_root_path': '/data/Projects/ADVENT/data/GTA5_ambulance_styleRetrain',
                        'list_path': '/data/Projects/ADVENT/data/GTA5',
                       'image_path': '/data/Projects/ADVENT/data/GTA5_ambulance_styleRetrain/images',
                       'gt_path': '/data/Projects/ADVENT/data/GTA5/labels'},
    'GTA5_ambulance_retrain_alpha1stylewt5':
                {'data_root_path': '/data/Projects/ADVENT/data/GTA5_ambulance_retrain_alpha1stylewt5',
                       'list_path': '/data/Projects/ADVENT/data/GTA5',
                       'image_path': '/data/Projects/ADVENT/data/GTA5_ambulance_retrain_alpha1stylewt5/images',
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
    'synthia': {'data_root_path': '/data/Projects/ADVENT/data/SYNTHIA', 'list_path': '/data/Projects/ADVENT/data/SYNTHIA/list',
                    'image_path':'/data/Projects/ADVENT/data/SYNTHIA/RGB',
                    'gt_path': '/data/Projects/ADVENT/data/GT/LABELS'},
    'NTHU': {'data_root_path': './datasets/NTHU_Datasets', 'list_path': './datasets/NTHU_list'}
    }



class resultEvaluater(object):
    def __init__(self, args, cuda=None, logger=None, \
                 datasets_path=None, styles_source=None, styles_target=None
                 ):
        self.args = args
        if torch.cuda.device_count()==1:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        self.cuda = cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.restore_id = args.restore_id
        self.logger = logger
        self.datasets_path = datasets_path
        self.styles_source = styles_source
        self.styles_target = styles_target
        self.ignore_index = -1
        self.current_round = self.args.init_round
        self.resultOutPath = os.path.join(self.args.save_dir,'savePics')
        if not os.path.exists(self.resultOutPath):
            os.mkdir(self.resultOutPath)

        # Metric definition
        self.Eval = Eval(self.args.num_classes)
        self.totalEval = Eval(self.args.num_classes)

        # model
        self.model, params = get_model(self.args)
        if torch.cuda.device_count()>1:
            print("let us use {} GPUs".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)
        elif torch.cuda.device_count()==1:
            self.model = nn.DataParallel(self.model, device_ids=[0])

        self.model.to(self.device)

        ## source train loader
        if 'GTA5' in self.styles_source[0]:
            print('build source_train',self.styles_source)
            source_trans_data_set_train = GTA5_Dataset(args,
                                   data_root_path=self.datasets_path[self.styles_source[0]]['data_root_path'],
                                   list_path=self.datasets_path[self.styles_source[0]]['list_path'],
                                   gt_path=self.datasets_path[self.styles_source[0]]['gt_path'],
                                   split='train',
                                   base_size=args.base_size,
                                   crop_size=args.crop_size)

            ## source validation loader
            source_trans_data_set_val = GTA5_Dataset(args,
                                           data_root_path=self.datasets_path[self.styles_source[0]]['data_root_path'],
                                           list_path=self.datasets_path[self.styles_source[0]]['list_path'],
                                           gt_path=self.datasets_path[self.styles_source[0]]['gt_path'],
                                           split='test',
                                           base_size=args.base_size,
                                           crop_size=args.crop_size)

        elif 'SYNTHIA' in self.styles_source[0]:
            source_trans_data_set_train = SYNTHIA_Dataset(args,
                                   data_root_path=self.datasets_path[self.styles_source[0]]['data_root_path'],
                                   list_path=self.datasets_path[self.styles_source[0]]['list_path'],
                                   split='train',
                                   base_size=args.base_size,
                                   crop_size=args.crop_size)


            ## source validation loader
            source_trans_data_set_val = SYNTHIA_Dataset(args,
                                           data_root_path=self.datasets_path[self.styles_source[0]]['data_root_path'],
                                           list_path=self.datasets_path[self.styles_source[0]]['list_path'],
                                           split='val',
                                           base_size=args.base_size,
                                           crop_size=args.crop_size)

        self.source_trans_dataloader = \
            data.DataLoader(source_trans_data_set_train,
                           batch_size=self.args.batch_size,
                           shuffle=True,
                           num_workers=self.args.data_loader_workers,
                           pin_memory=self.args.pin_memory,
                           drop_last=True)

        self.source_trans_val_dataloader = data.DataLoader(source_trans_data_set_val,
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

        self.val_loader = self.source_trans_val_dataloader

        self.valid_iterations = (len(source_trans_data_set_val) + self.args.batch_size) // self.args.batch_size

    def load_checkpoint(self, filename):
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.module.load_state_dict(checkpoint)
            self.logger.info("Checkpoint loaded successfully from "+filename)

            if "optimizer" in checkpoint:
                self.current_epoch = checkpoint["epoch"]
                self.current_iter = checkpoint["iteration"]
                self.best_MIou = checkpoint["best_MIou"]

        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.args.checkpoint_dir))
            self.logger.info("**First time to train**")


    def main(self):
        # load pretrained checkpoint
        if self.args.checkpoint_dir is not None:
            self.args.pretrained_ckpt_file = os.path.join(self.args.checkpoint_dir, self.restore_id + 'best.pth')
            self.load_checkpoint(self.args.pretrained_ckpt_file)

        self.getvalResult(self.resultOutPath,0) # check image summary

    def getvalResult(self, outputPath, threshold):
        self.logger.info('\nget result one epoch...')
        self.Eval.reset()
        self.totalEval.reset()

        def val_info(Eval):
            PA = Eval.Pixel_Accuracy()
            MPA = Eval.Mean_Pixel_Accuracy()
            MIoU,IOUS = Eval.Mean_Intersection_over_Union()
            # print('MIoU,ooooou IOUS',MIoU,IOUS)
            FWIoU = Eval.Frequency_Weighted_Intersection_over_Union()

            return PA, MPA, MIoU, FWIoU, IOUS

        with torch.no_grad():
            tqdm_batch = tqdm(self.val_loader, total=self.valid_iterations,
                              desc="Val Epoch-{}-".format(self.current_epoch + 1))
            self.model.eval()
            id2mIOU = []
            for x, y, id in tqdm_batch:
                if self.cuda:
                    x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

                # model
                pred = self.model(x)
                if isinstance(pred, tuple):
                    pred = pred[0]

                y = torch.squeeze(y, 1)

                pred = pred.data.cpu().numpy()
                label = y.cpu().numpy()
                argpred = np.argmax(pred, axis=1)

                self.Eval.add_batch(label, argpred)
                self.totalEval.add_batch(label,argpred)
                PA, MPA, MIoU, FWIoU, _ = val_info(self.Eval)
                id2mIOU.append((id[0], PA, MPA, MIoU, FWIoU))
                if MIoU< threshold:
                    # print("########## Eval ############")
                    id = Path(id[0])
                    base_out_str = '{}_PA_{:.2f}_MPA_{:.2f}_MIoU_{:.2f}_FWIoU_{:.2f}_'.\
                                        format(id.stem, PA, MPA, MIoU, FWIoU)
                    # print('picInfo', base_out_str)

                    images_inv = inv_preprocess(x.clone().cpu(), self.args.show_num_images,
                                                numpy_transform=self.args.numpy_transform).squeeze(0).permute(1,2,0)
                    images_inv_np = images_inv.numpy()*255
                    # print('images_inv_np,type(images_inv_np)',images_inv_np,np.uint8(images_inv_np),type(images_inv_np),images_inv_np.shape)
                    images_inv_np = Image.fromarray(np.uint8(images_inv_np))
                    labels_colors, label_img = decode_labels(label, self.args.show_num_images)
                    preds_colors, pred_img = decode_labels(argpred, self.args.show_num_images)
                    # print('save images to', outputPath)
                    label_img.save(os.path.join(outputPath, base_out_str + 'label_img.png'))
                    pred_img.save(os.path.join(outputPath, base_out_str + 'pred_img.png'))
                    images_inv_np.save(os.path.join(outputPath, base_out_str + 'style_img.png'))

                self.Eval.reset()

            ## output total result
            totalPA, totalMPA, totalMIoU, totalFWIoU, totalIous= val_info(self.totalEval)
            print('##### IOU for each class is #####')
            for key in totalIous:
                print(key,totalIous[key])
            print('totalPA, totalMPA, totalMIoU, totalFWIoU',totalPA, totalMPA, totalMIoU, totalFWIoU)

            tqdm_batch.close()

        return id2mIOU


def add_UDA_train_args(arg_parser):
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
    return arg_parser


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    arg_parser = argparse.ArgumentParser()
    arg_parser = add_train_args(arg_parser)
    arg_parser = add_UDA_train_args(arg_parser)

    args = arg_parser.parse_args()
    args, _, logger = init_args(args)


    styles_source=['GTA5_ambulance_gta5pcity_retrain_alpha1stylewt1']
    styles_target = ['Cityscapes_ambulance_gta5pcity_retrain_alpha1stylewt1']
    # styles_target = ['cityscapes']

    # logger.info("styles_souce,style_target", styles_source, styles_target)

    agent = resultEvaluater(args=args, cuda=True,
                       logger=logger, datasets_path=datasets_path,
                       styles_source=styles_source, styles_target=styles_target)
    agent.main()