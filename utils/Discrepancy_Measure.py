import sys
import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import lr_scheduler

sys.path.append(os.path.abspath('.'))
from utils.train_helper import STNet_refer, Net
from datasets.gta5_Dataset import FlatFolderDataset
from graphs.models.resnet import resnet18


datasets_path={
    'Cityscapes': {'list_path': '/data/Projects/ADVENT/data/Cityscapes/leftImg8bit',
                    'image_path':'/data/Projects/ADVENT/data/Cityscapes/'},
    'GTA5': {'list_path': '/data/Projects/ADVENT/data/GTA5',
             'image_path':'/data/Projects/ADVENT/data/GTA5/images'}}

classes = ['GTA5', 'Cityscapes']

class CityscapeDataset(data.Dataset):
    def __init__(self, dataset_path, split,args):
        self.args = args
        self.dataset_path = dataset_path
        self.split = split
        if 'train' in self.split:
            item_list_filepath = os.path.join(dataset_path['list_path'], 'train'+".txt")
        elif "val" in self.split:
            item_list_filepath = os.path.join(dataset_path['list_path'], 'val'+".txt")

        self.items = [id.strip() for id in open(item_list_filepath)]


    def __getitem__(self, item):
        id = self.items[item]
        imagepath = os.path.join(self.dataset_path['image_path'], id)
        image = Image.open(imagepath).convert('RGB')
        label = 1
        return self.transform()(image),label

    def __len__(self):
        return len(self.items)


    def transform(self):
        size = (self.args.base_size[1], self.args.base_size[0])
        transformlist = [
            transforms.Resize(size),
            transforms.ToTensor()
        ]
        return transforms.Compose(transformlist)


class GTA5dataset(data.Dataset):
    def __init__(self,dataset_path, split,args):
        self.args = args
        self.dataset_path = dataset_path
        self.image_filepath = dataset_path['image_path']
        self.split = split

        if 'train' in self.split:
            item_list_path = os.path.join(dataset_path['list_path'], 'train_c.txt')
        elif 'val' in self.split:
            item_list_path = os.path.join(dataset_path['list_path'], 'test_c.txt')

        self.items = [itemid.strip() for itemid in open(item_list_path)]


    def __getitem__(self, item):
        id = int(self.items[item])
        image_path = os.path.join(self.image_filepath, "{:0>5d}.png".format(id))
        image = Image.open(image_path).convert("RGB")
        label = 0

        return self.transform()(image), label

    def __len__(self):
        return len(self.items)

    def transform(self):
        size = (self.args.base_size[1], self.args.base_size[0])
        transformlist = [
            transforms.Resize(size),
            transforms.ToTensor()
        ]
        return transforms.Compose(transformlist)


def showimg(trainloader):
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


class Trainer():
    def __init__(self,args):
        self.args = args
        self.writer = SummaryWriter(self.args.save_dir)

        print('self.args.ST',self.args.save_dir)
        if self.args.ST:
            self.styletrans_network = STNet_refer()
            self.styletrans_network.eval()

        self.classifier = resnet18()
        if torch.cuda.is_available():
            self.classifier.to('cuda:0')
        self.classifier.train()
        self.criterion = torch.nn.CrossEntropyLoss()


        GTA5_dataset_train = GTA5dataset(datasets_path['GTA5'], split='train',args = self.args)
        Cityscape_dataset_train = CityscapeDataset(datasets_path['Cityscapes'], split='train',args = self.args)
        train_datasets = data.ConcatDataset([GTA5_dataset_train, Cityscape_dataset_train])
        self.dataloader_train = data.DataLoader(train_datasets, self.args.batch_size, True,
                                                num_workers=self.args.data_loader_workers,)

        GTA5_dataset_test = GTA5dataset(datasets_path['GTA5'], split='val',args = self.args)
        Cityscape_dataset_test = CityscapeDataset(datasets_path['Cityscapes'], split='val',args = self.args)
        test_datasets = data.ConcatDataset([GTA5_dataset_test, Cityscape_dataset_test])
        self.dataloader_test = data.DataLoader(test_datasets, self.args.batch_size, False,
                                               num_workers=self.args.data_loader_workers,)

        style_dataset = FlatFolderDataset(
            root="/data/Projects/MaxSquareLoss/imagenet_style/ambulance",
            transform=Cityscape_dataset_train.transform())

        self.style_dataloader=iter(data.DataLoader(
            style_dataset, batch_size=4, shuffle=False,
            num_workers=self.args.data_loader_workers,
            pin_memory=self.args.pin_memory, drop_last=True))

        self.batch_style = next(self.style_dataloader)

        # showimg(self.dataloader_train)

        self.optimizer = torch.optim.SGD(lr=self.args.lr, params=self.classifier.parameters(),
                                         momentum=self.args.momentum,weight_decay=self.args.weight_decay)

        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, mode='min',
            factor=0.5, patience=2, verbose=True)

    def evaluate(self):
        class_correct = [0.,0.]
        class_total = [0.,0.]
        with torch.no_grad():
            for data in tqdm(self.dataloader_test):
                images, labels = data

                if args.ST:
                    with torch.no_grad():
                        result = []
                        for singledata in images:
                            torch.cuda.empty_cache()
                            single_gt = self.styletrans_network(singledata.unsqueeze(0),self.batch_style)
                            result.append(single_gt)
                        images = torch.cat(result,0)

                if torch.cuda.is_available():
                    images,labels = images.to('cuda:0'),labels.to('cuda:0')

                images = F.interpolate(images,size=(256,512))
                outputs = torch.nn.Softmax(1)(self.classifier(images))
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(self.args.batch_size):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                # print(labels,predicted)

        total_acc = sum(class_correct)/sum(class_total)
        GTA5acc = class_correct[0] / class_total[0]
        Cityacc = class_correct[1] / class_total[1]

        return total_acc,GTA5acc,Cityacc


    def save_model(self,path):
        torch.save(self.classifier.state_dict(),os.path.join(path,'best.pth'))


    def train(self):
        tqdm_epoch = tqdm(self.dataloader_train)
        best_acc = 0.0
        i = 0
        for epoch in tqdm(range(self.args.num_epoch)):
            loss_epoch = 0.0
            for data,label in tqdm_epoch:
                if args.ST:
                    with torch.no_grad():
                        result = []
                        for singledata in data:
                            # print('single_data.shape',singledata.shape) #[3, 896, 1792]
                            single_gt = self.styletrans_network(singledata.unsqueeze(0),self.batch_style)
                            result.append(single_gt)
                        # save_image(result[0],'/data/result/discrepancy.png')
                        data = torch.cat(result, 0)

                if torch.cuda.is_available():
                    data,label = Variable(data).to("cuda:0"),Variable(label).to('cuda:0')

                data = F.interpolate(data,size=(256,512))
                pred = self.classifier(data)
                pred = torch.nn.Softmax(1)(pred)

                loss = self.criterion(pred, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_epoch += loss.item()
                # break

            self.scheduler.step(loss_epoch)

            accuracy, GTA5acc, Cityacc = self.evaluate()
            print('loss_epoch: {:.2f}, total accuracy: {:.2f}, GTA5 accuracy: {:.2f}, City accuracy: {:.2f}'
                                            .format(loss_epoch, accuracy, GTA5acc, Cityacc))
            self.writer.add_scalar('loss_epoch', loss_epoch, epoch)
            self.writer.add_scalar('total_acc', accuracy, epoch)
            self.writer.add_scalar('GTA5_acc', GTA5acc, epoch)
            self.writer.add_scalar('City_acc', Cityacc, epoch)

            if accuracy> best_acc:
                best_acc = accuracy
                self.save_model(path = self.args.save_dir)
                print('saved model to {}'.format(self.args.save_dir))

        self.writer.close()

def add_args(parser):
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--momentum', type=float,default=0.9)
    parser.add_argument('--weight_decay',type=float,default=1e-6)
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--base_size',type=tuple,default=(1792,896))

    parser.add_argument('--num_epoch', type=int,default=20)
    parser.add_argument('--data_loader_workers',type=int,default=16)
    arg_parser.add_argument('--pin_memory', default=2, type=int,
                            help='pin_memory of Dataloader')
    parser.add_argument('--save_dir', type=str,default=
       '/data/Projects/MaxSquareLoss/log/train/Discrepancy_measure')

    parser.add_argument('--ST',type=bool,default=False,
                        help='if use style transfer to transfer input image')

    return parser


if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser = add_args(arg_parser)
    args = arg_parser.parse_args()

    agent = Trainer(args)
    agent.train()
