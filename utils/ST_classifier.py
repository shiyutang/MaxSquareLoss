import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.abspath('.'))

from utils.train_helper import STNet, vgg, decoder
from graphs.models.resnet import resnet18
from utils.Discrepancy_Measure import datasets_path
from datasets.gta5_Dataset import FlatFolderDataset

classes = ['GTA5', 'Cityscapes']


def add_args(parser):
    # Directories
    parser.add_argument('--style_dir', type=str,
                        default='/data/Projects/MaxSquareLoss/imagenet_style/ambulance/',
                        help='Directory path to a batch of style images')
    parser.add_argument('--save_dir',
                        default='./experiments/gta5pcity_ambulance_alpha1wts1_classifier_512r1024_traininterval3_trackgrad',
                        help='Directory to save the model')
    parser.add_argument('--log_dir', default='./logs/gta5pcity_ambulance_alpha1wts1_classifier_512r1024_traininterval3_trackgrad',
                        help='Directory to save the log')

    parser.add_argument('--vgg', type=str, default='/data/Projects/pytorch-AdaIN/models/vgg_normalised.pth')
    parser.add_argument('--classifier', type=str, default='/data/Projects/MaxSquareLoss/log/'
                                                          'train/Discrepancy_measure/alpha1wts1/best.pth')
    parser.add_argument("--decoder", type=str, default='/data/Projects/pytorch-AdaIN/experiments/gta5pcity_ambulance_'
                                                       'alpha1wts1awts1e-4_affineloss_pretrain11/decoder_iter_66000.pth.tar')  # models/decoder.pth')

    # training options
    parser.add_argument('--lr_ST', type=float, default=1e-4)
    parser.add_argument('--lr_decay_ST', type=float, default=5e-5)
    parser.add_argument('--lr_c', type=float, default=1e-3)
    parser.add_argument('--weight_decay_c', type=float, default=1e-6)
    parser.add_argument('--momentum_c', type=float, default=0.9)

    parser.add_argument('--classify_size', type=tuple, default=(256, 512))
    parser.add_argument('--epoch_num', type=int, default=40)
    parser.add_argument('--max_iter', type=int, default=160000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--style_weight', type=float, default=1)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--classify_weight', type=float, default=1.0)

    # optional modules
    parser.add_argument('--style_train_interval', type=int, default=3)

    # settings
    parser.add_argument('--n_threads', type=int, default=0)
    parser.add_argument('--pin_memory', default=2, type=int,
                        help='pin_memory of Dataloader')
    parser.add_argument('--save_model_interval', type=int, default=40000)

    return parser


def adjust_learning_rate(optimizer, args, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr_ST / (1.0 + args.lr_decay_ST * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def saveimg(g_t, fname, args):
    from torchvision.utils import save_image
    output = g_t.cpu()
    output_name = args.save_dir + '/{:s}_.png'.format(fname)
    # Path(content_image_path[0]).stem)
    print("output_name", output_name, str(output_name))
    save_image(output, str(output_name))


class CityscapeDataset(data.Dataset):
    def __init__(self, dataset_path, split, args):
        self.args = args
        self.dataset_path = dataset_path
        self.split = split
        if 'train' in self.split:
            item_list_filepath = os.path.join(dataset_path['list_path'], 'train' + ".txt")
        elif "val" in self.split:
            item_list_filepath = os.path.join(dataset_path['list_path'], 'val' + ".txt")

        self.items = [id.strip() for id in open(item_list_filepath)]

    def __getitem__(self, item):
        id = self.items[item]
        imagepath = os.path.join(self.dataset_path['image_path'], id)
        image = Image.open(imagepath).convert('RGB')
        label = 1  # 0.9 + 0.1 * random.random()

        return self.train_transform()(image), label

    def __len__(self):
        return len(self.items)

    def train_transform(self):
        transform_list = [
            transforms.Resize(size=(1056, 1920)),  # 不合适的输入，通过decoder之后会得到不一样的输出
            transforms.RandomCrop((256, 512)),
            transforms.ToTensor()
        ]
        return transforms.Compose(transform_list)


class GTA5dataset(CityscapeDataset):
    # noinspection PyMissingConstructor
    def __init__(self, dataset_path, split, args):
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
        label = 0  # 0.1 * random.random()

        return self.train_transform()(image), label


class Trainer():
    def __init__(self, args):
        self.args = args
        self.device_init()
        self.current_iter = 0

        ## path
        self.save_dir = Path(self.args.save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir = Path(self.args.log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        ## data
        GTA5_dataset_train = GTA5dataset(datasets_path['GTA5'], split='train', args=self.args)
        Cityscape_dataset_train = CityscapeDataset(datasets_path['Cityscapes'], split='train', args=self.args)
        train_datasets = data.ConcatDataset([GTA5_dataset_train, Cityscape_dataset_train])
        self.dataloader_train = data.DataLoader(train_datasets, self.args.batch_size, True,
                                                num_workers=self.args.n_threads)

        GTA5_dataset_test = GTA5dataset(datasets_path['GTA5'], split='val', args=self.args)
        Cityscape_dataset_test = CityscapeDataset(datasets_path['Cityscapes'], split='val', args=self.args)
        test_datasets = data.ConcatDataset([GTA5_dataset_test, Cityscape_dataset_test])
        self.dataloader_test = data.DataLoader(test_datasets, self.args.batch_size, False,
                                               num_workers=self.args.n_threads)

        style_dataset = FlatFolderDataset(root=self.args.style_dir,
                                          transform=Cityscape_dataset_train.train_transform())
        self.style_dataloader = iter(data.DataLoader(style_dataset, batch_size=self.args.batch_size, shuffle=False,
                                                     num_workers=self.args.n_threads,
                                                     pin_memory=self.args.pin_memory, drop_last=True))
        self.batch_style = next(self.style_dataloader).to(self.device0)

        ## network
        self.network = self.netinit(vgg, decoder).to(self.device0)
        self.network.train()
        self.classifier.train()

        ## optimizer and criterion
        self.optimizer = torch.optim.Adam(self.network.module.decoder.parameters(),
                                          lr=args.lr_ST)

        self.optimizer_classify = torch.optim.SGD(self.classifier.parameters(), lr=args.lr_c,
                                                  weight_decay=args.weight_decay_c,
                                                  momentum=args.momentum_c)

        self.criterion_classify = torch.nn.CrossEntropyLoss()

        self.scheduler_c = lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer_classify, mode='min',
                                                          factor=0.9, patience=3, verbose=True)

    def device_init(self):
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.device0 = torch.device('cuda:0')
                self.device1 = torch.device('cuda:1')
            else:
                self.device0 = torch.device('cuda')
        else:
            self.device0 = torch.device('cpu')
            self.device1 = torch.device('cpu')

    def netinit(self, encoder, decode_net):
        encoder.load_state_dict(torch.load(self.args.vgg))
        encoder = nn.Sequential(*list(encoder.children())[:31])
        decode_net.load_state_dict(torch.load(self.args.decoder, map_location=self.device0))

        network = STNet(encoder, decode_net)
        network.train()
        network.to(self.device0)
        network = nn.DataParallel(network)

        self.classifier = resnet18()
        if self.args.classifier:
            self.classifier.load_state_dict(torch.load(self.args.classifier, map_location=self.device0))
        self.classifier.to(self.device1)

        return network

    def save_checkpoint(self, netsave, save_name):
        state_dict = netsave.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, self.save_dir / save_name)

    def evaluate(self):
        class_correct = [0., 0.]
        class_total = [0., 0.]
        with torch.no_grad():
            for data in tqdm(self.dataloader_test):
                images, labels = data

                torch.cuda.empty_cache()
                images_ST = self.network(images.to(self.device0), self.batch_style)

                images_ST, labels = images_ST[2].to(self.device1), labels.to(self.device1)

                images = torch.nn.functional.interpolate(images_ST, size=self.args.classify_size)
                outputs = torch.nn.Softmax(1)(self.classifier(images))
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                if self.args.batch_size == 1:
                    class_correct[labels] += c.item()
                    class_total[labels] += 1
                else:
                    for i in range(self.args.batch_size):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1
                # print(labels,predicted)

        total_acc = sum(class_correct) / sum(class_total)
        GTA5acc = class_correct[0] / class_total[0]
        Cityacc = class_correct[1] / class_total[1]

        return total_acc, GTA5acc, Cityacc

    def track_grad(self):
        classifier_grad, network_grad = [], []
        for i, item in enumerate(self.classifier.named_parameters()):
            h = item[1].register_hook(lambda grad: grad)
            if i == 0:
                grad = item[1].grad.data.cpu().detach().numpy().flatten()[0]
                classifier_grad.append(grad)
        grad = item[1].grad.data.cpu().detach().numpy().flatten()[0]
        classifier_grad.append(grad)
        for j, item in enumerate(self.network.module.decoder.named_parameters()):
            h = item[1].register_hook(lambda grad: grad)
            if j == 0:
                grad = item[1].grad.data.cpu().detach().numpy().flatten()[0]
                network_grad.append(grad)
        grad = item[1].grad.data.cpu().detach().numpy().flatten()[0]
        network_grad.append(grad)
        return classifier_grad, network_grad

    def log_one_epoch(self, epoch):
        self.writer.add_scalar('total_acc', self.total_acc, epoch)
        self.writer.add_scalar('GTA5_acc', self.GTA5acc, epoch)
        self.writer.add_scalar('City_acc', self.Cityacc, epoch)
        self.writer.add_scalar('classifier_top_grad', self.classifier_grad[0], epoch)
        self.writer.add_scalar('classifier_down_grad', self.classifier_grad[1], epoch)
        self.writer.add_scalar('network_top_grad', self.network_grad[0], epoch)
        self.writer.add_scalar('network_down_grad', self.network_grad[1], epoch)

    def train(self):
        import torch.nn.functional as F
        tqdm_epoch = tqdm(self.dataloader_train)
        for epoch in tqdm(range(self.args.epoch_num)):
            for data, label in tqdm_epoch:
                adjust_learning_rate(self.optimizer, self.args, iteration_count=self.current_iter)

                loss_c, loss_s, data_ST = self.network(data.to(self.device0), self.batch_style)

                data_ST = F.interpolate(data_ST, self.args.classify_size)
                pred = self.classifier(data_ST.to(self.device1))
                pred = torch.nn.Softmax(1)(pred)

                loss_classify = self.criterion_classify(pred, label.to(self.device1))

                if epoch % self.args.style_train_interval == 0:
                    loss_c = self.args.content_weight * loss_c
                    loss_s = self.args.style_weight * loss_s
                    loss_discrepancy = self.args.classify_weight * \
                                       self.criterion_classify(pred, (1 - label).to(self.device1))

                    self.optimizer.zero_grad()
                    loss_ST = loss_c + loss_s
                    loss_discrepancy.backward(retain_graph=True)
                    loss_ST.backward(retain_graph=True)
                    self.optimizer.step()

                self.optimizer_classify.zero_grad()
                loss_classify.backward()
                self.optimizer_classify.step()

                if self.current_iter % 10 == 0:
                    self.writer.add_scalar('loss_content', loss_c.item(), self.current_iter + 1)
                    self.writer.add_scalar('loss_style', loss_s.item(), self.current_iter + 1)

                if (self.current_iter + 1) % self.args.save_model_interval == 0 or \
                        (self.current_iter + 1) == self.args.max_iter:
                    selrf.save_checkpoint(self.network.module.decoder,
                                         'decoder_iter_{:d}.pth.tar'.format(self.current_iter + 1))
                    self.save_checkpoint(self.classifier,
                                         'classifier_iter_{:d}.pth.tar'.format(self.current_iter + 1))
                if self.current_iter == 0:
                    saveimg(data_ST, 'classify_epoch_{}'.format(epoch), self.args)

                self.current_iter += 1
                # break

            self.total_acc, self.GTA5acc, self.Cityacc = self.evaluate()
            print(' total accuracy: {:.2f}, GTA5 accuracy: {:.2f}, City accuracy: {:.2f}'
                  .format(self.total_acc, self.GTA5acc, self.Cityacc))

            saveimg(data_ST, 'classify_epoch_{}'.format(epoch), self.args)

            self.classifier_grad, self.network_grad = self.track_grad()
            self.log_one_epoch(epoch)

        self.writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    arguments = parser.parse_args()
    agent = Trainer(arguments)
    agent.train()
