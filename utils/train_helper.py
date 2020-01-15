import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from graphs.models.deeplab_multi import DeeplabMulti

def get_model(args):
    if args.backbone == "deeplabv2_multi":
        model = DeeplabMulti(num_classes=args.num_classes,
                            pretrained=args.imagenet_pretrained)
        params = model.optim_parameters(args) #分类层是十倍的学习率，其他是正常学习率
        args.numpy_transform = True
    return model, params