# Domain Adaptation for Semantic Segmentation with Maximum Squares Loss

By Minghao Chen, Hongyang Xue, Deng Cai.

## Introduction

A **PyTorch** implementation of a coarse to fine feature alignment method useing style transfer based coarse feature alignment and entropy minimization based fine feature alignment 

## Requirements
The code is implemented with Python(3.6) and Pytorch(1.0.0).

Install the newest Pytorch from https://pytorch.org/.

To install the required python packages, run

```python
pip install -r requirements.txt
```

## Setup

#### GTA5-to-Cityscapes:

- Download [**GTA5 datasets**](https://download.visinf.tu-darmstadt.de/data/from_games/), which contains 24,966 annotated images with 1914×1052 resolution taken from the GTA5 game. We use the sample code for reading the label maps and a split into training/validation/test set from [here](https://download.visinf.tu-darmstadt.de/data/from_games/code/read_mapping.zip). In the experiments, we resize GTA5 images to 1280x720.
- Download [**Cityscapes**](https://www.cityscapes-dataset.com/), which contains 5,000 annotated images with 2048 × 1024 resolution taken from real urban street scenes. We resize Cityscapes images to 1024x512 (or 1280x640 which yields sightly better results but costs more time). 
- Download the **[checkpoint](https://drive.google.com/open?id=1KP37cQo_9NEBczm7pvq_zEmmosdhxvlF)** pretrained on GTA5.
- If you want to pretrain the model by yourself, download [**the model**](http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth) pretrained on ImageNet.

#### SYNTHIA-to-Cityscapes:

- Download [**SYNTHIA-RAND-CITYSCAPES**](http://synthia-dataset.net/download/808/) consisting of 9,400 1280 × 760 synthetic images. We resize images to 1280x760.
- Download the [**checkpoint**](https://drive.google.com/open?id=1wLffQRljXK1xoqRY64INvb2lk2ur5fEL) pretrained on SYNTHIA.

#### Cityscapes-to-CrossCity

- Download [**NTHU dataset**](https://yihsinchen.github.io/segmentation_adaptation_dataset/), which consists of images with 2048 × 1024 resolution from four different cities: Rio, Rome, Tokyo, and Taipei. We resize images to 1024x512, the same as Cityscapes.
- Download the **[checkpoint](https://drive.google.com/open?id=1QMpj7sPqsVwYldedZf8A5S2pT-4oENEn)** pretrained on Cityscapes.

Put all datasets into "datasets" folder and all checkpoints into "pretrained_model" folder.

## Results

We present several transfered results reported in our paper and provide the corresponding checkpoints.

![results](/result.png)

### GTA5-to-Cityscapes:

| Method  | Source | style transfer | [MaxSquare](https://drive.google.com/open?id=1KmM8zBD1G1XTmzaV_I_aJgi9DW-49kxc) | [MaxSquare+IW](https://drive.google.com/open?id=11oliS-Vu2W6dB8W9ZvqlN0R4cC4Pb8i6) | [MaxSquare+IW+Multi](https://drive.google.com/open?id=1YwK68IMmWHZnAL8FU9ZY-Le34P80Kf86) |
| :-----: | :----: | :----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| mIoU(%) |  36.9  |   44.2 |                             44.3                             |                             45.2                             |                             46.4                             |



## Training

### GTA5-to-Cityscapes:

#### 1. (Optional) Pretrain the model on the source domain (GTA5). 


```
rlaunch --gpu=1 --cpu=10 --memory=10000 -- python3 tools/train_source.py --gpu "0" --dataset 'gta5' --checkpoint_dir "./log/gta5_pretrain/" --iter_max 200000 --iter_stop 80000 --freeze_bn False --weight_decay 5e-4 --lr 2.5e-4 --crop_size "1280,720"
```

Pretrain the multi-level model on the source domain (GTA5) by adding "--multi True". 

```
rlaunch --gpu=1 --cpu=10 --memory=10000 -- python3 tools/train_source.py --gpu "0" --dataset 'gta5' --checkpoint_dir "./log/train/add_multi" --iter_max 200000 --iter_stop 80000 --freeze_bn False --weight_decay 5e-4 --lr 2.5e-4 --crop_size "1280,720" --multi True
```

Otherwise, download the [checkpoint](https://drive.google.com/open?id=1KP37cQo_9NEBczm7pvq_zEmmosdhxvlF) pretrained on GTA5 in "Setup" section.

#### 2. train the model on source and target domain 
Then in next step, set `--checkpoint_dir "./log/train" --restore_id GTA5_source` to load the GTA5_sourcebest.pth in ./log/train.

Or set the --checkpint_dir and restore_id to load  `"checkpoint_dir/restore_idbest.pth"` that trained by yourself.

Also, remember to set the `--exp_tag` to mark output files and `--save_dir` as the directory you want to save your output.

- MaxSquare


```
rlaunch --gpu=1 --cpu=10 --memory=10000 -- python3 tools/solve_gta5.py --gpu "0" --backbone "deeplabv2_multi" --dataset 'cityscapes' --exp_tag image_net_pretrain_MS_repeat --restore_id image_net_pretrain --checkpoint_dir "./log/train/image_net_pretrain_bs1_right_model_gta_only/image_net_pretrain"  --save_dir "./log/train/image_net_pretrain_bs1_right_model_gta_only/MS_repeat"  --round_num 15 --target_mode "maxsquare" --freeze_bn False --weight_decay 5e-4 --lr 2.5e-4 --lambda_target 0.1
```

- MaxSquare+IW


```
rlaunch --gpu=1 --cpu=10 --memory=10000 -- python3 tools/solve_gta5.py --gpu "0" --backbone "deeplabv2_multi" --dataset 'cityscapes' --exp_tag image_net_pretrain_IW_MS_repeat --restore_id image_net_pretrain --checkpoint_dir "./log/train/image_net_pretrain_bs1_right_model_gta_only/image_net_pretrain" --save_dir "./log/train/image_net_pretrain_bs1_right_model_gta_only/IW_MS_repeat" --round_num 15 --target_mode "IW_maxsquare" --freeze_bn False --weight_decay 5e-4 --lr 2.5e-4 --lambda_target 0.1 --IW_ratio 0.2
```


- MaxSquare+IW+Multi

```
rlaunch --gpu=1 --cpu=10 --memory=10000 -- python3 tools/solve_gta5.py --gpu "0" --backbone "deeplabv2_multi" --dataset 'cityscapes' --exp IW_MS_repeat --restore_id add_multi --checkpoint_dir "./log/train/add_multi_gta_only" --save_dir "./log/train/add_multi_gta_only/multi_MS_IW_repeat"  --round_num 15 --target_mode "IW_maxsquare" --freeze_bn False --weight_decay 5e-4 --lr 2.5e-4 --target_crop_size "1280,640" --lambda_target 0.09 --IW_ratio 0.2 --multi True --lambda_seg 0.1 --threshold 0.95
short version:
rlaunch --gpu=1 --cpu=10 --memory=10000 -- python3 tools/solve_gta5.py --exp IW_MS_target_solo --restore_id add_multi --checkpoint_dir "./log/train/add_multi_gta_only" --save_dir "./log/train/add_multi_gta_only/multi_MS_IW_target_solo"  --round_num 20 --target_mode "IW_maxsquare" --lambda_target 0.09 --multi True
```
##### if your want to continue training 
Set`"--continue_training True"` and change the init lr same as where it ends in `"--lr"`

Remember that the `"--exp_tag"` and `"--save_dir"` needs to be the same as before to see continuous training curve in tensorboard 

- MaxSquare

```
rlaunch --gpu=1 --cpu=10 --memory=10000 -- python3 tools/solve_gta5.py --gpu "0" --continue_training True --backbone "deeplabv2_multi" --dataset 'cityscapes' --exp_tag image_net_pretrain_MS --restore_id gta52cityscapes_maxsquare --checkpoint_dir "./log/train/image_net_pretrain_bs1_right_model_gta_only/MS" --save_dir "./log/train/image_net_pretrain_bs1_right_model/MS"  --round_num 5 --target_mode "maxsquare" --freeze_bn False --weight_decay 5e-4 --lr 2.05e-7 --lambda_target 0.1
```

- MaxSquare+IW
```
rlaunch --gpu=1 --cpu=10 --memory=10000 -- python3 tools/solve_gta5.py --gpu "0" --continue_training True --backbone "deeplabv2_multi" --dataset 'cityscapes' --exp_tag image_net_pretrain_MS_IW  --restore_id gta52cityscapes_IW_maxsquare --checkpoint_dir "./log/train/image_net_pretrain_bs1_right_model_gta_only/IW_MS"  --save_dir "./log/train/image_net_pretrain_bs1_right_model_gta_only/IW_MS"  --round_num 10 --target_mode "IW_maxsquare" --freeze_bn False --weight_decay 5e-4 --lr 2.05e-7 --lambda_target 0.1 --IW_ratio 0.2

```

- MaxSquare+IW+Multi

```
rlaunch --gpu=1 --cpu=10 --memory=10000 -- python3 tools/solve_gta5.py --gpu "0" --continue_training True --backbone "deeplabv2_multi" --dataset 'cityscapes' --exp_tag image_net_pretrain_MS_IW_Multi --restore_id gta52cityscapes_IW_maxsquare --checkpoint_dir "./log/train/multi_MS_IW"  --save_dir "./log/train/multi_MS_IW" --round_num 5 --target_mode "IW_maxsquare" --freeze_bn False --weight_decay 5e-4 --lr 2.05e-7 --target_crop_size "1280,640" --lambda_target 0.09 --IW_ratio 0.2 --multi True --lambda_seg 0.1 --threshold 0.95

```

#### 3.evaluation
Eval:

```
python3 tools/evaluate.py --gpu "0" --dataset 'cityscapes' --checkpoint_dir "./log/eval_city" --pretrained_ckpt_file "./log/gta2city_AdaptSegNet_ST=0.1_maxsquare_round=5/gta52city_maxsquarebest.pth" --image_summary True --flip True
```

```
rlaunch --gpu=1 --cpu=10 --memory=10000 -- python3 tools/evaluate.py --gpu "0" --dataset 'cityscapes' --checkpoint_dir "./log/eval_city/add_multi_gta_only/multi_MS_IW/" --pretrained_ckpt_file "./log/train/add_multi_gta_only/multi_MS_IW/gta52cityscapes_IW_maxsquarebest.pth" --image_summary True --flip True
```

To have a look at predicted examples, run tensorboard as follows:

```
sudo tensorboard --logdir=./log/train 
```



### Cityscapes-to-CrossCity

(Optional) Pretrain the model on the source domain (Cityscapes). 

```
python3 tools/train_source.py --gpu "0" --dataset 'cityscapes' --checkpoint_dir "./log/cityscapes_pretrain_class13/" --iter_max 200000 --iter_stop 80000 --freeze_bn False --weight_decay 5e-4 --lr 2.5e-4 --crop_size "1024,512" --num_classes 13
```

- MaxSquare (take "Rome" for example)

```
python3 tools/solve_crosscity.py --gpu "0" --city_name 'Rome' --source_dataset 'cityscapes' --checkpoint_dir "./log/city2Rome_maxsquare/" --pretrained_ckpt_file "./pretrained_model/Cityscapes_source_class13.pth"  --crop_size "1024,512" --target_crop_size "1024,512"  --epoch_num 10 --target_mode "maxsquare" --lr 2.5e-4 --lambda_target 0.1 --num_classes 13
```



## Acknowledgment

The structure of this code is largely based on [this repo](https://github.com/ZJULearning/MaxSquareLoss).

Deeplabv2 model is borrowed from [Pytorch-Deeplab](https://github.com/speedinghzl/Pytorch-Deeplab).
