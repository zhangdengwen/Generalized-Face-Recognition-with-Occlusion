# 测试遮挡嘴巴和眼睛的图片是否正确

import os
import sys
import shutil
import argparse
import logging as logger

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import os.path as osp
sys.path.append('../../')
from utils.AverageMeter import AverageMeter
from data_processor.train_dataset import ImageDataset,ImageDataset_KD,ImageDataset_Crop, ImageDataset_KD_glasses, ImageDataset_KD_glasses_sunglasses, ImageDataset_KD_glasses_sunglasses_save
from backbone.backbone_def import BackboneFactory
from head.head_def import HeadFactory
from backbone.iresnet import iresnet100
import clip
import random
import numpy as np
import torch.nn as nn
import cv2

import os
import cv2
import torch
import numpy as np
from data_processor.train_dataset import ImageDataset_Crop

data_root = "/home/srp/face_recognition/data/CASIA-WebFace"
train_file = "/home/srp/face_recognition/training_mode/webface_train_list.txt"

# 先定义dataset
dataset = ImageDataset_Crop(data_root, train_file, crop_eye_or_mouth=True)

save_dir = "./debug_imgs"
os.makedirs(save_dir, exist_ok=True)

num_save = 100
for i in range(num_save):
    img, label = dataset[i]

    img_np = img.numpy()
    img_np = (img_np / 0.0078125 + 127.5).astype(np.uint8)
    img_np = img_np.transpose(1, 2, 0)

    save_path = os.path.join(save_dir, f"debug_img_{i}.jpg")
    cv2.imwrite(save_path, img_np)

print(f"保存了{num_save}张遮挡图片到 {save_dir} 文件夹")

# python -m training_mode.conventional_training.test