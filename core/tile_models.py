# !usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2022/6/20
# @Author   : ChenLingHao
# @File     : tile_models.py
import sys
import numpy as np
import torch
from cv2 import cv2
import timm
from torch import nn
from torchvision import models
from torchvision.transforms import transforms as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
sys.path.append('..')

test_transforms = F.Compose([F.ToPILImage(),
                             F.ToTensor(),
                             F.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                            ])

def get_classifier(config):
    model = timm.create_model(model_name=config.backbone, pretrained=config.pretrained,
                              num_classes=config.num_classes)
    return model

def tiles2features(config, tile_dst):
    features = []
    model = timm.create_model(config.backbone, pretrained=True, num_classes=2)
    loader = output_loader(config, tile_dst)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        for images in tqdm(loader, desc="[Generating features]"):
            images = images.to(device)
            feature = model.forward_features(images)
            feature = feature.view(feature.size(0), -1)
            features.append(feature.detach().cpu().numpy())  # 要放到cpu ，以免爆显存
        features = np.concatenate(features, axis=0)
    return features


class BasicTileDataset(Dataset):
    def __init__(self, tiles_dst, config):
        self.tile_paths = tiles_dst
        self.cfg = config

    def __getitem__(self, item):
        img = cv2.imread(self.tile_paths[item])
        img = cv2.resize(img, (self.cfg.resize_img, self.cfg.resize_img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = test_transforms(img)
        return img

    def __len__(self):
        return len(self.tile_paths)

def output_loader(config, tiles_dst):
    loader = DataLoader(BasicTileDataset(tiles_dst, config),
                        batch_size=config.batch_size, pin_memory=False, num_workers=0)
    return loader

class Tee(nn.Module):
    def __init__(self, config):
        super(Tee, self).__init__()
        ...

    def forward(self):
        ...

from timm.models import resnet