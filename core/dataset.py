# !usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 下午6:49
# @Author   : ChenLingHao
# @File     : dataset.py
import os
from abc import ABC

import pandas as pd
import _pickle as pickle

from cv2 import cv2
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import transforms as F
from torch.utils.data import DataLoader, Dataset


train_transforms = F.Compose([F.ToPILImage(),
                              F.RandomCrop(size=(500, 500))])

test_transforms = F.Compose([F.ToPILImage(),
                             F.ToTensor(),
                             F.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                            ])


class TileDataset(Dataset):
    def __init__(self, config, phase='train', transforms=None, fold=0):
        """先确定csv的结构"""
        self.cfg = config
        self.phase = phase

        csv_src = os.path.join(config.data_root, 'documents')

        tile_df = pd.read_csv(os.path.join(csv_src, 'train_dataset_tile.csv'))
        if phase == 'train':
            self.target_df = tile_df[tile_df.phase != fold].reset_index(drop=True)
        else:
            self.target_df = tile_df[tile_df.phase == fold].reset_index(drop=True)

        self.transforms = transforms

        if config.partial:
            self.target_df = self.target_df[:int(0.2*len(self.target_df))]  # 如果part

    def __getitem__(self, item):
        img_path, label = self._get_info(self.target_df.loc[item, :])

        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.cfg.resize_img, self.cfg.resize_img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(img)

        if not isinstance(label, int):
            label = int(label)

        return img, label

    def _get_info(self, input_df):
        tile_id = input_df.tile_id
        slide_id = input_df.slide_id.split('.')[0]
        tile_label = input_df[self.cfg.target_label_name]

        tile_src = os.path.join(self.cfg.data_root, 'data', slide_id, tile_id+'.png')
        return tile_src, tile_label

    def __len__(self):
        return len(self.target_df)


class TileTestDataset(Dataset):
    def __init__(self, config, target_slide_id):
        self.cfg = config
        self.slide_tiles_root = os.path.join(self.cfg.data_root, 'data', target_slide_id)
        slide_tile_df = pd.read_csv(os.path.join(self.cfg.data_root,
                                                 'documents', 'slides_tiles_csv',
                                                 target_slide_id+".csv"))
        self.df = slide_tile_df[slide_tile_df == 'tissue'].reset_index(drop=True)

    def __getitem__(self, item):
        row = self.df.loc[item, :]
        tile_path = os.path.join(self.slide_tiles_root, row.tile_id+'.png')
        img = cv2.imread(tile_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = test_transforms(img)
        return img

    def __len__(self):
        return len(self.df)


class MILDataset(Dataset):
    def __init__(self, config, phase='train', transforms=None, fold=0):
        """先确定csv的结构"""
        self.cfg = config
        self.phase = phase
        tile_df = pd.read_csv(os.path.join(self.cfg.documents_root, 'train_dataset_{}.csv'.format(self.cfg.task)))
        self.target_df = tile_df[tile_df.phase == phase].reset_index(drop=True)
        self.transforms = transforms


class MilTestDataset(Dataset):
    def __init__(self, config, target_slide_id, transforms):
        self.cfg = config
        self.slide_tile_df = pd.read_csv(os.path.join(self.cfg.data_root,
                                                      'documents', 'slides_tiles_csv',
                                                      target_slide_id + ".csv"))

        self.transforms = transforms

    def __getitem__(self, item):
        ...

    def __len__(self):
        return None


param_dataloader = dict(pin_memory=False, num_workers=0)
def dataloader(config, k=0):
    if config.task == 'tile':
        train_set = TileDataset(config, 'train', transforms=test_transforms, fold=k)
        valid_set = TileDataset(config, 'valid', transforms=test_transforms, fold=k)
        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, **param_dataloader)
        valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, **param_dataloader)
    else:
        train_set = MILDataset(config, 'train', transforms=test_transforms, fold=k)
        valid_set = MILDataset(config, 'valid', transforms=test_transforms, fold=k)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=False, **param_dataloader)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, **param_dataloader)
    return train_loader, valid_loader


def test_loader(config, slide_id):
    if config.task == 'tile':
        test_set = TileTestDataset(config, slide_id)
        loader = DataLoader(test_set, batch_size=config.batch_size, **param_dataloader)
    else:
        test_set = MilTestDataset(config, slide_id)
        loader = DataLoader(test_set, batch_size=config.batch_size, **param_dataloader)
    return loader

















