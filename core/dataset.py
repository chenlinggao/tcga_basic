# !usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 下午6:49
# @Author   : ChenLingHao
# @File     : dataset.py
import os
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

param_dataloader = dict(pin_memory=False, num_workers=0)

partial_ratio = 0.02


class TileDataset(Dataset):
    # finish
    def __init__(self, config, phase='train', transforms=None, fold=0):
        """先确定csv的结构"""
        self.cfg = config
        self.phase = phase

        csv_src = os.path.join(config.data_root, 'documents')

        train_df = pd.read_csv(os.path.join(csv_src, 'train_dataset_tile.csv'))

        if config.train_all:
            self.target_df = train_df
        else:
            if phase == "train":
                self.target_df = train_df[train_df.phase != fold].reset_index(drop=True)
            else:
                self.target_df = train_df[train_df.phase == fold].reset_index(drop=True)

        if self.cfg.partial:
            self.target_df = self.target_df[:int(partial_ratio * len(self.target_df))]
        self.transforms = transforms

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

        tile_src = os.path.join(self.cfg.data_root, 'data', slide_id, tile_id + '.png')
        return tile_src, tile_label

    def __len__(self):
        return len(self.target_df)


class TileTestDataset(Dataset):
    def __init__(self, config, target_slide_id):
        """读取目标slide的所有tile后，排除bg的tile，再对其余tile进行预测"""
        self.cfg = config
        self.slide_tiles_root = os.path.join(self.cfg.data_root, 'data', target_slide_id)
        slide_tile_df = pd.read_csv(os.path.join(self.cfg.data_root,
                                                 'documents', 'slides_tiles_csv',
                                                 target_slide_id + ".csv"))
        self.df = slide_tile_df[slide_tile_df == 'tissue'].reset_index(drop=True)

    def __getitem__(self, item):
        row = self.df.loc[item, :]
        tile_path = os.path.join(self.slide_tiles_root, row.tile_id + '.png')
        img = cv2.imread(tile_path)
        img = cv2.resize(img, (self.cfg.resize_img, self.cfg.resize_img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = test_transforms(img)
        return img

    def __len__(self):
        return len(self.df)


class MILDataset(Dataset):
    def __init__(self, config, phase='train', fold=0):
        self.cfg = config
        self.phase = phase
        train_df = pd.read_csv(os.path.join(self.cfg.data_root, 'documents', 'train_dataset_mil.csv'))

        if config.train_all:
            self.target_df = train_df
        else:
            if phase == "train":
                self.target_df = train_df[train_df.phase != fold].reset_index(drop=True)
            else:
                self.target_df = train_df[train_df.phase == fold].reset_index(drop=True)

        if self.cfg.partial:
            self.target_df = self.target_df[:int(partial_ratio * len(self.target_df))]

        self.slide_ids = self.target_df.slide_id

    def __getitem__(self, item):
        # 进入target slide的pkl
        with open(os.path.join(self.cfg.data_root, 'features', self.slide_ids[item] + '.pkl'), "rb") as f:
            bag = pickle.load(f)  # np.ndarray
        bag = torch.tensor(bag)

        # slide_max_tiles可能会添加
        # get label
        row = self.target_df[self.target_df.slide_id == self.slide_ids[item]]
        label = row[self.cfg.target_label_name].to_list()[0]

        return bag, int(label)

    def __len__(self):
        return len(self.slide_ids)


class MilTestDataset(Dataset):
    def __init__(self, config, target_slide_id):
        self.cfg = config
        df = pd.read_csv(os.path.join(self.cfg.data_root, 'documents', 'train_dataset_mil.csv'))
        self.target_df = df[df.slide_id == target_slide_id]

    def __getitem__(self, item):
        slide_name = self.target_df.slide_id.to_list()[0]
        # 进入target slide的pkl
        with open(os.path.join(self.cfg.data_root, 'features', slide_name + 'pkl'), "rb") as f:
            bag = pickle.load(f)  # np.ndarray
        bag = torch.tensor(bag)
        return bag

    def __len__(self):
        return len(self.target_df)


def dataloader(config, k=0):
    if config.task == 'tile':
        if config.train_all:
            train_set = TileDataset(config, 'train', transforms=test_transforms, fold=k)
            train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, **param_dataloader)
            valid_loader = None
        else:
            train_set = TileDataset(config, 'train', transforms=test_transforms, fold=k)
            valid_set = TileDataset(config, 'valid', transforms=test_transforms, fold=k)
            train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, **param_dataloader)
            valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, **param_dataloader)
    else:
        if config.train_all:
            train_set = MILDataset(config, )
            train_loader = DataLoader(train_set, batch_size=1, shuffle=False, **param_dataloader)
            valid_loader = None
        else:
            train_set = MILDataset(config, 'train', fold=k)
            valid_set = MILDataset(config, 'valid', fold=k)

            # 如果需要在train时候随机选一定数量的特征，在collate_fn中进行筛选，这样也可以shuffle
            train_loader = DataLoader(train_set, batch_size=1, shuffle=False, **param_dataloader)
            valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, **param_dataloader)
    return train_loader, valid_loader


def output_test_loader(config, slide_id):
    if config.task == 'tile':
        test_set = TileTestDataset(config, slide_id)
        loader = DataLoader(test_set, batch_size=config.batch_size, **param_dataloader)
    else:
        test_set = MilTestDataset(config, slide_id)
        loader = DataLoader(test_set, batch_size=1, **param_dataloader)
    return loader
