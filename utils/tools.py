# !usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 上午10:48
# @Author   : ChenLingHao
# @File     : tools.py

import os
import copy
import logging
import shutil
import numpy as np
import pandas as pd
from glob import glob
from random import shuffle
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split


def message_output(input_string, input_logger=None, level='info'):
    if input_logger:
        if level == 'info':
            input_logger.info(input_string)
        elif level == 'warn':
            input_logger.warning(input_string)
        elif level == 'error':
            input_logger.error(input_string)
    else:
        print(input_string)


def construct_logger(log_root, log_name=None, save_time=True):
    """
    将运行的日志进行保存
    :param log_root: 日志的地址
    :param save_time: Optional['time', 'msg']
    :param log_name: 日志的名称
    :return:
    """
    fh_fmt = logging.Formatter('%(message)s')
    if save_time:
        fh_fmt = logging.Formatter('[%(asctime)s]: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    if log_name is None:
        log_name = 'trained_model'

    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join(log_root, log_name + '.logs'), encoding='utf-8')
    sh = logging.StreamHandler()

    # 创建logger并设置日志级别和格式对象
    logger.setLevel(logging.DEBUG)
    fh.setLevel(level=logging.DEBUG)
    sh.setLevel(logging.DEBUG)

    sh.setFormatter(logging.Formatter('%(message)s'))
    # fh.setFormatter(logging.Formatter('%(message)s'))
    fh.setFormatter(fh_fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class FolderTool:
    def __init__(self, input_path, renew=False, logger=None):
        """
        创建文件夹
        :param input_path: 地址的目录，可以传入一个目录的列表，创建多个目录
        :param renew: 清空文件夹
        """
        self.folder_path = input_path
        self.renew = renew
        self.logger = logger

    def doer(self):
        if isinstance(self.folder_path, str):
            if not os.path.exists(self.folder_path):
                self.renew_folder(self.folder_path)

        elif isinstance(self.folder_path, list):
            for p in self.folder_path:
                if not os.path.exists(p):
                    self.renew_folder(p)
        else:
            msg = ("[WARNING]: Path [{}] is exists".format(self.folder_path))
            message_output(msg, self.logger, level='warn')

    def renew_folder(self, p):
        if os.path.exists(p) and self.renew:
            print("\n[WARNING]: Delete [{}]".format(p))
            shutil.rmtree(p)
            os.makedirs(p)
        else:
            os.makedirs(p)


class TrainValidTestSplit:
    """input a csv, and split all to 'train-valid-test'"""

    def __init__(self, data_csv, ratio: list = None, stratify_name=None, random_state: int = 2022, names=None):
        if isinstance(data_csv, str):
            self.df = pd.read_csv(data_csv)
        else:
            self.df = data_csv

        if ratio is None:
            self.ratio = [8, 1, 1]
        else:
            self.ratio = ratio

        if names == 'tt':
            self.names = ['train', 'test']
        elif names == 'tv':
            self.names = ['train', 'valid']
        else:
            self.names = ['train', 'valid']

        assert len(self.ratio) == len(self.names), "[Error] ratio length of [{}] != names length of [{}]".format(self.ratio, self.names)

        self.stratify_name = stratify_name
        self.random_state = random_state

    def fit(self, save_dst=None, name=None):
        if len(self.ratio) == 2:
            output_df = self.split_train_valid()
        elif len(self.ratio) == 3:
            output_df = self.split_train_valid_test()
        else:
            raise IOError("length of ratio must be [2 0r 3], now is [{}]".format(len(self.ratio)))

        if name is None:
            name = "data_set.csv"

        if save_dst is not None:
            output_df.to_csv(os.path.join(save_dst, name), index=False)
        return output_df

    def split_train_valid_test(self):
        df = self.df
        # split train and test
        df = self._split_set(input_df=df,
                             ratio=[self.ratio[0] + self.ratio[1], self.ratio[2]],
                             names=['train', 'test'])
        # split train and valid
        df = self._split_set(input_df=df,
                             ratio=[self.ratio[0], self.ratio[1]],
                             names=['train', 'valid'])
        return df

    def split_train_valid(self):
        df = self.df
        # split train and valid
        df = self._split_set(input_df=df,
                             ratio=[self.ratio[0], self.ratio[1]],
                             names=self.names)
        return df

    def _split_set(self, input_df, ratio, names=None):
        if names is None:
            names = ['train', 'test']
        indexes = input_df.index.to_list()
        shuffle(indexes)
        if self.stratify_name is not None:
            labels = input_df[self.stratify_name].to_list()
            X_train, X_test = train_test_split(indexes, test_size=ratio[1] / ratio[0],
                                               random_state=self.random_state, stratify=labels)
        else:
            X_train, X_test = train_test_split(indexes, test_size=ratio[1] / ratio[0], random_state=self.random_state)

        input_df.loc[X_train, 'phase'] = names[0]
        input_df.loc[X_test, 'phase'] = names[1]
        return input_df


class TrainValidTestSplit_k_fold:
    def __init__(self, data_csv, k_fold: int = 5, stratify_name=None,
                 random_state: int = 2022, column_name='phase'):
        if isinstance(data_csv, str):
            self.df = pd.read_csv(data_csv)
        else:
            self.df = data_csv
        self.stratify_name = stratify_name
        self.k = k_fold
        self.rs = random_state
        self.column_name = column_name

    def fit(self, save_dst=None, name=None):
        self._split()

        if name is None:
            name = "data_set.csv"
        if save_dst is not None:
            self.df.to_csv(os.path.join(save_dst, name), index=False)
        return self.df

    def _split(self):
        indexes = self.df.index.to_list()
        k_labels = list(range(self.k))

        if self.stratify_name is not None:
            labels = self.df[self.stratify_name].to_list()
            kf = StratifiedKFold(n_splits=self.k, random_state=self.rs, shuffle=True)
        else:
            labels = None
            kf = KFold(n_splits=self.k, random_state=self.rs, shuffle=True)

        for idx, (_, valid_indexes) in enumerate(kf.split(X=indexes, y=labels)):
            self.df.loc[valid_indexes, self.column_name] = k_labels[idx]


# if __name__ == '__main__':

