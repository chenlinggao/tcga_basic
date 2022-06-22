# !usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 下午4:44
# @Author   : ChenLingHao
# @File     : datasets.py
"""
    train-test
    fuse_csv
    train_dataset
    test_dataset
"""
import os
import h5py
import argparse
import pandas as pd
import _pickle as pickle
from glob import glob
from tqdm import tqdm
from random import shuffle

from utils.tools import TrainValidTestSplit, TrainValidTestSplit_k_fold, message_output


class PrepareTileSet:
    """
        1. 随机从一个slide的csv中抽取5000个tile
        2. 将这些tile的信息都整合在一个csv中
    """
    def __init__(self, config):
        self.cfg = config
        self.documents_root = os.path.join(config.data_root, 'documents')
        self.csv_root = os.path.join(self.documents_root, 'slides_tiles_csv')
        self.gene_df = pd.read_csv(os.path.join(self.documents_root, 'fused_slides_gene_info.csv'))
        self.gene_slide_ids = self.gene_df.slide_id.to_list()
        self.output_df = None

    def preprocess_slides_info_csv(self):
        """
            将slide info csv分成['train', 'test'].
            提取'train'进行训练.
        """
        splitor = TrainValidTestSplit(data_csv=self.gene_df, ratio=[8, 2],
                                      stratify_name=self.cfg.target_label_name, names='tt')
        self.gene_df = splitor.fit()
        self.gene_df.to_csv(os.path.join(self.documents_root, 'fused_slides_gene_info.csv'), index=False)
        self.gene_df = self.gene_df[self.gene_df.phase == 'train'].reset_index(drop=True)

    def preprocess_tile_csv(self, tile_df):
        target_slide_id = tile_df.loc[0, 'slide_id'].split('.')[0]
        target_case_gene_info = self.gene_df[self.gene_df.slide_id == target_slide_id]
        if not target_case_gene_info.empty:
            tile_df['tmb_label'] = target_case_gene_info['tmb_label'].to_list()[0]
            tile_df['tmb_score'] = target_case_gene_info['tmb_score'].to_list()[0]
            return tile_df
        else:
            return None

    def fit(self):
        # -------------- split slides_info.csv ---------------
        self.preprocess_slides_info_csv()

        # -------------- fuse slide_tiles.csv ---------------
        tile_csv_list = glob(self.csv_root + '/*.csv')
        for tile_csv_src in tqdm(tile_csv_list):
            if os.path.basename(tile_csv_src).split('.')[0] not in self.gene_slide_ids:
                continue
            df = pd.read_csv(tile_csv_src)
            df = self.preprocess_tile_csv(df)
            if df is None:
                continue    # 跳过测试集
            sampled_df = self.sample_(input_df=df)
            if self.output_df is None:
                self.output_df = sampled_df
            else:
                self.output_df = pd.concat([self.output_df, sampled_df])

        df = self.output_df.reset_index(drop=True)
        """分成train_test_valid"""
        if self.cfg.cv <= 2:
            splitor = TrainValidTestSplit(data_csv=df, ratio=[8, 2], stratify_name=self.cfg.target_label_name,
                                          random_state=self.cfg.random_state)
        else:
            splitor = TrainValidTestSplit_k_fold(data_csv=df, k_fold=self.cfg.cv,
                                                 stratify_name=self.cfg.target_label_name)
        df = splitor.fit()
        df['phase'] = df.phase.apply(lambda x: int(x))
        df.to_csv(os.path.join(self.documents_root, 'train_dataset_{}.csv'.format(self.cfg.task)), index=False)
        with open(os.path.join(self.documents_root, 'train_dataset_{}.pkl'.format(self.cfg.task)), 'wb') as f:
            pickle.dump(df, f)

    def sample_(self, input_df):
        df = input_df[input_df.tile_type == 'tissue'].reset_index(drop=True)
        target_index = list(range(len(df)))
        shuffle(target_index)
        target_index = target_index[:self.cfg.slide_max_tiles]
        sampled_df = df.iloc[target_index, :self.cfg.slide_max_tiles]
        return sampled_df


class PrepareMilSet(PrepareTileSet):
    def __init__(self, config):
        super().__init__(config)
        # self.data_save_type = config.data_save_type

    def preprocess_slides_info_csv_k_fold(self):
        self.preprocess_slides_info_csv()
        if self.cfg.cv >= 2:
            splitor = TrainValidTestSplit_k_fold(data_csv=self.gene_df, k_fold=self.cfg.cv,
                                                 stratify_name=self.cfg.target_label_name, colmun_name='phase')
        else:
            splitor = TrainValidTestSplit(data_csv=self.gene_df, ratio=[8, 2],
                                      stratify_name=self.cfg.target_label_name, names='tt')
        self.gene_df = splitor.fit()
        self.gene_df.to_csv(os.path.join(self.documents_root, 'fused_slides_gene_info.csv'), index=False)

        output = self.gene_df[self.gene_df.phase != 'test'].reset_index(drop=True)
        return output

    def fit(self):
        # 将slide分成交叉验证的三个部分
        train_valid_df = self.preprocess_slides_info_csv_k_fold()

        # 将提取每个slide的tile的特征，并集成在一个pkl/h5中
        for idx, slide_id in enumerate(self.gene_df.slide_id):
            tile_folder = os.path.join(self.cfg.data_root, 'data', slide_id)
            tiles_dst = glob(tile_folder+'/*.png')



def fuse_slides_tmb_info(config):
    """把slide信息和tmb信息合并在一起"""
    documents_root = os.path.join(config.data_root, 'documents')
    slides_df = pd.read_csv(os.path.join(documents_root, 'all_slides_info.csv'))
    tmb_df = pd.read_csv(os.path.join(documents_root, 'gene_info.csv'))

    for slide_id in slides_df.slide_id:
        case_id = slide_id[:12]
        target_df_index = slides_df[slides_df.slide_id.isin([slide_id])].index

        target_tmb_row = tmb_df[tmb_df.Patient_ID.isin([case_id])]
        if target_tmb_row.empty:
            o_msg = "[Warning] [{}] Not Exist, and pass it".format(slide_id)
            print(o_msg)

            # empty_index = target_df_index
            slides_df = slides_df.drop(target_df_index)
            continue

        tmb_score = target_tmb_row.tmb.to_list()[0]
        tmb_label = 0
        if tmb_score > 20:
            tmb_label = 1

        slides_df.loc[target_df_index, 'tmb_score'] = tmb_score
        slides_df.loc[target_df_index, 'tmb_label'] = tmb_label
        slides_df.loc[target_df_index, 'survival_overall'] = target_tmb_row.survival_overall.to_list()[0]

    slides_df.to_csv(os.path.join(documents_root, 'fused_slides_gene_info.csv'), index=False)
    return 0

def setting_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="选择任务类型: ['tile', 'mil]")
    parser.add_argument("--documents_root", help="所有文件信息的根目录")
    parser.add_argument("--cv", default=5)
    parser.add_argument("--slide_max_tiles", type=int, help="每个slide最多拿出的tile的数量")
    parser.add_argument("--random_state", help="固定随机数", default=2022)
    parser.add_argument("--target_label_name", help="")

    return parser.parse_args()

def preparation4csv(args, input_logger):
    message_output("\n{:-^50}".format(" Preparing CSVs "), input_logger)
    fuse_slides_tmb_info(args)
    # if pass_flag:
        # message_output("[Warning]: Finished before", input_logger)
    # else:
    #     PrepareTileSet(args).fit()
    PrepareTileSet(args).fit()

def main():
    args = setting_config()
    args.task = 'tile'
    args.slide_max_tiles = 20
    args.data_root = '/home/msi/chenlinghao/future/tcga_colon/data/tumor/1_512/tiles'
    args.target_label_name = 'tmb_label'
    fuse_slides_tmb_info(args)

    if args.task == 'tile':
        PrepareTileSet(config=args).fit()
    elif args.task == 'mil':
        ...


if __name__ == '__main__':
    main()
