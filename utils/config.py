# !usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 上午9:51
# @Author   : ChenLingHao
# @File     : random_state.py

import os
import argparse
from distutils.util import strtobool

from utils.tools import message_output


class _BasicConfig(object):
    def __init__(self, config_name: str):
        self.parser = argparse.ArgumentParser(config_name)

        self.path_parser = self.parser.add_argument_group(title="[Path Setting]")
        self.others_parser = self.parser.add_argument_group(title="[Others Setting]")
        self.others_parser.add_argument("--random_state", type=int, default=2022)

    def _fit_all_config(self):
        return self.parser.parse_args()

    @property
    def output_parser(self):
        return self._fit_all_config()

    def print_config(self, logger=None):
        config_dict = vars(self.parser.parse_args())
        for k, v in config_dict.items():
            output_string = "{: >20} ==> {}".format(k, v)
            message_output(input_string=output_string, input_logger=logger)


class Slide2TileConfig(_BasicConfig):
    def __init__(self, config_name: str):
        super().__init__(config_name)
        self.tiles_parser = self.parser.add_argument_group(title="[Tile Setting]")

    def add_path_config(self):
        self.path_parser.add_argument("--data_root",
                                      default="/home/msi/disk3/tcga/data/test",
                                      help="所有图片数据的根目录")

    def add_tile_config(self):
        self.tiles_parser.add_argument("--tile_size", default=512, type=int, help="tile的大小")
        self.tiles_parser.add_argument("-m", "--magnification", default=1, type=int, help="病理图的放大倍数，用于计算合适的target_level")
        self.tiles_parser.add_argument("--plot_tile", default=0, type=lambda x: bool(strtobool(x)), help="执行时候看tile的情况")

        self.tiles_parser.add_argument("--ratio_threshold", default=0.8, type=float, help="背景色的比例")
        self.tiles_parser.add_argument("--white_threshold", default=206, type=int, help="过滤白色背景的[0-255]的阈值")
        self.tiles_parser.add_argument("--black_threshold", default=20, type=int, help="过滤黑色背景的[0-255]的阈值")

    def add_others_config(self):
        self.others_parser.add_argument("--debug", default=0, type=lambda x: bool(strtobool(x)), help="测试是否有错")
        # self.others_parser.add_argument("--folder_dst", help="当前tile_size和level的存储根路径")
        # self.others_parser.add_argument("--folder_tiles_dst", help="当前tile_size和level的tiles存储路径")
        # self.others_parser.add_argument("--documents_root", help="所有文件数据的根目录")
        # self.others_parser.add_argument("--documents_csv_folder", help="每个slide的tile的信息目录")
        self.others_parser.add_argument("--restart_totally", default=1, type=lambda x: bool(strtobool(x)), help="每个slide的tile的信息目录")

    def _fit_all_config(self):
        self.add_path_config()
        self.add_tile_config()
        self.add_others_config()
        return self.parser.parse_args()


class TrainConfig(_BasicConfig):
    def __init__(self, config_name: str):
        super().__init__(config_name)
        self.task_parser = self.parser.add_argument_group(title="[Task Setting]")
        self.component_parser = self.parser.add_argument_group(title="[Component Setting]")
        self.hyper_parser = self.parser.add_argument_group(title="[Hyper-parameter Setting]")

    def add_task_config(self):
        self.task_parser.add_argument("--task", default='tile', help="[Options]: 'tile', 'mil'")
        self.task_parser.add_argument("--target_label_name", default="tmb_label",
                                      help="[Options]: 'tmb_label', 'tmb_score'")
        self.task_parser.add_argument("--magnification", default=1, type=int,
                                      help="病理图的放大倍数，用于计算合适的target_level")
        self.task_parser.add_argument("--tile_size", default=1024, type=int,
                                      help="tile的大小")
        self.task_parser.add_argument("--resize_img", default=512, type=int, help="[]: ")
        self.task_parser.add_argument("--slide_max_tiles", default=2000, type=int, help="[]: 每个slide最多拿出的tile的数量")

    def add_path_config(self):
        self.path_parser.add_argument("--data_root", default="/home/msi/disk3/tcga", help="所有图片数据的根目录")
        self.path_parser.add_argument("--document_root", default="../documents", help="所有文件数据的根目录")

    def add_hyper_config(self):
        self.hyper_parser.add_argument("-e", "--epochs", default=20, type=int, help="[]: 迭代次数")
        self.hyper_parser.add_argument("-b", "--batch_size", default=128, type=int, help="[]: batch的大小")
        self.hyper_parser.add_argument("-lr", "--learning_rate", default=3e-4, type=float, help="[]: 学习率")
        self.hyper_parser.add_argument("--pretrained", default=1, type=lambda x: bool(strtobool(x)), help="[]: ")
        self.hyper_parser.add_argument("--metric", default="f1", help="[]: ")
        # self.hyper_parser.add_argument("--", default=, type=, help="[]: ")
        # self.hyper_parser.add_argument("--", default=, type=, help="[]: ")

    def add_component_config(self):
        # self.component_parser.add_argument("-mm", "--mil_method", default=, type=, help="[]: ")
        self.component_parser.add_argument("--backbone", default='resnet18', help="[]: ")
        self.component_parser.add_argument("--criterion", default='ce', help="[]: ")
        self.component_parser.add_argument("--optimizer", default='adam', help="[]: ")
        self.component_parser.add_argument("--scheduler", default='cosine', help="[]: ")
        self.component_parser.add_argument("--warm_up_epochs", default=0, type=int, help="[]: ")

        # self.component_parser.add_argument("--ema", default=, type=, help="[]: ")
        self.component_parser.add_argument("--early_stop_standard", default='loss', help="[]: ")
        self.component_parser.add_argument("--early_stop_patience", default=200, type=int, help="[]: ")
        self.component_parser.add_argument("--early_stop_delta", default=3e-4, type=float, help="[]: ")

    def add_others_config(self):
        self.others_parser.add_argument("--train", default=1, type=lambda x: bool(strtobool(x)),
                                        help="if set true, will train the model")
        self.others_parser.add_argument("--partial", default=1, type=lambda x: bool(strtobool(x)),
                                        help="if set true, use 20% data to train the model")
        self.others_parser.add_argument("--cv", default=5, type=int,
                                        help="set number of fold for cross validation, usually[5, 10]")
        self.others_parser.add_argument("--use_cv", default=0, type=lambda x: bool(strtobool(x)),
                                        help="if set true, use cross validation")
        self.others_parser.add_argument("--print_interval", default=10,
                                        type=int, help="setting for print batch information ")

    def _fit_all_config(self):
        # self.add_path_config()
        self.add_task_config()
        self.add_hyper_config()
        self.add_component_config()
        self.add_others_config()
        return self.parser.parse_args()


class TestConfig(_BasicConfig):
    def __init__(self, config_name: str):
        super().__init__(config_name)
        self.task_parser = self.parser.add_argument_group(title="[Task Setting]")
        self.component_parser = self.parser.add_argument_group(title="[Component Setting]")
        self.hyper_parser = self.parser.add_argument_group(title="[Hyper-parameter Setting]")

    def add_task_config(self):
        ...

    def add_path_config(self):
        self.path_parser.add_argument("--data_root", default="/home/msi/disk3/tcga", help="所有图片数据的根目录")
        self.path_parser.add_argument("--document_root", default="documents", help="所有文件数据的根目录")

    def add_component_config(self):
        ...

    def add_hyper_config(self):
        ...

    def add_others_config(self):
        ...

    def _fit_all_config(self):
        self.add_path_config()
        self.add_task_config()
        self.add_hyper_config()
        self.add_component_config()
        self.add_others_config()
        return self.parser.parse_args()

def args_printer(args, input_logger, filter_=None):
    """打印args信息"""
    if filter_ is None:
        filter_ = ['epochs', 'backbone', 'train', 'version_name']

    input_logger.info("{:-^100}".format("Setting"))
    for k, v in vars(args).items():
        output_string = "{: >20} ==> {}".format(k, v)
        if k in filter_:
            output_string_ = '{}\n'.format('-'*50)
            message_output(output_string_, input_logger)
        message_output(input_string=output_string, input_logger=input_logger)


def output_version_name(args):
    p = args
    name = "{}_{}_{}_{}_{}".format(
        p.task, p.backbone, p.target_label_name, p.batch_size, p.learning_rate
    )
    if p.task == 'mil':
        name = "{}_{}_{}_{}_{}_{}".format(
        p.task, p.mil_method, p.backbone, p.target_label_name, p.batch_size, p.learning_rate
    )
    return name










