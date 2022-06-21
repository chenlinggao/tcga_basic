# !usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 上午9:44
# @Author   : ChenLingHao
# @File     : train_eval.py
import os

from tensorboardX import SummaryWriter

from core.dataset import dataloader
from core.trainer import generate_trainer
from utils.config import TrainConfig, args_printer, output_version_name
from utils.data_preparation import preparation4csv
from utils.tools import construct_logger, FolderTool, message_output

def construct_folder(args):
    """建文件夹架构, 并放到args里面, 方便其他函数调用"""
    ab_root = os.path.abspath('..')                                             # /project_root
    data_root = os.path.join(ab_root, 'data/tumor',
                             "tiles/{}_{}".format(args.magnification,
                                                  args.tile_size))              # /project_root/data/

    results_root = os.path.join(ab_root, 'results')                             # /project_root/results
    results_docs = os.path.join(results_root, 'documents')                      # /project_root/results/documents
    results_trained_models = os.path.join(results_root, 'trained_models')       # /project_root/results/trained_models

    FolderTool([results_root, results_docs, results_trained_models]).doer()
    args.project_root = ab_root
    args.data_root = data_root
    args.results_root = results_root
    args.models_dst = results_trained_models

    return args, ab_root, data_root, results_root


def construct_version_folder(args):
    """建立关于目标模型的文件夹架构, log tensorboard等"""
    version_dst = os.path.join(args.models_dst, args.version_name)
    checkpoints = os.path.join(version_dst, 'checkpoints')
    logs_dst = os.path.join(version_dst, 'logs')
    tb = os.path.join(version_dst, 'tensorboard')
    FolderTool([version_dst, checkpoints, logs_dst]).doer()
    args.model_dst = version_dst

    if args.target_label_name.split('_')[-1] == 'score':
        args.num_classes = 1
        args.task_type = 'regression'
    elif args.target_label_name.split('_')[-1] == 'label':
        args.num_classes = 2
        args.task_type = 'classification'

    tb = SummaryWriter(tb)
    logger = construct_logger(logs_dst, 'train', True)

    return args, logger, tb


def main():
    config_generator = TrainConfig("Training Config")
    args = config_generator.output_parser
    version_name = output_version_name(args)  # {task}_{target_label_name}_{backbone}_{batch_size}_{lr}
    args.version_name = version_name

    args, project_root, data_root, results_root = construct_folder(args)
    args, logger, tensorboard = construct_version_folder(args)
    args_printer(args, logger)  # 打印信息

    preparation4csv(args, logger)   # 分成交叉验证啥的

    # 训练
    message_output(input_string="{:-^100}".format(" Training "), input_logger=logger)
    if args.use_cv:
        for k in range(args.cv):
            concept = " Fold_{} ".format(k)
            message_output(input_string="{:-^80}".format(concept), input_logger=logger)
            train_loader, valid_loader = dataloader(args, k)
            trainer = generate_trainer(args, logger, tensorboard, model_result_root=args.model_dst, fold=k)
            trainer.fit(train_loader=train_loader, valid_loader=valid_loader)
    else:
        k = 0
        train_loader, valid_loader = dataloader(args, k)
        trainer = generate_trainer(args, logger, tensorboard, model_result_root=args.model_dst, fold=k)
        trainer.fit(train_loader=train_loader, valid_loader=valid_loader)


if __name__ == '__main__':
    main()