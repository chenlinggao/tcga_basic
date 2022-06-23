# !usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 下午6:43
# @Author   : ChenLingHao
# @File     : dl_tools.py
import os
import random
import _pickle as pickle
from random import shuffle
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import WarmUpLR
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, precision_score, roc_curve, auc
from sklearn.metrics import confusion_matrix


def set_seed(input_seed):
    """固定随机种子"""
    random.seed(input_seed)
    np.random.seed(input_seed)
    torch.manual_seed(input_seed)
    torch.cuda.manual_seed(input_seed)


class BasicTrainer:
    def __init__(self, config, logger, tensorboard, model_result_root, fold=None):
        self.cfg = config
        self.logger = logger
        self.tb = tensorboard
        self.model_result_root = model_result_root
        self.k = fold
        self.print_counter = 0

        set_seed(input_seed=config.random_state)

        self.model = self.get_model()

        component_creator = ModelComponent(input_config=self.cfg, logger=logger)
        self.optimizer = component_creator.get_optimizer(self.model)
        self.scheduler = component_creator.get_scheduler(optimizer=self.optimizer)
        self.loss_fn = component_creator.get_criterion()
        self.early_stopper = component_creator.get_early_stopper()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
        self.loss_fn.to(self.device)


    def fit(self, train_loader, valid_loader):
        # train
        self.logger.info("\n"
                         "------------------------------------ Start Training ------------------------------------")
        for epoch in range(1, self.cfg.epochs + 1):
            early_stopper_flag = self.train_one_epoch(train_loader, valid_loader,
                                                      epoch, self.model_result_root,
                                                      early_stopper=self.early_stopper)
            if early_stopper_flag:
                break

    def train_one_epoch(self, train_loader, valid_loader, epoch, model_result_root, early_stopper):
        return None

    def get_model(self):
        return None

    def printer(self):
        # 输出到tensorboard之类的
        ...

class EMA:
    ...


class EarlyStopper:
    def __init__(self, phase='loss', patience=7, delta=0.5, logger=None, work=None):
        """

        :param phase:
        :param patience:
        :param delta:
        :param logger:

        [Example]:
            early_stopper = EarlyStopper(...)

        """
        self.phase = phase
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_flag = False

        self.logger = logger
        self.work = work

    def __call__(self, input_value):
        score = self.preprocess_input_value(input_value)

        if self.best_score is None:
            # 第一轮
            self.best_score = score
            self.save_flag = True

        elif (self.best_score - self.delta) < score < (self.best_score + self.delta):
            # 不满足条件的score
            self.counter += 1

            # 当work为True则打印, 否则不打印
            if self.work:
                if self.logger is not None:
                    self.logger.info(f'EarlyStopping counter: [{self.counter} / {self.patience}]')
                else:
                    print(f'EarlyStopping counter: [{self.counter} / {self.patience}]')
            if self.counter >= self.patience:
                self.save_flag = False
                self.early_stop = True

        else:
            # 满足条件的score
            self.best_score = score
            self.counter = 0
            self.save_flag = True

    def preprocess_input_value(self, value):
        if self.phase == 'loss':
            return -value
        else:
            return value


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch, input_logger=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if input_logger is None:
            print('\t'.join(entries))
        else:
            input_logger.info('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class ModelComponent:
    """[OUTPUT]: model, criterion, optimizer, warm_up_scheduler, scheduler, early_stopper"""

    def __init__(self, input_config, logger=None):
        self.cfg = input_config
        self.logger = logger

    def get_criterion(self):
        if self.cfg.criterion == "mse":
            return nn.MSELoss()
        elif self.cfg.criterion == "ce":
            return nn.CrossEntropyLoss()
        elif self.cfg.criterion == "bce":
            if self.cfg.mil_instance_mode == 'embedding':
                return nn.BCEWithLogitsLoss()
            elif self.cfg.mil_instance_mode == 'instance':
                return nn.BCELoss()
        elif self.cfg.criterion == "smooth_l1_loss":
            return nn.SmoothL1Loss()
        else:
            raise NotImplementedError

    def get_optimizer(self, model):
        if self.cfg.optimizer == 'sgd':
            return optim.SGD(params=model.parameters(), lr=self.cfg.learning_rate, momentum=.9)
        elif self.cfg.optimizer == 'adam':
            return optim.Adam(params=model.parameters(), lr=self.cfg.learning_rate)
        elif self.cfg.optimizer == 'adamw':
            return optim.AdamW(params=model.parameters(), lr=self.cfg.learning_rate)
        else:
            raise NotImplementedError

    def get_early_stopper(self):
        early_stop_work = True if self.cfg.early_stop_patience < self.cfg.epochs else False
        output = EarlyStopper(phase=self.cfg.early_stop_standard,
                              patience=self.cfg.early_stop_patience,
                              delta=self.cfg.early_stop_delta,
                              logger=self.logger,
                              work=early_stop_work)
        return output

    def get_scheduler(self, optimizer):
        warm_up_scheduler, scheduler = None, None
        if self.cfg.warm_up_epochs > 0:
            warm_up_scheduler = WarmUpLR(optimizer=optimizer,
                                         total_iters=self.cfg.warm_up_epochs)

        if self.cfg.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             T_max=self.cfg.epochs)
        elif self.cfg.scheduler == "cosine_restart":
            warm_up_scheduler = None
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                       T_0=5, T_mult=10)


        return [warm_up_scheduler, scheduler]

    def component_info(self):
        if self.logger:
            self.logger.info("[Info]:\n"
                             "\t[Component]: \n"
                             "\t\t [Model]: {}\n"
                             "\t\t [Optimizer]: {}\n"
                             "\t\t [Scheduler]: {}\n"
                             "\t\t [Warm up Epoch]: {}\n"
                             "\t\t [Criterion]: {}\n".format(self.cfg.model, self.cfg.optimizer,
                                                             self.cfg.scheduler, self.cfg.warm_up_epochs,
                                                             self.cfg.criterion))
        else:
            print("[Info]:\n"
                  "\t[Component]: \n"
                  "\t\t [Model]: {}\n"
                  "\t\t [Optimizer]: {}\n"
                  "\t\t [Scheduler]: {}\n"
                  "\t\t [Warm up Epoch]: {}\n"
                  "\t\t [Criterion]: {}\n".format(self.cfg.model, self.cfg.optimizer,
                                                  self.cfg.scheduler, self.cfg.warm_up_epochs,
                                                  self.cfg.criterion))


class ResultReport:
    def __init__(self, pred_label, ground_truth, pred_probabilities=None, **kwargs):
        if isinstance(pred_label, torch.Tensor):
            pred_label = pred_label.detach().cpu().numpy()
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.cpu().numpy()

        self.gt = ground_truth
        self.pred = pred_label
        self.pred_prob = pred_probabilities

    @staticmethod
    def calculate_specificity(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn/(tn+fp)

    def calculate_results(self, class_list=None):
        result_dict = {}
        if class_list is None:
            class_list = ['Accuracy', 'Recall', 'Precision', 'F1', 'Specificity', 'Sensitivity', 'AUC']
        for target in class_list:
            _value = self.calculate_single_result(target)
            result_dict[target] = round(_value, 4)
        return result_dict

    def calculate_single_result(self, method_name):
        if method_name.lower() == 'accuracy':
            return accuracy_score(self.gt, self.pred)
        elif method_name.lower() == 'precision':
            return precision_score(self.gt, self.pred)
        elif method_name.lower() == 'recall':
            return recall_score(self.gt, self.pred)
        elif method_name.lower() == 'f1':
            return f1_score(self.gt, self.pred)
        elif method_name.lower() == 'sensitivity':
            return recall_score(self.gt, self.pred)
        elif method_name.lower() == 'specificity':
            return self.calculate_specificity(self.gt, self.pred)
        elif method_name.lower() == 'auc':
            return roc_auc_score(self.gt, self.pred_prob)
