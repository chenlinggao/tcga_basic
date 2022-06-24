# !usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2022/6/22
# @Author   : ChenLingHao
# @File     : tester.py

import torch

from core.dataset import test_loader
from utils.dl_tools import AverageMeter, ModelComponent, ResultReport


class Tester:
    def __init__(self, config, target_slide_id, target_slide_label):
        self.cfg= config
        self.target = target_slide_id
        self.label = target_slide_label

    def fit(self):
        # 只测一个
        tiles_result_csv = self.predictor()
        return tiles_result_csv

    def predictor(self):
        loader = test_loader(self.cfg, self.target)
        model = self.get_model()
        loss_fn = ModelComponent(self.cfg).get_criterion()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        loss = []
        metrics = []

        model.to(device)
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                labels = torch.full_like(images, self.label).to(device)
                outputs = model(images)
                result_dict, preds_, probs_ = self.outputs2label(outputs, labels)


    def get_model(self):
        empty_model = ...
        model = ...
        return model


    def outputs2label(self, outputs, labels):
        if self.cfg.task_type == 'classification':
            pred_labels = torch.softmax(outputs, dim=1)
            pred_probs = pred_labels[:, -1].detach().cpu().numpy()
            pred_labels = (pred_labels.float() > 0.5)
            pred_labels = pred_labels[:, 1].cpu().numpy()
            metric_all = ResultReport(pred_labels, labels,
                                      pred_probabilities=pred_probs).calculate_results()
            return metric_all, pred_labels, pred_probs
