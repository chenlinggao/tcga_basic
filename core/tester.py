# !usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2022/6/22
# @Author   : ChenLingHao
# @File     : tester.py
import os
import numpy as np
import pandas as pd

import torch

from core.dataset import test_loader
from core.tile_models import get_classifier
from utils.dl_tools import AverageMeter, ModelComponent, ResultReport


class Predictor:
    def __init__(self, config, trained_model, target_slide_id, target_slide_label):
        self.cfg= config
        self.target = target_slide_id
        self.label = target_slide_label
        self.model = trained_model

    def fit(self):
        # 只测一个
        if self.cfg.task == 'tile':
            prob, slide_label = self.tile_predictor()
        else:
            prob, slide_label = self.mil_predictor()
        return prob, slide_label

    def tile_predictor(self):
        loader = test_loader(self.cfg, self.target)
        loss_fn = ModelComponent(self.cfg).get_criterion()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        probs, preds = [], []

        self.model.to(device)
        # 后续需要针对不同的classification和regression增加不同的test策略
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                outputs = self.model(images)
                # 把outputs保存出来，然后再计算result
                if self.cfg.task_type == 'classification':
                    probs_ = outputs[:, 1].detach().cpu().numpy().float()
                    preds_ = probs_[probs_ > 0.5]
                probs = probs.append(probs_)
                preds = preds.append(preds_)

        probs = np.concatenate(probs, axis=0)
        preds = np.concatenate(preds, axis=0)
        gts = np.full_like(probs, self.label)
        loss = loss_fn(preds, gts)
        # result_dict = self.outputs2label(ground_truth=gts, predicts=preds, probabilities=probs)
        # result_dict.update({"loss": loss})
        # return result_dict
        global_prob = np.mean(preds)
        global_label = 1 if (global_prob > 0.5) else 0

        return global_prob, global_label

    def mil_predictor(self):
        slide_label = ...
        return slide_label


def load_trained_model(config):
    checkpoint = torch.load(config.trained_model_src)

    if config.task == 'tile':
        model = get_classifier(config)
        model.load_state_dict(checkpoint['model_dict'])
    else:
        raise NotImplementedError
    return model

def test(config):
    probs, preds, gts = [], [], []
    # 遍历获取test的slide数据等路径
    slide_info = pd.read_csv(os.path.join(config.documents_root, 'fused_slides_gene_info.csv'))
    test_slide_info = slide_info[slide_info.phase == 'test']
    model = load_trained_model(config)

    for idx, row in test_slide_info.iterrows():
        slide_id = row.slide_id
        tile_dst = row.tile_dst
        label = row[config.target]
        predictor = Predictor(config=config, trained_model=model, target_slide_id=slide_id, target_slide_label=label)
        prob, predict_label = predictor.fit()

        probs.append(prob)
        preds.append(predict_label)
        gts.append(label)

    result_dict = ResultReport(ground_truth=gts, pred_probabilities=probs, pred_label=preds).calculate_results()
    return result_dict



class Tester:
    """把预测全部"""
    def __init__(self, config):
        self.cfg = config

    def fit(self, model, loader):
        if self.cfg.task == 'tile':
            predictions = self.tile_predictor(model, loader)

        else:
            raise NotImplementedError
        return predictions

    def tile_predictor(self, trained_model, loader):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.cfg.task_type == 'classification':
            probs = []
            for idx, images in enumerate(loader):
                images = images.to(device)
                outputs = trained_model(images)
                probs_ = outputs[:, -1].detach().cpu().numpy().float()
                trained_model.append(probs_)
                probs.append(probs_)
            probs = np.concatenate(probs, axis=0)
            return probs
        else:
            ...

    def mil_predictor(self):
        if self.cfg.task_type == 'classification':
            ...
        else:
            ...

