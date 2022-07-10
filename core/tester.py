# !usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2022/6/22
# @Author   : ChenLingHao
# @File     : tester.py
import torch
import numpy as np
from tqdm import tqdm


class Tester:
    """预测全部"""
    def __init__(self, config):
        self.cfg = config

    def fit(self, model, loader):
        if self.cfg.task == 'tile':
            predictions = self.tile_predictor(model, loader)
        elif self.cfg.task == 'mil':
            predictions = self.mil_predictor(model, loader)
        else:
            raise NotImplementedError
        return predictions

    def tile_predictor(self, trained_model, loader):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.cfg.task_type == 'classification':
            trained_model.to(device)
            probs = []
            for idx, images in enumerate(tqdm(loader)):
                images = images.to(device)
                outputs = trained_model(images)
                probs_ = outputs.detach().cpu()
                probs.append(probs_)
            probs = torch.cat(probs).softmax(1)
            probs = probs.numpy()
            return probs[:, 1]
        else:
            raise NotImplementedError

    def mil_predictor(self, trained_model, loader):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.cfg.task_type == 'classification':
            probs = []
            for idx, features in enumerate(tqdm(loader)):
                features = features.to(device)
                outputs, attention_weight = trained_model(features, return_attention=True)
                outputs = outputs.softmax(dim=1).detach().cpu().numpy()
                prob = outputs[:, -1]
                probs.append(prob)
            probs = np.concatenate(probs, axis=0)
            return attention_weight, probs
        else:
            raise NotImplementedError
