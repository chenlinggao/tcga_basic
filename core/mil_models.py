# !usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2022/6/20
# @Author   : ChenLingHao
# @File     : mil_models.py
import torch
import torchvision
import numpy as np

"""
Config:
    
"""
channels_dict = dict(resnet18=[(224, 25088), (256, 32768), (512, 131072), (1024, 524288)],
                     resnet50=[(224, 100352), (256, 131072), (512, 524288), (1024, 2097152)],
                     )


class MILArchitecture(torch.nn.Module):
    def __int__(self, config):
        self.cfg = config

        self.n = config.num_classes
        self.mil_input_channels = self.set_input_channels()
        self.mil_arch = self.get_mil_arch()
        self.classifier = torch.nn.Linear(in_features=self.mil_input_channels, out_features=config.num_classes)

    def forward(self, features, return_attention=False):
        embedding_features, attention_weight = self.mil_arch(features)  # KxL
        output = self.classifier(embedding_features)
        if return_attention:
            return output, attention_weight
        else:
            return output, None

    def get_mil_arch(self):
        if self.cfg.mil_arch == 'attention_mil':
            mil_arch = AttentionMIL(L=self.mil_input_channels)
        else:
            raise NotImplementedError
        return mil_arch

    def set_input_channels(self):
        # 后续补充，先预设为224
        if self.cfg.backbone == 'resnet18':
            mil_input_channels = channels_dict['resnet18'][0][1]
        else:
            raise NotImplementedError
        return mil_input_channels


# paper:
class AttentionMIL(torch.nn.Module):
    def __int__(self, L=1024, D=512, K=1):
        self.L = L
        self.D = D
        self.K = K
        self.attention_V = torch.nn.Sequential(
            torch.nn.Linear(self.L, self.D),
            torch.nn.Tanh()
        )
        self.attention_U = torch.nn.Sequential(
            torch.nn.Linear(self.L, self.D),
            torch.nn.Sigmoid()
        )
        self.attention_weights = torch.nn.Linear(self.D, self.K)

    def forward(self, features):
        # Attention weights computation & features.shape == (B, L)
        A_V = self.attention_V(features)         # Attention     (B, D)
        A_U = self.attention_U(features)         # Gate          (B, D)
        attention_w = self.attention_weights(A_V.mul(A_U))     # (B, K)

        attention_w_ = attention_w.tranpose(1, 0)               # (K, B)
        attention_features = torch.mm(attention_w_, features)   # (K, L)

        return attention_features, attention_w


