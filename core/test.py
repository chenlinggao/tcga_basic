# !usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2022/7/5
# @Author   : ChenLingHao
# @File     : test.py
import torch
from torch import nn


class ConvNet(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.i = i
        self.p = 0






