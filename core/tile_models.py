# !usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2022/6/20
# @Author   : ChenLingHao
# @File     : tile_models.py
import sys
import timm
from torchvision import models

def get_classifier(config):
    model = timm.create_model(model_name=config.backbone, pretrained=config.pretrained)
    return model

# def load_dict(checkpoint, )
