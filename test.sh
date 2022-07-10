#!/bin/bash

# ===================== Configure ===================== #
trained_model_name=tile_resnet18_tmb_label_512_0.0003_whole    # must to input

# -------- data config -------- #
task=tile
magnification=1
tile_size=512
resize_img=224
batch_size=256    # if the memory of cuda is big, can set bigger; if not, set 64/128/256
backbone=resnet18 # <options>: [vgg13, resnet18, resnet34, resnet50]

target_label_name=tmb_label # <options>: [*_score, *_label].
                            # '*' means biomarkers' name
                            # score is considered a 'regression' mission
                            # score is considered a 'classification' mission


python test.py --trained_model_name $trained_model_name