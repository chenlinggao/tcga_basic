#!/bin/bash

# ===================== Configure ===================== #
#data_root=...
trained_model_name=...

# -------- data config -------- #
magnification=1
tile_size=512
resize_img=224
target_label_name=tmb_label

# -------- hyper-meters config -------- #
task=tile
batch_size=512    # if the memory of cuda is big, can set bigger; if not, set 64/128/256
metric=auc        # <options>: [acc, recall, f1, auc]


python test.py --trained_model_name $trained_model_name