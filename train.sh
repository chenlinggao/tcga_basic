#!/bin/bash

epochs=30
batch_size=128
learning_rate=3e-4
#metric=f1
#backbone=resnet18

#task=tile
#target_label_name=tmb_label
#magnification=1
#tile_size=1024
#resize_img=334
#slide_max_tiles=500
#
#warm_up_epochs=10
#early_stop_patience=200
#
#train=1
#debug=0
#use_cv=0
#print_interval=0

python train_eval.py -e $epochs -b $batch_size -lr $learning_rate

