#!/bin/bash

# ===================== Configure ===================== #
#data_root=...

# -------- data config -------- #
#magnification=1
#tile_size=1024
#resize_img=224
#target_label_name=tmb_label

# -------- hyper-meters config -------- #
task=mil
epochs=20
batch_size=128
learning_rate=3e-4
train=1
#use_cv=0
partial=0   # if true, test few data for training

# -------- model config -------- #
metric=auc
optimizer=sgd
backbone=resnet18
#print_interval=0

python train_eval.py -e $epochs -b $batch_size -lr $learning_rate --partial $partial --optimizer $optimizer --metric $metric --task $task

