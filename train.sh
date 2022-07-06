#!/bin/bash

# ===================== Configure ===================== #
#data_root=...

# -------- data config -------- #
magnification=1
tile_size=512
resize_img=224
target_label_name=tmb_label

# -------- hyper-meters config -------- #
task=tile
epochs=20
early_stop_patience=5    # early_stop_patience < epochs
batch_size=64
learning_rate=3e-4
train_all=1
#use_cv=0   # if ture, will train with cv-fold, and output num_cv checkpoints
partial=0   # if true, test few data for training

# -------- model config -------- #
metric=auc
optimizer=sgd
backbone=resnet18   # if mil, means that is a classifier; if tile, means that is a features extractor.
#print_interval=0

python train_eval.py --train_all $train_all --backbone $backbone --epochs $epochs -b $batch_size -lr $learning_rate --partial $partial --optimizer $optimizer --metric $metric --task $task --magnification $magnification --tile_size $tile_size --target_label_name $target_label_name --resize_img $resize_img --early_stop_patience $early_stop_patience
