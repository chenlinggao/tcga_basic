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
epochs=6
early_stop_patience=5    # early_stop_patience < epochs
batch_size=512
learning_rate=3e-4
train_all=0 # if true, train model with no validation.
#use_cv=0   # if ture, will train with cv-fold, and output num_cv checkpoints.
partial=1   # not train with all data.

# -------- model config -------- #
metric=auc                # <options>: [acc, recall, f1, auc]
optimizer=sgd             # <options>: sgd, adam
backbone=resnet18         # if mil, means that is a classifier; if tile, means that is a features extractor.
mil_arch=attention_mil    # <options>: attention_mil,
print_interval=10         # 一个epoch内打印的次数

python train_eval.py --mil_arch $mil_arch --train_all $train_all --backbone $backbone --epochs $epochs -b $batch_size -lr $learning_rate --partial $partial --optimizer $optimizer --metric $metric --task $task --magnification $magnification --tile_size $tile_size --target_label_name $target_label_name --resize_img $resize_img --early_stop_patience $early_stop_patience --print_interval $print_interval
