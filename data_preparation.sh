#!/bin/bash

# ===================== Configure ===================== #
debug=0
data_root=/home/msi/disk3/tcga/data/tumor/

# --- tiling config
magnification=1
tile_size=512
restart_totally=1 # if 0则会跳过已经完成的图片(如果进行一半被终止的文件,需要将其删除再进行重新分割,否则会跳过该图片)

# ===================== Main ===================== #
# step_1: slide2tile
python utils/slide_core.py -m $magnification --tile_size $tile_size --data_root $data_root --restart_totally $restart_totally --debug $debug

# step_2:  process csv documents
python data_preparation.py --task tile --data_root $data_root -m $magnification --tile_size $tile_size
python data_preparation.py --task mil --data_root $data_root -m $magnification --tile_size $tile_size

# <option> stain normalization (take times)
#stain_norm=1
#resize_shape=224
#backbone=resnet18
#task=mil   # if task==mil, will calculate stain_normed images.
#batch_size=128
#python stain_norm.py --data_root $data_root -m $magnification --tile_size $tile_size --resize_shape $resize_shape --backbone $backbone --batch_size $batch_size --task $task
