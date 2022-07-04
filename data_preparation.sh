#!/bin/bash

# ===================== Configure ===================== #
debug=0
data_root=/home/msi/disk3/tcga/data/

# --- tiling config
magnification=1
tile_size=512
resize_shape=512  #
restart_totally=1 # if 0则会跳过已经完成的图片(如果进行一半被终止的文件,需要将其删除再进行重新分割,否则会跳过该图片)
#reference_img=""

# --- csv config
task=tile
documents_root=...
slide_max_tiles=...

# ===================== Main ===================== #
# step_1: slide2tile
python utils/slide_core.py -m $magnification --tile_size $tile_size --data_root $data_root --restart_totally $restart_totally --debug $debug

# <option> stain normalization (take times)
#stain_norm=1
#python utils/stain_norm.py --data_root $data_root -m $magnification --tile_size $tile_size --resize_shape $resize_shape

# step_2:  process csv documents
python utils/data_preparation.py --task $task --documents_root $documents_root --slide_max_tiles $slide_max_tiles


