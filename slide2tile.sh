#!/bin/bash
#data_root_tumor=/home/msi/disk3/tcga/test

data_root_tumor=/home/msi/disk3/tcga/data/tumor/
data_root_normal=/home/msi/disk3/tcga/data/normal

tile_size=512    # 生成图像块的大小
magnification=10   # 病理图的放大倍数
restart_totally=1 # if 0则会跳过已经完成的图片(如果进行一半被终止的文件,需要将其删除再进行重新分割,否则会跳过该图片)
debug=0

python utils/slide_core.py -m $magnification --tile_size $tile_size --data_root $data_root_tumor --restart_totally $restart_totally --debug $debug

python utils/slide_core.py -m $magnification --tile_size $tile_size --data_root $data_root_normal --restart_totally $restart_totally --debug $debug
