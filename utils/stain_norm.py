# !usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2022/7/1
# @Author   : ChenLingHao
# @File     : stain_norm.py
import os
from cv2 import cv2
import argparse
from tqdm import tqdm
from glob import glob
from utils.slide_core import StainNorm
from utils.tools import FolderTool


def construct_config():
    parser = argparse.ArgumentParser("Stain Normalization")
    parser.add_argument("--data_root", help="[]")
    parser.add_argument("--tile_size", type=int, default=512, help="[]")
    parser.add_argument("-m", "--magnification", type=int, default=1, help="[]")
    parser.add_argument("--resize_shape", type=int, default=512, help="[]")

    parser.add_argument("--reference_img", help="[]")
    return parser.parse_args()


def main(args):
    normalizer = StainNorm(resize_shape=(args.resize_shape, args.resize_shape))
    root = os.path.join(args.data_root,
                        "tiles/{}_{}".format(args.magnification, args.tile_size),
                        "data")
    slide_names = os.listdir(root)

    normed_img_root = os.path.join(args.data_root,
                                   "tiles/{}_{}".format(args.magnification, args.tile_size),
                                   "stain_norm_{}".format(args.resize_shape))
    FolderTool(normed_img_root).doer()
    for slide_name in slide_names:
        FolderTool(os.path.join(normed_img_root, slide_name)).doer()
        img_list = glob(os.path.join(root, slide_name, "/*.png"))
        for img_path in tqdm(img_list, desc="[Stain Norm [{}]]...".format(slide_name)):
            img_name = os.path.basename(img_path)
            normed_img = normalizer.fit(img_path)
            cv2.imwrite(os.path.join(normed_img_root, slide_name, img_name), normed_img)


if __name__ == '__main__':
    config = construct_config()
    main(config)
