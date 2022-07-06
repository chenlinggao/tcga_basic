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
import _pickle as pickle
from tile_models import tiles2features
from utils.slide_core import StainNorm
from utils.tools import FolderTool


def construct_config():
    parser = argparse.ArgumentParser("Stain Normalization")
    parser.add_argument("--data_root", help="[]")
    parser.add_argument("--tile_size", type=int, default=512, help="[]")
    parser.add_argument("-m", "--magnification", type=int, default=1, help="[]")
    parser.add_argument("--resize_shape", type=int, default=512, help="[]")
    parser.add_argument("--task", default="mil")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--reference_img", help="[]")
    return parser.parse_args()


def main(args):
    normalizer = StainNorm(resize_shape=(args.resize_shape, args.resize_shape))
    img_root = os.path.join(args.data_root,
                            "tiles/{}_{}".format(args.magnification, args.tile_size),
                            "data/original")
    slide_names = os.listdir(img_root)

    normed_img_root = os.path.join(args.data_root,
                                   "tiles/{}_{}".format(args.magnification, args.tile_size),
                                   "data/sn_{}".format(args.resize_shape))
    normed_pkl_root = os.path.join(args.data_root,
                                   "tiles/{}_{}".format(args.magnification, args.tile_size),
                                   "features/sn_{}".format(args.resize_shape))
    FolderTool(normed_img_root).doer()
    for slide_name in slide_names:
        FolderTool(os.path.join(normed_img_root, slide_name)).doer()
        img_list = glob(os.path.join(img_root, slide_name, "/*.png"))
        for img_path in tqdm(img_list, desc="[Stain Norm [{}]]...".format(slide_name)):
            img_name = os.path.basename(img_path)
            normed_img = normalizer.fit(img_path)
            cv2.imwrite(os.path.join(normed_img_root, slide_name, img_name), normed_img)

        print("\n[Stain_normed Tiles to Features ......]")
        tile_list = glob(os.path.join(normed_img_root, slide_name, '*.png'))
        features = tiles2features(config, tile_list)
        with open(os.path.join(normed_pkl_root, slide_name + '.pkl'), 'wb') as f:
            pickle.dump(features, f)


if __name__ == '__main__':
    config = construct_config()
    main(config)
