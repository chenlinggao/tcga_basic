# !usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 下午8:56
# @Author   : ChenLingHao
# @File     : slide_core.py
import os
import sys
sys.path.append('..')

from tqdm import tqdm
import pandas as pd
from glob import glob

import cv2
import numpy as np
import openslide
import staintools
from PIL import Image
import matplotlib.pyplot as plt

from .tools import message_output, FolderTool, construct_logger
from .config import Slide2TileConfig
from .image_tools import hsv_otsu_image, plot_multi_subplot_one_row, binary_image

import logging
logging.getLogger('matplotlib.font_manager').disabled = True

def isBackground(input_image, ratio_threshold=0.7, white_threshold=None, black_threshold=None):
    """
        # 可能有既有白也有黑的情况，但先不解决
        # 还有一些质量不好的图片可能参杂在里面，需要考虑是否、如何在训练集中滤去
    """
    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image)[:, :, :3]

    # 如果同时输入两个threshold，就报错
    if white_threshold is not None and black_threshold is not None:
        assert "[Error] Don't input white_threshold and black_threshold at the same time."
    elif white_threshold is None and black_threshold is None:
        assert "[Error] Please input white_threshold or black_threshold at the same time."

    filter_flag = False

    img_min_pixel = input_image.min()
    if img_min_pixel > 100:
        # 是否包含有细胞
        filter_flag = True
        return filter_flag

    # 判断背景区域的占比是否超过ratio_threshold
    white_count, black_count = 0, 0
    if white_threshold is not None:
        white_count = np.sum(input_image > white_threshold)
    if black_threshold is not None:
        black_count = np.sum(input_image < black_threshold)

    # 判断是否超过ratio_threshold
    ratio = (white_count + black_count) / input_image.size
    if ratio > ratio_threshold:
        filter_flag = True
    return filter_flag


def isBackground_otsu(input_image, ratio_threshold=0.7, black_threshold=None, otsu_threshold=None):
    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image)[:, :, :3]

    _, overlap_img = binary_image(input_image, low_boundary=otsu_threshold, overlap=True)

    filter_flag = False
    black_count = np.sum(overlap_img < black_threshold)
    ratio = black_count / overlap_img.size

    if ratio > ratio_threshold:
        filter_flag = True
    return filter_flag



class SlideReader:
    def __init__(self, slide_src):
        self.slide = openslide.OpenSlide(slide_src)
        self.slide_name = os.path.basename(slide_src)

    def output_slide_info(self, level=4, magnification=None, logger=None):
        output_string = "\t[Dimensions]\t\t==> {}\n" \
                        "\t[Downsamples]\t\t==> {}".format(self.get_all_dimensions,
                                                           self.get_all_downsamples)
        if magnification is not None:
            self.best_level = self.slide.get_best_level_for_downsample(magnification)
            self.best_dimension = self.slide.level_dimensions[self.best_level]
            output_string += "[Best_level_of_Magnification[{}]\t==> {} - {}\n".format(magnification,
                                                                                      self.best_level,
                                                                                      self.best_dimension)

        if logger is not None:
            logger.info(output_string)
        else:
            print(output_string)

    @property
    def get_all_dimensions(self):
        return self.slide.level_dimensions

    @property
    def get_all_downsamples(self):
        return self.slide.level_downsamples

    def watch_thumbnail(self, figsize=(1024, 1024), save_dst=None, watch=False):
        thumbnail = self.slide.get_thumbnail(size=figsize)
        thumbnail = np.array(thumbnail)[:, :, :3]

        if save_dst is not None:
            cv2.imwrite(save_dst, thumbnail)
        if watch:
            plt.imshow(thumbnail)
            plt.axis('off')
            plt.show()


class SlideProcessor(SlideReader):
    """
        处理一个slide
        功能：
            1. tiling整个slide，不保存背景(将白色超过70%的tile看作为背景块)
            2. get_one_tile(): 获取一个tile，plot_tile=True时显示tile
            3. record*_df函数保存slide和tile的信息
    """

    def __init__(self, config, slide_src, all_slides_info_df, logger):
        super().__init__(slide_src)
        self.cfg = config
        self.logger = logger

        self.slide_id = self.slide_name.split('.')[0]
        self.case_id = self.slide_name[:12]
        self.slides_info_df = all_slides_info_df
        self.slide_tiles_df = pd.DataFrame([], columns=['tile_id', 'case_id', 'coordination', 'slide_id', 'tile_type', 'tile_path'])
        self.tiles_count_all = 0
        self.tiles_count_tissue = 0

    def otsu_tissue_mask(self, watch=False):
        thumbnail_dst = os.path.join(self.cfg.data_root, 'thumbnails', self.slide_name.split('.')[0])
        thumbnail = np.array(self.slide.get_thumbnail(self.slide.level_dimensions[-1]))[:, :, :3]
        otsu_threshold, otsu_mask, overlap_image = hsv_otsu_image(input_image=thumbnail, overlap=True)

        cv2.imwrite(thumbnail_dst + '_overall.png', thumbnail)
        cv2.imwrite(thumbnail_dst + '_otsu_mask.png', otsu_mask)
        cv2.imwrite(thumbnail_dst + '_otsu_image.png', overlap_image)

        if watch:
            plot_multi_subplot_one_row(images=[thumbnail, otsu_mask, overlap_image],
                                       names=['Overall', 'OTSU Mask', 'OTSU Image'],
                                       fig_title=os.path.basename(thumbnail))

        return otsu_threshold, otsu_mask, overlap_image


    def fit(self, magnification, tile_size):
        best_level = self.slide.get_best_level_for_downsample(magnification)
        best_dimension = self.slide.level_dimensions[best_level]
        # best_downsample = self.slide.level_downsamples[best_level]
        self.output_slide_info(level=best_level, logger=self.logger)
        message = "\t[Magnification]\t\t==>\t[{}x]\n" \
                  "\t[Best_level]\t\t==>\t{}\n" \
                  "\t[Best_Dimension]\t==>\t{}".format(magnification, best_level, best_dimension)
        message_output(input_string=message, input_logger=self.logger)
        otsu_threshold, _, _ = self.otsu_tissue_mask()

        slide_dst = os.path.join(self.cfg.folder_tiles_dst, self.slide_name.split('.')[0])
        FolderTool(input_path=slide_dst).doer()

        width_steps = best_dimension[0] // tile_size
        height_steps = best_dimension[1] // tile_size
        plt.ion()
        for w in tqdm(range(width_steps)):
            for h in range(height_steps):
                # get the tile
                start_point = (w * tile_size, h * tile_size)
                tile_shape = (tile_size, tile_size)
                tile_id = "{}_{}_[{}_{}]".format(self.slide_name.split('.')[0], best_level,
                                                 w * tile_size, h * tile_size)
                self.tiles_count_all += 1

                tile = self.get_one_tile(left_top=start_point, level=best_level, tile_size=tile_shape,
                                         plot_tile=self.cfg.plot_tile)
                tile_array = np.array(tile)[:, :, :3]

                # tell type of tile
                tile_type = 'tissue'
                if isBackground_otsu(input_image=tile_array, ratio_threshold=self.cfg.ratio_threshold,
                                     black_threshold=self.cfg.black_threshold, otsu_threshold=otsu_threshold):
                    tile_type = 'background'
                    cv2.imwrite("{}/{}.png".format(slide_dst+'/background', tile_id), tile_array)
                    self.record_tiles_info(tile_id=tile_id, coor=start_point, tile_type=tile_type)
                    continue

                # color norm ------------------------------------------------

                # quality assess ------------------------------------------------

                # save tile
                cv2.imwrite("{}/{}.png".format(slide_dst, tile_id), tile_array)

                # 存储信息
                self.record_tiles_info(tile_id=tile_id, coor=start_point, tile_type=tile_type)
                self.tiles_count_tissue += 1

        self.slide_tiles_df.to_csv(os.path.join(self.cfg.documents_csv_folder, '{}.csv'.format(self.slide_id)),
                                   index=False)
        self.record_slides_info()

        return self.slides_info_df

    def get_one_tile(self, left_top: tuple, level: int, tile_size, plot_tile=False) -> Image:
        """
        获取一个tile
        :param left_top: 截图的左上角坐标
        :param level: 病理图金字塔的level
        :param tile_size: 截取出来图片的大小
        :param plot_tile: 是否显示tile图片
        :return: 返回tile(PIL.Image)
        """
        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)
        elif isinstance(tile_size, tuple):
            tile_size = tile_size
        else:
            assert "[Error] Tile_size should be a [tuple or int], now is [{}]".format(type(tile_size))

        # 处理left_top为level=0的范围
        w_, h_ = left_top
        w = w_ * self.slide.level_downsamples[level]
        h = h_ * self.slide.level_downsamples[level]
        tile = self.slide.read_region(location=(int(w), int(h)), level=level, size=tile_size)
        if plot_tile:
            plt.clf()
            plt.imshow(tile)
            plt.axis('off')
            plt.pause(0.01)  # 暂停0.01秒
            plt.ioff()  # 关闭画图的窗口
        return tile

    def record_tiles_info(self, tile_id, coor, tile_type):
        """
            * coor: 左上角坐标
            * tile_type: "background, tissue" --- 后期加上'tumor'
        """
        self.slide_tiles_df.loc[tile_id, 'tile_id'] = tile_id
        self.slide_tiles_df.loc[tile_id, 'case_id'] = self.slide_name[:12]
        self.slide_tiles_df.loc[tile_id, 'coordination'] = coor
        self.slide_tiles_df.loc[tile_id, 'slide_id'] = self.slide_name
        self.slide_tiles_df.loc[tile_id, 'tile_type'] = tile_type

    def record_slides_info(self):
        if self.slide_id not in self.slides_info_df.index:
            self.slides_info_df.loc[self.slide_id, 'num_tiles'] = self.tiles_count_all
            self.slides_info_df.loc[self.slide_id, 'num_tiles_tissue'] = self.tiles_count_tissue
            self.slides_info_df.loc[self.slide_id, 'num_tiles_background'] = self.tiles_count_all - self.tiles_count_tissue

            self.slides_info_df.loc[self.slide_id, 'slide_type'] = self.slide_name[13:15]
            self.slides_info_df.loc[self.slide_id, 'section_type'] = self.slide_name[17:19]
            self.slides_info_df.loc[self.slide_id, 'tiles_dst'] = os.path.join(self.cfg.folder_tiles_dst, self.slide_name.split('.')[0])
        else:
            message_output(input_string="[Warning] '{}' existed ---------- ! ".format(self.slide_id),
                           input_logger=self.logger)


def fit_slides2tiles(input_config, input_logger=None, restart_totally=False):
    slides_root = os.path.join(input_config.data_root, 'slides')
    slides_paths = glob(os.path.join(slides_root, '*.svs'))
    message_output("\n"
                   "{} Generating Tiles[level[{}] - Tile_size{}] {}".format('-' * 50,
                                                                            input_config.magnification, input_config.tile_size,
                                                                            '-' * 50), input_logger=input_logger)
    all_slides_info_df = pd.read_csv(os.path.join(input_config.documents_root, "all_slides_info.csv"), index_col='slide_id')

    if input_config.debug:
        message_output("[Info] {:-^80}".format(' Debugging '), input_logger=input_logger)
        slides_paths = slides_paths[:3]

    for idx, slides_path in enumerate(slides_paths):
        message_output(input_string="\n"
                                    "[{}/{}] {:-^100}".format(idx+1, len(slides_paths),
                                                              os.path.basename(slides_path)), input_logger=input_logger)
        if os.listdir(input_config.folder_tiles_dst) and not restart_totally:
            message_output(input_string="[Warning] ---- {} ---- Exists! ".format(os.path.basename(slides_path)),
                           input_logger=input_logger)
            continue


        processor = SlideProcessor(slide_src=slides_path, config=input_config,
                                   all_slides_info_df=all_slides_info_df, logger=input_logger)
        all_slides_info_df = processor.fit(magnification=input_config.magnification,
                                           tile_size=input_config.tile_size)

    all_slides_info_df.to_csv(os.path.join(input_config.documents_root, "all_slides_info.csv"))

def generate_tiles_folder(input_config):
    """创建存储tiles的一系列文件夹"""
    tiles_root = os.path.join(input_config.data_root, 'tiles')
    thumbnail_root = os.path.join(input_config.data_root, 'thumbnails')
    target_folder = os.path.join(tiles_root, "{}_{}".format(input_config.magnification,
                                                            input_config.tile_size))
    target_folder_data = os.path.join(target_folder, 'data')
    documents_root = os.path.join(target_folder, 'documents')
    documents_csv_folder = os.path.join(documents_root, 'slides_tiles_csv')

    FolderTool([thumbnail_root, tiles_root,
                target_folder, target_folder_data,
                documents_root, documents_csv_folder]).doer()
    return target_folder, target_folder_data, documents_root, documents_csv_folder


class StitchTiles:
    """缝合tiles"""
    def __init__(self, config):
        # 根据不同的放大倍数进行缝合，直接缩放就好？
        ...

    def fit(self):
        ...

    def stitch(self):
        ...

    def output_heatmap(self):
        # set output config: figure title, fig axis...
        ...

    def overlap(self):
        ...


class StainNorm:
    def __init__(self, reference_img='./utils/stain_norm_ref.png', method="vahadane"):
        self.normalizer = staintools.StainNormalizer(method)
        ref_img = staintools.read_image(reference_img)
        self.normalizer.fit(ref_img)

    def fit(self, dst_img):
        if not isinstance(dst_img, np.ndarray):
            dst_img = cv2.cvtColor(cv2.imread(dst_img), cv2.COLOR_BGR2RGB)
        return self.normalizer.transform(dst_img)

def main():
    args_generator = Slide2TileConfig(config_name="Generate Tiles for Slide")
    args = args_generator.output_parser
    folder_dst, data_dst, documents_root, documents_csv_folder = generate_tiles_folder(args)
    args.folder_dst = folder_dst  # 当前tile_size和level的存储根路径
    args.folder_tiles_dst = data_dst
    args.documents_root = documents_root
    args.documents_csv_folder = documents_csv_folder

    tiles_generator_logger = construct_logger(folder_dst, log_name="tiles_generator", save_time=False)
    pd.DataFrame([], columns=['slide_id', 'slide_type', 'section_type',
                              'num_tiles', 'num_tiles_tissue', 'num_tiles_background',
                               'tiles_dst']).to_csv(os.path.join(args.documents_root, "all_slides_info.csv"),
                                                   index=False)

    # 打印args信息
    tiles_generator_logger.info("{:-^100}".format("Setting"))
    for k, v in vars(args).items():
        output_string = "{: >20} ==> {}".format(k, v)
        message_output(input_string=output_string, input_logger=tiles_generator_logger)

    fit_slides2tiles(input_config=args,
                     input_logger=tiles_generator_logger,
                     restart_totally=args.restart_totally)


if __name__ == '__main__':
    main()
