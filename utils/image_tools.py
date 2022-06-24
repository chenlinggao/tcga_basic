# !usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 下午4:21
# @Author   : ChenLingHao
# @File     : image_tools.py
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import timm
import torch
from torchvision.transforms import transforms as F

"""
    plot_image_hist(input_image): 输出图片、直方图
    hsv_otsu_image(input_image, overlap=False): 
"""


def plot_image_hist(input_image):
    """画图片的灰度直方图"""
    plt.figure(figsize=(15, 4))  # 设置画布的大小
    ax0 = plt.subplot(141)
    ax0.imshow(input_image)
    ax0.set_title("Max_pixel [{}]\nMin_pixel [{}]".format(input_image.max(), input_image.min()))
    ax0.axis('off')
    # B通道 直方图
    ax1 = plt.subplot(142)
    ax1.hist(input_image[:, :, 0].ravel(), bins=50, color='b')
    ax1.set_title("B channel")
    # G通道 直方图
    ax2 = plt.subplot(143)
    ax2.hist(input_image[:, :, 1].ravel(), bins=50, color='g')
    ax2.set_title("G channel")
    # R通道 直方图
    ax3 = plt.subplot(144)
    ax3.hist(input_image[:, :, 2].ravel(), bins=50, color='row')
    ax3.set_title("R channel")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------- 待整理 -----------------------------------------------------
# 整理成类

def plot_multi_row_col(images, names=None, save_dst=None, figsize=(20, 8), fig_title=None):
    if not isinstance(images, np.ndarray):
        images = np.array(images)

    num_row = images.shape[0]
    num_col = images.shape[1]
    if names is None:
        names = [None] * num_col

    plt.figure(figsize=figsize)
    plt.title(label=fig_title)
    for r in range(num_row):
        for c in range(num_col):
            if r == 0:
                ax = plt.subplot(num_row, num_col, r*num_col+(c+1))
                ax.imshow(images[r, c])
                ax.set_title(names[c])
                ax.axis('off')
            else:
                ax = plt.subplot(num_row, num_col, r*num_col+(c+1))
                ax.imshow(images[r, c])
                ax.axis('off')
    if save_dst is not None:
        plt.savefig(os.path.join(save_dst, fig_title + '.png'))
    plt.tight_layout()
    plt.show()


def plot_multi_subplot_one_row(images, row=1, names=None, save_dst=None, figsize=(20, 8), fig_title=None):
    num_col = len(images)
    if names is None:
        names = [None] * num_col

    plt.figure(figsize=figsize)
    plt.title(label=fig_title)

    for idx, (image, name) in enumerate(zip(images, names)):
        ax = plt.subplot(row, num_col, idx + 1)
        ax.imshow(image)
        ax.set_title(name)
        ax.axis('off')

    if save_dst is not None:
        plt.savefig(os.path.join(save_dst, fig_title + '.png'))

    plt.show()


def binary_image(input_image, low_boundary=0, upper_boundary=255, overlap=False):
    """hsv->binary->overlap"""
    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    _, mask = cv2.threshold(hsv_image[:, :, 1], low_boundary, upper_boundary, cv2.THRESH_BINARY)

    overlap_image = None
    if overlap:
        overlap_image = cv2.add(hsv_image, np.zeros_like(hsv_image), mask=mask)

    return mask, overlap_image


def hsv_otsu_image(input_image, overlap=False):
    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    otsu_threshold, otsu_mask = cv2.threshold(hsv_image[:, :, 1], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    overlap_image = None
    if overlap:
        overlap_image = cv2.add(hsv_image, np.zeros_like(hsv_image), mask=otsu_mask)

    return otsu_threshold, otsu_mask, overlap_image


# ----------------------------------------------------- 待整理 -----------------------------------------------------
class PreprocessMask:
    def __init__(self, input_image, kernel_name='rect', overlap=False, watch=False):
        self.img = input_image
        self.kernel_name = kernel_name

    def _set_kernel(self, shape=(5, 5)):
        """
        kernel_name = ['rect', 'cross', 'ellipse']
        :param shape: kernel_shape
        :return: kernel
        """
        if self.kernel_name == 'rect':
            kernel = np.ones(shape=shape, dtype=np.uint8)
        else:
            raise NotImplementedError("No Kernel [{}]".format(self.kernel_name))
        return kernel

    def operate_open(self, input_image):
        kernel = self._set_kernel()
        opened_img = cv2.morphologyEx(input_image, cv2.MORPH_OPEN, kernel)
        return opened_img

    def operate_close(self, input_image):
        kernel = self._set_kernel()
        closed_img = cv2.morphologyEx(input_image, cv2.MORPH_CLOSE, kernel)
        return closed_img


# ----------------------------------------------------- 提取特征 -----------------------------------------------------
basic_transforms = F.Compose([F.ToPILImage(), F.ToTensor(),
                              F.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])])
class FeaturesGenerator:
    def __init__(self, model_name, last_feature,
                 use_cuda, batch_size, out_indices=None, **kwargs):
        # self.model =
        ...

    def fit(self, img):
        ...

    def get_model(self, model_name, pretrained=True,
                  last_feature=True, features_only=True, out_indices=None):
        if last_feature and out_indices is None:
            model = timm.create_model(model_name, pretrained, features_only=features_only)
        elif out_indices is not None:
            model = timm.create_model(model_name, pretrained, out_indices=out_indices)
        else:
            raise IOError

    def get_loader(self):
        ...


