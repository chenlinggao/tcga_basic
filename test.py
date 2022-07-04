# !usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2022/6/20
# @Author   : ChenLingHao
# @File     : test.py
import os
import torch
import argparse
import pandas as pd
import numpy as np
from glob import glob

import cv2
import matplotlib.pyplot as plt

from core.dataset import output_test_loader
from core.tester import Tester
from core.tile_models import get_classifier
from utils.dl_tools import ResultReport
from utils.tools import FolderTool, construct_logger, message_output


def test_config():
    parser = argparse.ArgumentParser("Test Config")
    parser.add_argument("--magnification", default=1, type=int, help="[]")
    parser.add_argument("--tile_size", default=1024, type=int, help="[]")
    parser.add_argument("--trained_model_name", default="tile_resnet18_tmb_label_128_0.0003", help="[]")
    parser.add_argument("--target_label_name", default="tmb_label", help="[]")
    parser.add_argument("--task", default="tile", help="[]")
    # parser.add_argument("--", default="", type=, help="[]")
    return parser.parse_args()

def construct_test_folder(config):
    if config.target_label_name.split('_')[-1] == "label":
        config.task_type = "classification"
    elif config.target_label_name.split('_')[-1] == "score":
        config.task_type = "regression"

    project_root = os.path.abspath("..")
    data_root = os.path.join(project_root, "data/tumor", "tiles",
                             "{}_{}".format(config.magnification, config.tile_size))    # ----------- 后期要改中间的tumor部分
    result_root = os.path.join(project_root, "results")
    trained_model_root = os.path.join(result_root, "trained_models", config.trained_model_name)

    test_result_root = os.path.join(trained_model_root, "test_result")
    test_result_figure = os.path.join(test_result_root, "figure")
    test_result_tiles = os.path.join(test_result_root, "test_tiles_results")
    FolderTool([test_result_root, test_result_figure, test_result_tiles]).doer()

    config.data_root = data_root
    config.result_root = result_root
    config.trained_model_root = trained_model_root
    config.test_result_root = test_result_root
    return config


"""对应csv生成还没完成"""
class TestFullFlow:
    def __init__(self, config, logger=None):
        self.cfg = config
        df = pd.read_csv(os.path.join(config.data_root, 'documents',
                                      'fused_slides_gene_info.csv'))
        self.test_df = df[df.phase == 'test'].reset_index(drop=True)

        model_slides_result = os.path.join(config.trained_model_root, "test_results.csv")
        if not os.path.exists(model_slides_result):
            self.model_slides_result = pd.DataFrame([])
        else:
            self.model_slides_result = pd.read_csv(model_slides_result)
        self.logger = logger

    def fit(self):
        labels = []
        probs, preds = [], []
        model = self.output_model()
        for idx, row in self.test_df.iterrows():
            slide_id = row['slide_id']
            message_output("[{}/{}]: Processing {} ".format(idx, len(self.test_df), slide_id), input_logger=self.logger)
            label = row[self.cfg.target_label_name]
            labels.append(label)

            tile_probs, slide_prob, slide_label = self.test_one_slide(model, slide_id)
            probs.append(slide_prob)
            preds.append(slide_label)

            self.save_slide_result(slide_id, tiles_probabilities=tile_probs, slide_gt_label=label)  # 保存tile的预测信息

            # 保存slide的预测信息
            self.model_slides_result.loc[idx, 'slide_id'] = slide_id
            self.model_slides_result.loc[idx, 'prob'] = slide_prob
            self.model_slides_result.loc[idx, 'pred'] = slide_label
            self.model_slides_result.loc[idx, 'ground_truth'] = label

        # 保存所有的模型信息
        result_dict = ResultReport(ground_truth=labels, pred_probabilities=probs, pred_label=preds).calculate_results()
        result_dict.update({"model": self.cfg.trained_model_name})

        total_result_csv = os.path.join(self.cfg.test_results_root, "models_results.csv")
        if os.path.exists(total_result_csv):
            total_result_df = pd.DataFrame([])
            length_df = 0
        else:
            total_result_df = pd.read_csv(total_result_csv)
            length_df = len(total_result_df)

        total_result_df.loc[length_df, 'model'] = self.cfg.trained_model_name
        for k, v in result_dict.items():
            total_result_df.loc[length_df, k] = v
        total_result_df.to_csv(total_result_csv, index=False)

    def test_one_slide(self, trained_model, target_id):
        test_loader = output_test_loader(config=self.cfg, slide_id=target_id)
        predictor = Tester(config=self.cfg)
        tile_probs = predictor.fit(trained_model, test_loader)

        if self.cfg.task == 'tile':
            slide_prob, slide_label = self.tiles_probs_to_label(tile_probs)
            return tile_probs, slide_prob, slide_label

        else:
            raise NotImplementedError

    def save_slide_result(self, slide_id, tiles_probabilities, slide_gt_label):
        # 保存tile的指标信息
        labels = np.full_like(tiles_probabilities, slide_gt_label)
        tile_preds = np.int32(tiles_probabilities > 0.5)
        tiles_result = ResultReport(ground_truth=labels,
                                    pred_probabilities=tiles_probabilities,
                                    pred_label=tile_preds).calculate_results()
        result_df = pd.DataFrame(tiles_result)
        result_df.to_csv(os.path.join(self.cfg.test_result_tiles, slide_id+".csv"), index=False)

    @staticmethod
    def tiles_probs_to_label(probabilities):
        preds = np.mean(probabilities>0.5)
        if preds > 0.5:
            return preds, 1
        else:
            return preds, 0

    def output_model(self):
        if not self.cfg.use_cv:
            trained_info = torch.load(os.path.join(self.cfg.trained_model_root, 'checkpoints', 'checkpoint_0.pth'))
            if self.cfg.task == 'tile':
                model = get_classifier(self.cfg)
                model.load_state_dict(trained_info['model_dict'])

            # elif self.cfg.task == 'mil':
            #     ...
            else:
                raise NotImplementedError
        else:
            # 不是交叉验证
            raise NotImplementedError
        return model


def main(config):
    """
        1. prepare (folder, ckpt_src, test_result_dst)
        2. predict
        3. output metrics of prediction
        4. output heatmap
    """
    test_logger = construct_logger(config.test_result_root, log_name="test")
    message_output("{:-^}\n".format(" Start Testing "))
    tester = TestFullFlow(config, logger=test_logger)
    tester.fit()


if __name__ == '__main__':
    args = test_config()
    args = construct_test_folder(args)
    main(args)


