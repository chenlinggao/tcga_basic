# !usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 下午4:45
# @Author   : ChenLingHao
# @File     : trainer.py
import os
import sys
import torch
import numpy as np
from time import time
from tqdm import tqdm

from core.mil_models import MILArchitecture
from core.tile_models import get_classifier, Tee
from utils.tools import calculate_hms
from utils.dl_tools import BasicTrainer, AverageMeter, ResultReport

sys.path.append('..')


class TileTrainer(BasicTrainer):
    def __init__(self, config, logger, tensorboard, model_result_root, fold=None):
        super().__init__(config, logger, tensorboard, model_result_root, fold)

    def get_model(self):
        model = get_classifier(self.cfg)
        return model

    def train_no_valid(self, train_loader, epoch, model_result_root):
        train_loss, train_metric = self._train(train_loader, epoch)

        epoch_string = "\n[Info] Epoch[{}/{}] - Loss[{:.6f}] - {}[{:.6f}]".format(epoch, self.cfg.epochs,
                                                                                  train_loss,
                                                                                  self.cfg.metric,
                                                                                  train_metric)
        self.logger.info(epoch_string)

        if self.cfg.warm_up_epochs > 0 and epoch < self.cfg.warm_up_epochs:
            scheduler = self.scheduler[0]
        else:
            scheduler = self.scheduler[1]
        scheduler.step()

        save_dict = dict(config=self.cfg,
                         model_dict=self.model.state_dict(),
                         stop_epoch=epoch,
                         optimizer=self.optimizer.state_dict(),
                         scheduler=scheduler.state_dict())

        save_dst = os.path.join(model_result_root, 'checkpoints', "checkpoint_final.pth")
        torch.save(save_dict, save_dst)

    def train_one_epoch(self, train_loader, valid_loader, epoch, model_result_root, early_stopper):
        train_loss, train_metric = self._train(train_loader, epoch)
        valid_loss, valid_metric = self._valid(valid_loader, epoch)

        epoch_string = "\n[Info] Epoch[{}/{}] - Loss[{:.6f}/{:.6f}] - {}[{:.6f}/{:.6f}]".format(epoch, self.cfg.epochs,
                                                                                                train_loss, valid_loss,
                                                                                                self.cfg.metric,
                                                                                                train_metric, valid_metric)
        self.logger.info(epoch_string)

        if self.cfg.warm_up_epochs > 0 and epoch < self.cfg.warm_up_epochs:
            scheduler = self.scheduler[0]
        else:
            scheduler = self.scheduler[1]
        scheduler.step()

        # sentence of early stop
        early_stopper(valid_loss)
        if early_stopper.save_flag:
            save_dict = dict(config=self.cfg,
                             model_dict=self.model.state_dict(),
                             stop_epoch=epoch,
                             optimizer=self.optimizer.state_dict(),
                             scheduler=scheduler.state_dict())

            save_dst = os.path.join(model_result_root, 'checkpoints', "checkpoint_{}.pth".format(self.k))
            torch.save(save_dict, save_dst)
        if early_stopper.early_stop:
            return True
        else:
            return False

    def _train(self, train_loader, epoch):
        start = time()
        interval_start = None
        losses = AverageMeter('Loss', ':.4e')
        metric = AverageMeter('{}'.format(self.cfg.metric), ':.5f')
        self.model.train()

        self.logger.info("\n[Info] Loading Data......")
        for idx, (images, labels) in enumerate(tqdm(train_loader)):
            if idx == 0:
                self.logger.info("\n[Info] Loaded Data and cost [{:.4f}min]\n".format((time() - start) / 60))
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)

            metric_ = self.outputs2label(outputs, labels)
            losses.update(loss.item(), self.cfg.batch_size)
            metric.update(metric_, self.cfg.batch_size)

            if self.cfg.print_interval > 0 and idx % (len(train_loader) // self.cfg.print_interval) == 0:
                if idx == 0:
                    interval_start = time()
                    self.logger.info("\r[Epoch-{}] batch_idx[{}/{}] - loss[{:.6f}] - {}[{:.6f}]".format(epoch, idx, len(train_loader),
                                                                                                        loss.item(),
                                                                                                        self.cfg.metric, metric_))
                else:
                    interval_end = time()
                    self.logger.info("\r[Epoch-{}] batch_idx[{}/{}] - loss[{:.6f}] - {}[{:.6f}] - Cost[{}]".format(epoch, idx, len(train_loader),
                                                                                                                   loss.item(),
                                                                                                                   self.cfg.metric, metric_,
                                                                                                                   calculate_hms(interval_start, interval_end)))
                    interval_start = interval_end
                self.tb.add_scalars('interval/train_loss', {'train': losses.avg}, self.print_counter)
                self.print_counter += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.tb.add_scalars("epoch/loss", {'train': losses.avg}, epoch)
        self.tb.add_scalars("epoch/{}".format(self.cfg.metric), {'train': metric.avg}, epoch)
        return losses.avg, metric.avg

    @torch.no_grad()
    def _valid(self, valid_loader, epoch):
        losses, preds, probs = [], [], []
        gts = []
        self.model.eval()
        for idx, (images, labels) in enumerate(tqdm(valid_loader)):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            probs_, preds_ = self.outputs2label(outputs, labels, get_probs=True)

            losses.append(loss.item())
            preds.append(preds_)
            probs.append(probs_)
            gts.append(labels.cpu().numpy())
        losses = np.array(losses)
        preds = np.concatenate(preds)
        probs = np.concatenate(probs)
        gts = np.concatenate(gts)
        results = ResultReport(pred_label=preds, ground_truth=gts,
                               pred_probabilities=probs).calculate_results()

        self.tb.add_scalars("epoch/loss", {'valid': np.mean(losses)}, epoch)
        self.tb.add_scalars("epoch/{}".format(self.cfg.metric), {'valid': results['AUC']}, epoch)
        return np.mean(losses), results['AUC']

    def outputs2label(self, outputs, labels, get_probs=False):
        pred_labels = torch.softmax(outputs, dim=1)
        pred_probs = pred_labels[:, -1].detach().cpu().numpy()
        pred_labels = (pred_labels.float() > 0.5)
        pred_labels = pred_labels[:, 1].cpu().numpy()
        pred_labels = np.int32(pred_labels)
        if get_probs:
            return pred_probs, pred_labels
        else:
            metric_ = ResultReport(pred_labels, labels,
                                   pred_probabilities=pred_probs).calculate_single_result(self.cfg.metric)
            return metric_


class MILTrainer(BasicTrainer):
    def __init__(self, config, logger, tensorboard, model_result_root, fold=None):
        super(MILTrainer, self).__init__(config, logger, tensorboard, model_result_root, fold)

    def get_model(self):
        model = MILArchitecture(config=self.cfg)
        return model

    def train_no_valid(self, train_loader, epoch, model_result_root):
        train_loss, train_metric = self._train(train_loader, epoch)

        epoch_string = "[Info] Epoch[{}/{}] - Loss[{:.6f}] - {}[{:.6f}]".format(epoch, self.cfg.epochs,
                                                                                train_loss,
                                                                                self.cfg.metric,
                                                                                train_metric)
        self.logger.info(epoch_string)

        if self.cfg.warm_up_epochs > 0 and epoch < self.cfg.warm_up_epochs:
            scheduler = self.scheduler[0]
        else:
            scheduler = self.scheduler[1]
        scheduler.step()

        save_dict = dict(config=self.cfg,
                         model_dict=self.model.state_dict(),
                         stop_epoch=epoch,
                         optimizer=self.optimizer.state_dict(),
                         scheduler=scheduler.state_dict())

        save_dst = os.path.join(model_result_root, 'checkpoints', "checkpoint_final.pth")
        torch.save(save_dict, save_dst)

    def train_one_epoch(self, train_loader, valid_loader, epoch, model_result_root, early_stopper):
        train_loss, train_metric = self._train(train_loader, epoch)
        valid_loss, valid_metric = self._valid(valid_loader, epoch)

        epoch_string = "[Info] Epoch[{}/{}] - Loss[{:.6f}/{:.6f}] - {}[{:.6f}/{:.6f}]\n".format(epoch, self.cfg.epochs,
                                                                                                train_loss, valid_loss,
                                                                                                self.cfg.metric,
                                                                                                train_metric, valid_metric)
        self.logger.info(epoch_string)

        if self.cfg.warm_up_epochs > 0 and epoch < self.cfg.warm_up_epochs:
            scheduler = self.scheduler[0]
        else:
            scheduler = self.scheduler[1]
        scheduler.step()

        # sentence of early stop
        early_stopper(valid_loss)
        if early_stopper.save_flag:
            save_dict = dict(config=self.cfg,
                             model_dict=self.model.state_dict(),
                             stop_epoch=epoch,
                             optimizer=self.optimizer.state_dict(),
                             scheduler=scheduler.state_dict())

            save_dst = os.path.join(model_result_root, 'checkpoints', "checkpoint_{}.pth".format(self.k))
            torch.save(save_dict, save_dst)
        if early_stopper.early_stop:
            return True
        else:
            return False

    def _train(self, loader, epoch):
        losses = AverageMeter("MIL_Train_Loss")
        metrics = AverageMeter("MIL_Train_{}".format(self.cfg.metric))
        probs, labels = [], []
        self.model.train()

        for features, label in tqdm(loader, desc="[Epoch-{}_train]".format(epoch)):
            labels.append(label.numpy())
            # 每个features是一个slide
            features = features.to(self.device)
            label = label.to(self.device)
            output, _ = self.model(features, return_attention=False)
            loss = self.loss_fn(output, label)
            losses.update(loss.item(), 1)

            output = output.softmax(dim=1).detach().cpu().numpy()
            prob = output[:, -1]
            probs.append(prob)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        probs = np.concatenate(probs, axis=0)
        preds = np.int32(probs > 0.5)
        labels = np.concatenate(labels)
        metric = ResultReport(ground_truth=labels,
                              pred_label=preds,
                              pred_probabilities=probs).calculate_single_result(self.cfg.metric)
        metrics.update(metric, 1)
        self.tb.add_scalars("epoch/loss", {'train': losses.avg}, epoch)
        self.tb.add_scalars("epoch/{}".format(self.cfg.metric), {'train': metrics.avg}, epoch)

        return losses.avg, metrics.avg

    def _valid(self, loader, epoch):
        losses = AverageMeter("MIL_Eval_Loss")
        metrics = AverageMeter("MIL_Eval_{}".format(self.cfg.metric))
        probs, labels = [], []
        self.model.eval()
        with torch.no_grad():
            for features, label in tqdm(loader, desc="[Epoch-{}_valid]".format(epoch)):
                labels.append(label)
                # 每个loader是一个slide
                features = features.to(self.device)
                label = label.to(self.device)
                output, _ = self.model(features, return_attention=False)
                loss = self.loss_fn(output, label)
                losses.update(loss.item(), 1)

                output = output.softmax(dim=1).detach().cpu().numpy()
                prob = output[:, -1]
                probs.append(prob)

        probs = np.concatenate(probs, axis=0)
        preds = np.int32(probs > 0.5)
        labels = np.concatenate(labels)
        metric = ResultReport(ground_truth=labels, pred_label=preds, pred_probabilities=probs).calculate_single_result(self.cfg.metric)
        metrics.update(metric, 1)
        self.tb.add_scalars("epoch/loss", {'valid': losses.avg}, epoch)
        self.tb.add_scalars("epoch/{}".format(self.cfg.metric), {'valid': metrics.avg}, epoch)

        return losses.avg, metrics.avg


def generate_trainer(config, logger, tensorboard, model_result_root, fold=None):
    if config.task == 'tile':
        return TileTrainer(config, logger, tensorboard, model_result_root, fold)
    elif config.task == 'mil':
        return MILTrainer(config, logger, tensorboard, model_result_root, fold)
