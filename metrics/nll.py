# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : nll.py
# @Time         : Created at 2019-05-31
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn

import config as cfg
from metrics.basic import Metrics


class NLL(Metrics):
    def __init__(self, name, if_use=False, gpu=False):
        super(NLL, self).__init__(name)

        self.if_use = if_use
        self.model = None
        self.data_loader = None
        self.label_i = None
        self.leak_dis = None
        self.gpu = gpu
        self.criterion = nn.NLLLoss()

    def get_score(self):
        """note that NLL score need the updated model and data loader each time, use reset() before get_score()"""
        if not self.if_use:
            return 0
        assert self.model and self.data_loader, 'Need to reset() before get_score()!'

        if self.leak_dis is not None:  # For LeakGAN
            return self.cal_nll_with_leak_dis(self.model, self.data_loader, self.leak_dis, self.gpu)
        elif self.label_i is not None:  # For category text generation
            return self.cal_nll_with_label(self.model, self.data_loader, self.label_i,
                                           self.criterion, self.gpu)
        else:
            return self.cal_nll(self.model, self.data_loader, self.criterion, self.gpu)

    def reset(self, model=None, data_loader=None, label_i=None, leak_dis=None):
        self.model = model
        self.data_loader = data_loader
        self.label_i = label_i
        self.leak_dis = leak_dis

    @staticmethod
    def cal_nll(model, data_loader, criterion, gpu=cfg.CUDA):
        """NLL score for general text generation model."""
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data['input'], data['target']
                if gpu:
                    inp, target = inp.cuda(), target.cuda()

                hidden = model.init_hidden(data_loader.batch_size)
                pred = model.forward(inp, hidden)
                loss = criterion(pred, target.view(-1))
                total_loss += loss.item()
        return round(total_loss / len(data_loader), 4)

    @staticmethod
    def cal_nll_with_label(model, data_loader, label_i, criterion, gpu=cfg.CUDA):
        """NLL score for category text generation model."""
        assert type(label_i) == int, 'missing label'
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data['input'], data['target']
                label = torch.LongTensor([label_i] * data_loader.batch_size)
                if gpu:
                    inp, target, label = inp.cuda(), target.cuda(), label.cuda()

                hidden = model.init_hidden(data_loader.batch_size)
                if model.name == 'oracle':
                    pred = model.forward(inp, hidden)
                else:
                    pred = model.forward(inp, hidden, label)
                loss = criterion(pred, target.view(-1))
                total_loss += loss.item()
        return round(total_loss / len(data_loader), 4)

    @staticmethod
    def cal_nll_with_leak_dis(model, data_loader, leak_dis, gpu=cfg.CUDA):
        """NLL score for LeakGAN."""
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data['input'], data['target']
                if gpu:
                    inp, target = inp.cuda(), target.cuda()

                loss = model.batchNLLLoss(target, leak_dis)
                total_loss += loss.item()
        return round(total_loss / len(data_loader), 4)
