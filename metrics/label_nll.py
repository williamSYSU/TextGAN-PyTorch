# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : label_nll.py
# @Time         : Created at 2019-05-31
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn

from metrics.basic import Metrics


class NLL(Metrics):
    def __init__(self, name, model, loader, gpu=True):
        super(NLL, self).__init__(name)

        self.model = model
        self.loader = loader
        self.criterion = nn.NLLLoss()
        self.gpu = gpu
        self.need_reset = False

    def get_score(self, model=None, loader=None, label_i=None, ignore=False):
        """note that NLL score need the updated model and data loader each time, use reset() before get_score()"""
        if ignore:
            return 0
        if model and loader:
            self.reset(model, loader)
        assert type(label_i) == int, 'missing label'
        assert not self.need_reset, 'need reset model and loader before calculating NLL'
        self.need_reset = True
        return self.cal_nll(label_i)

    def reset(self, model, loader):
        self.model = model
        self.loader = loader
        self.need_reset = False

    def cal_nll(self, label_i):
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.loader):
                inp, target = data['input'], data['target']
                label = torch.LongTensor([label_i] * self.loader.batch_size)
                if self.gpu:
                    inp, target, label = inp.cuda(), target.cuda(), label.cuda()

                hidden = self.model.init_hidden(self.loader.batch_size)
                if self.model.name == 'oracle':
                    pred = self.model.forward(inp, hidden)
                else:
                    pred = self.model.forward(inp, hidden, label)
                loss = self.criterion(pred, target.view(-1))
                total_loss += loss.item()
        return total_loss / len(self.loader)
