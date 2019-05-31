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

from metrics.basic import Metrics


class NLL(Metrics):
    def __init__(self, name, model, loader, gpu=True):
        super(NLL, self).__init__(name)

        self.model = model
        self.loader = loader
        self.gpu = gpu
        self.need_reset = True

    def get_score(self, model=None, loader=None, ignore=False):
        """note that NLL score need the updated model and data loader each time, use reset() before get_score()"""
        if ignore:
            return 0
        if model and loader:
            self.reset(model, loader)
        assert not self.need_reset, 'need reset model and loader before calculating NLL'
        self.need_reset = True
        return self.cal_nll()

    def reset(self, model, loader):
        self.model = model
        self.loader = loader
        self.need_reset = False

    def cal_nll(self):
        total_loss = 0
        criterion = nn.NLLLoss()
        with torch.no_grad():
            for i, data in enumerate(self.loader):
                inp, target = data['input'], data['target']
                if self.gpu:
                    inp, target = inp.cuda(), target.cuda()

                hidden = self.model.init_hidden(self.loader.batch_size)
                pred = self.model.forward(inp, hidden)
                loss = criterion(pred, target.view(-1))
                total_loss += loss.item()
        return total_loss / len(self.loader)
