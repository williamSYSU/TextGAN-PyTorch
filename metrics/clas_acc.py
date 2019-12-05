# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : clas_acc.py
# @Time         : Created at 2019/12/4
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch

from metrics.basic import Metrics


class ACC(Metrics):
    def __init__(self, if_use=True, gpu=True):
        super(ACC, self).__init__('clas_acc')

        self.if_use = if_use
        self.model = None
        self.data_loader = None
        self.gpu = gpu

    def get_score(self):
        if not self.if_use:
            return 0
        assert self.model and self.data_loader, 'Need to reset() before get_score()!'

        return self.cal_acc(self.model, self.data_loader)

    def reset(self, model=None, data_loader=None):
        self.model = model
        self.data_loader = data_loader

    def cal_acc(self, model, data_loader):
        total_acc = 0
        total_num = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data['input'], data['target']
                if self.gpu:
                    inp, target = inp.cuda(), target.cuda()

                pred = model.forward(inp)
                total_acc += torch.sum((pred.argmax(dim=-1) == target)).item()
                total_num += inp.size(0)
        return round(total_acc / total_num, 4)
