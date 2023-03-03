# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : basic.py
# @Time         : Created at 2019-05-14
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.

from abc import abstractmethod


class Metrics:
    def __init__(self, name, weight, if_use):
        self.name = name
        # represents effect on final score
        # ex.: self-bleu has weight = -1 (less is better)
        # bleu has weight = 1 (more is better)
        # weights needed for combined metric evaluation
        self.weight = weight
        self.if_use = if_use
        self.metric_value_with_current_state = None

    def get_score(self):
        if not self.if_use:
            return 0

        if self.metric_value_with_current_state is not None:
            return self.metric_value_with_current_state

        self.metric_value_with_current_state = self.calculate_metric()
        return self.metric_value_with_current_state

    def reset(*args, **kwargs):
        self.metric_value_with_current_state = None
        self._reset(*args, **kwargs)

    @abstractmethod
    def calculate_metric(self):
        pass

    @abstractmethod
    def _reset(self):
        pass
