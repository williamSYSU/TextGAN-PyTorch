# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : metrics.py
# @Time         : Created at 2019-05-14
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import random
import nltk
from abc import abstractmethod
from nltk.translate.bleu_score import SmoothingFunction


class Metrics:
    def __init__(self, name='Metric'):
        self.name = name

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    @abstractmethod
    def get_score(self):
        pass


class BLEU(Metrics):
    def __init__(self, test_text=None, real_text=None, gram=3, portion=1):
        super(BLEU, self).__init__('BLEU')
        self.name = 'BLEU'
        self.test_text = test_text
        self.real_text = real_text
        self.gram = gram
        self.sample_size = 200  # BLEU scores remain nearly unchanged for self.sample_size >= 200
        self.reference = None
        self.is_first = True
        self.portion = portion  # how many portions to use in the evaluation, default to use the whole test dataset

    def get_name(self):
        return self.name

    def get_score(self, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        return self.get_bleu()

    def get_reference(self):
        if self.reference is None:
            reference = self.real_text.copy()

            # randomly choose a portion of test data
            # In-place shuffle
            random.shuffle(reference)
            len_ref = len(reference)
            reference = reference[:int(self.portion * len_ref)]
            self.reference = reference
            return reference
        else:
            return self.reference

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        idx = 0
        for hypothesis in self.test_text:
            if idx >= self.sample_size:
                return sum(bleu) / len(bleu)
            bleu.append(self.calc_bleu(reference, hypothesis, weight))
            idx += 1
        return sum(bleu) / len(bleu)

    @staticmethod
    def calc_bleu(reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)
