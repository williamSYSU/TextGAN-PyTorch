# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : bleu.py
# @Time         : Created at 2019-05-31
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.
from multiprocessing import Pool

import os
import random

import nltk
from nltk.translate.bleu_score import SmoothingFunction
from tqdm import tqdm

from metrics.basic import Metrics


class BLEU(Metrics):
    """
    Get BLEU scores.
    :param is_fast: Fast mode
    :param given_gram: Calculate specific n-gram BLEU score
    """
    def __init__(self, name=None, weight=1, test_text=None, real_text=None, gram=3, portion=1, if_use=False):
        assert type(gram) == int or type(gram) == list, 'Gram format error!'
        super(BLEU, self).__init__('%s-%s' % (name, gram), weight, if_use)

        self.if_use = if_use
        self.test_text = test_text
        self.real_text = real_text
        self.gram = gram if type(gram) == int else gram
        self.sample_size = 200  # BLEU scores remain nearly unchanged for self.sample_size >= 200
        self.portion = portion  # how many portions to use in the evaluation, default to use the whole test dataset

    def _reset(self, test_text=None, real_text=None):
        self.test_text = test_text if test_text is not None else self.test_text
        self.real_text = real_text if real_text is not None else self.real_text

    def get_reference(self):
        reference = self.real_text.copy()
        # randomly choose a portion of test data
        # In-place shuffle
        random.shuffle(reference)
        len_ref = len(reference)
        reference = reference[:int(self.portion * len_ref)]
        return reference

    def get_bleu(self, given_gram=None):
        if type(self.gram) == int: # for single gram
            return self.get_blue_for_single_gram(self.gram)
        # for multiple gram
        all_bleu = []
        for ngram in self.gram:
            all_bleu.append(self.get_blue_for_single_gram(ngram))
        return all_bleu

    def get_blue_for_single_gram(self, ngram):
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        for idx, hypothesis in enumerate(self.test_text[:self.sample_size]):
            bleu.append(self.cal_bleu(reference, hypothesis, weight))
        return round(sum(bleu) / len(bleu), 3)

    @staticmethod
    def cal_bleu(reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def calculate_metric(self):
        if type(self.gram) == int: # for single gram
            return self.get_blue_for_single_gram(self.gram)
        # for multiple gram
        reference = self.get_reference()
        all_bleu = []
        for ngram in self.gram:
            all_bleu.append(self.get_bleu_parallel(ngram=ngram, reference=reference))
        return all_bleu

    def get_bleu_parallel(self, ngram, reference):
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = list()
        for idx, hypothesis in enumerate(self.test_text[:self.sample_size]):
            result.append(pool.apply_async(self.cal_bleu, args=(reference, hypothesis, weight)))
        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return round(score / cnt, 3)
