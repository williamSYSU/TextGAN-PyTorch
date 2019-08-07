# -*- coding: utf-8 -*-
from multiprocessing import Pool

import nltk
import os
import random
from nltk.translate.bleu_score import SmoothingFunction

from metrics.basic import Metrics


class SelfBLEU(Metrics):
    def __init__(self, test_text=None, gram=3, portion=1):
        if type(gram) == int:
            super(SelfBLEU, self).__init__('SelfBLEU-%d' % gram)
        elif type(gram) == list:
            super(SelfBLEU, self).__init__('SelfBLEU-%s' % gram)
        else:
            raise AssertionError('Gram format error!')

        self.test_text = test_text
        self.gram = [gram] if type(gram) == int else gram
        self.sample_size = 200  # BLEU scores remain nearly unchanged for self.sample_size >= 200
        self.portion = portion  # how many portions to use in the evaluation, default to use the whole test dataset

    def get_score(self, is_fast=True, ignore=False, given_gram=None, fmt_str=True):
        if ignore:
            return 0
        self.get_reference()
        if is_fast:
            return self.get_bleu_fast(given_gram, fmt_str)
        return self.get_bleu(given_gram, fmt_str)

    def get_reference(self):
        reference = self.test_text.copy()

        # randomly choose a portion of test data
        # In-place shuffle
        random.shuffle(reference)
        len_ref = len(reference)
        reference = reference[:int(self.portion * len_ref)]
        return reference

    def get_bleu(self, given_gram=None, fmt_str=True):
        if given_gram is not None:  # for single gram
            bleu = list()
            reference = self.get_reference()
            weight = tuple((1. / given_gram for _ in range(given_gram)))
            for idx, hypothesis in enumerate(self.test_text[:self.sample_size]):
                bleu.append(self.cal_bleu(reference, hypothesis, weight))
            return round(sum(bleu) / len(bleu), 3)
        else:  # for multiple gram
            all_bleu = []
            for ngram in self.gram:
                bleu = list()
                reference = self.get_reference()
                weight = tuple((1. / ngram for _ in range(ngram)))
                for idx, hypothesis in enumerate(self.test_text[:self.sample_size]):
                    bleu.append(self.cal_bleu(reference, hypothesis, weight))
                all_bleu.append(round(sum(bleu) / len(bleu), 3))
            # if fmt_str:
            #     return ', '.join([str(s) for s in all_bleu])
            return all_bleu

    @staticmethod
    def cal_bleu(reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self, given_gram=None, fmt_str=True):
        reference = self.get_reference()
        if given_gram is not None:  # for single gram
            return self.get_bleu_parallel(ngram=given_gram, reference=reference)
        else:  # for multiple gram
            all_bleu = []
            for ngram in self.gram:
                all_bleu.append(self.get_bleu_parallel(ngram=ngram, reference=reference))
            # if fmt_str:
            #     return ', '.join([str(s) for s in all_bleu])
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
