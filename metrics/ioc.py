from collections import Counter
from itertools import chain

import nltk
import os
import random

from metrics.basic import Metrics


class IOC(Metrics):
    def __init__(self, weight, name=None, test_text=None, real_text=None, if_use=True):
        super(IOC, self).__init__('Index of Coincidence', weight, if_use)

        self.if_use = if_use
        self.test_text = test_text
        self.real_text_ioc = self.calculate_ioc(real_text.tokens) if real_text else None
        if real_text_ioc:
            print(f'Dataset Index of coincidence: {self.real_text_ioc}')
        self.reference = None
        self.is_first = True

    def _reset(self, test_text=None, real_text=None):
        self.test_text = test_text if test_text else self.test_text
        self.real_text_ioc = self.get_ioc(real_text.tokens) if real_text else self.real_text_ioc

    def calculate_metric(self):
        return self.calculate_ioc(self.test_text) / self.real_text_ioc

    def calculate_ioc(self, tokenized_text):
        """Index Of coincidence: probability of 2 random tokens in text to equal."""
        tokens = list(chain(*tokenized_text))
        counts = Counter(tokens)
        total = sum(ni * (ni - 1) for ni in counts.values())
        N = len(tokens)
        return total / N / (N - 1)
