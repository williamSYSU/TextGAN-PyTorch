from collections import Counter
from itertools import chain

import nltk
import os
import random

from metrics.basic import Metrics


class IOC(Metrics):
    def __init__(self, name=None, test_text=None, real_text=None, if_use=True):
        super(IOC, self).__init__('Index of Coincedense')

        self.if_use = if_use
        self.test_text = test_text
        self.real_text_ioc = self.get_ioc(real_text) if real_text else None
        self.reference = None
        self.is_first = True
        self.portion = 0.01#portion  # how many portions to use in the evaluation, default to use the whole test dataset

    def get_score(self):
        """Get IOC score."""
        if not self.if_use:
            return 0
        return self.get_ioc(self.test_text) / self.real_text_ioc

    def reset(self, test_text=None, real_text=None):
        self.test_text = test_text if test_text else self.test_text
        self.real_text_ioc = self.get_ioc(real_text) if real_text else self.real_text_ioc

    def get_ioc(self, list_tokens):
        """Index Of Coincedense: probability of 2 random tokens in text to equal."""
        tokens = list(chain(*list_tokens))
        counts = Counter(tokens)
        total = sum(ni * (ni - 1) for ni in counts.values())
        N = len(tokens)
        return total / N / (N - 1)
