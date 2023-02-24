from collections import Counter
from itertools import chain
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

from metrics.basic import Metrics


class GPTNLL(Metrics):
    def __init__(self, weight, name=None, test_text=None, real_text=None, if_use=True):
        super(GPTNLL, self).__init__('GPT2 as oracle', weight, if_use)

        self.if_use = if_use
        self.test_text = test_text

        self.NLLloss = torch.nn.NLLLoss()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        print('Calculating dataset NLL')
        self.real_text_nll = self.calcualte_NLL(random.sample(real_text, 500)) if real_text else None
        print(f'dataset NLL based on GPT2 is {self.real_text_nll}')
        print('GPT2 as oracle metric will be calculated relative to this value')

    def reset(self, test_text=None, real_text=None):
        self._reset()
        self.test_text = test_text if test_text else self.test_text
        self.real_text_nll = self.calcualte_NLL(real_text) if real_text else self.real_text_nll

    def calculate_metric(self):
        """Get gpt2 NLL score difference with dataset NLL."""
        return self.calcualte_NLL(self.test_text) - self.real_text_nll

    def calcualte_NLL(self, messages):
        if type(messages[0]) == list: #we received list of tokens
            messages = [' '.join(msg) for msg in messages]

        all_logits = []
        for message in messages:
            message = self.tokenizer.eos_token + message + self.tokenizer.eos_token
            inputs = self.tokenizer(message, return_tensors="pt")
            logits = self.model(**inputs)[0][0]
            logits = F.log_softmax(logits, dim=1)
            # calculating NLL loss on token appearing on it's position
            all_logits.append(
                self.NLLloss(logits[:-1], inputs["input_ids"][0][1:]).detach().numpy()
            )
        return np.mean(all_logits)
