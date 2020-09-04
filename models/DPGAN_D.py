# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : SeqGAN_G.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn.functional as F

import config as cfg
from models.generator import LSTMGenerator
from utils.data_loader import GenDataIter


class DPGAN_D(LSTMGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(DPGAN_D, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'dpgan_d'

    def getReward(self, samples):
        """
        Get word-level reward and sentence-level reward of samples.
        """
        batch_size, _ = samples.size()
        inp, target = GenDataIter.prepare(samples, cfg.CUDA)

        hidden = self.init_hidden(batch_size)
        pred = self.forward(inp, hidden)

        word_reward = F.nll_loss(pred, target.view(-1), reduction='none').view(batch_size, -1)
        sentence_reward = torch.mean(word_reward, dim=-1, keepdim=True)

        return word_reward, sentence_reward
