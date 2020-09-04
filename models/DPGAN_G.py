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

from models.generator import LSTMGenerator


class DPGAN_G(LSTMGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(DPGAN_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'dpgan_g'

    def sample_teacher_forcing(self, inp):
        """
        Generating samples from the real data via teacher forcing
        :param inp: batch_size * seq_len
        :param target: batch_size * seq_len
        :return
            samples: batch_size * seq_len
            log_prob: batch_size * seq_len  (log probabilities)
        """
        batch_size, _ = inp.size()
        hidden = self.init_hidden(batch_size)

        pred = self.forward(inp, hidden)
        samples = torch.argmax(pred, dim=-1).view(batch_size, -1)
        log_prob = F.nll_loss(pred, samples.view(-1), reduction='none').view(batch_size, -1)
        # samples = torch.multinomial(torch.exp(log_prob), 1)

        return samples, log_prob
