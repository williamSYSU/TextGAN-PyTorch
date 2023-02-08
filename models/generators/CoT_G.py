# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : CoT_G.py
# @Time         : Created at 2020/4/20
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.


import torch

from models.generator import LSTMGenerator


class CoT_G(LSTMGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(CoT_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'cot'

    def get_loss(self, input, rewards):
        """
        Calculate generator loss
        @param input: samples with start token, batch size * seq_len
        @param rewards: rewards form mediator, (batch size * seq_len) * vocab_size
        @return:
        """
        log_pred = self.forward(input, self.init_hidden(input.size(0)))  # (batch_size * seq_len) * vocab_size
        g_pred = torch.exp(log_pred)
        loss = -torch.sum(g_pred * (rewards - log_pred)) / rewards.size(0)
        return loss
