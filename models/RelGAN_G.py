# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : RelGAN_G.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.generator import LSTMGenerator
from models.relational_rnn_general import RelationalMemory

import config as cfg


class RelGAN_G(LSTMGenerator):
    def __init__(self, mem_slots, num_heads, head_size, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx,
                 temperature, gpu=False):
        super(RelGAN_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'relgan'

        self.temperature = temperature
        # RMC
        self.hidden_dim = mem_slots * num_heads * head_size
        self.lstm = RelationalMemory(mem_slots=mem_slots, head_size=head_size, input_size=embedding_dim,
                                     num_heads=num_heads, return_all_outputs=True)
        self.lstm2out = nn.Linear(self.hidden_dim, vocab_size)

        self.init_params()

    def step(self, inp, hidden):
        """
        RelGAN step forward
        :param inp: [batch_size]
        :param hidden: memory size
        :param temperature: temperature control
        :return: pred, hidden, next_token, next_token_onehot, next_o
            - pred: batch_size * vocab_size, use for adversarial training backward
            - hidden: next hidden
            - next_token: [batch_size], next sentence token
            - next_token_onehot: batch_size * vocab_size, not used yet
            - next_o: batch_size * vocab_size, not used yet
        """
        emb = self.embeddings(inp).unsqueeze(1)
        out, hidden = self.lstm(emb, hidden)
        gumbel_t = self.add_gumbel(self.lstm2out(out.squeeze(1)))
        next_token = torch.argmax(gumbel_t, dim=1).detach()
        # next_token_onehot = F.one_hot(next_token, cfg.vocab_size).float()  # not used yet
        next_token_onehot = None

        pred = F.softmax(gumbel_t * self.temperature, dim=-1)  # batch_size * vocab_size
        # next_o = torch.sum(next_token_onehot * pred, dim=1)  # not used yet
        next_o = None

        return pred, hidden, next_token, next_token_onehot, next_o

    def sample(self, num_samples, batch_size, one_hot=False, start_letter=cfg.start_letter):
        """
        Sample from RelGAN Generator
        - one_hot: if return pred of RelGAN, used for adversarial training
        :return:
            - all_preds: batch_size * seq_len * vocab_size, only use for a batch
            - samples: all samples
        """
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()
        if one_hot:
            all_preds = torch.zeros(batch_size, self.max_seq_len, self.vocab_size)

        # 用LSTM随机生成batch_size个句子，生成下一个词的时候是按多项式分布来选择的，而不是概率最大那个
        for b in range(num_batch):
            hidden = self.init_hidden(batch_size)
            inp = torch.LongTensor([start_letter] * batch_size)
            if self.gpu:
                inp = inp.cuda()

            for i in range(self.max_seq_len):
                pred, hidden, next_token, _, _ = self.step(inp, hidden)
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token
                if one_hot:
                    all_preds[:, i] = pred
                inp = next_token
        samples = samples[:num_samples]  # num_samples * seq_len

        if one_hot:
            return all_preds  # batch_size * seq_len * vocab_size
        return samples

    def init_hidden(self, batch_size=cfg.batch_size):
        """init RMC memory"""
        memory = self.lstm.initial_state(batch_size)
        return memory.cuda() if self.gpu else memory

    @staticmethod
    def add_gumbel(o_t, eps=1e-10, gpu=cfg.CUDA):
        """Add o_t by a vector sampled from Gumbel(0,1)"""
        u = torch.rand(o_t.size())
        if gpu:
            u = u.cuda()
        g_t = -torch.log(-torch.log(u + eps) + eps)
        gumbel_t = torch.add(o_t, g_t)
        return gumbel_t
