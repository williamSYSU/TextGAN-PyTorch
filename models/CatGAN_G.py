# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : CatGAN_G.py
# @Time         : Created at 2019-05-28
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from models.generator import LSTMGenerator
from models.relational_rnn_general import RelationalMemory
from utils.helpers import truncated_normal_


class CatGAN_G(LSTMGenerator):
    def __init__(self, k_label, mem_slots, num_heads, head_size, embedding_dim, hidden_dim, vocab_size, max_seq_len,
                 padding_idx,
                 gpu=False):
        super(CatGAN_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'catgan'

        self.k_label = k_label
        self.temperature = 1.0  # init value is 1.0

        # Category matrix
        # self.cat_mat = nn.Parameter(torch.rand(self.k_label, embedding_dim), requires_grad=True)
        self.cat_mat = nn.Parameter(torch.eye(k_label), requires_grad=False)

        # RMC
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.hidden_dim = mem_slots * num_heads * head_size
        self.lstm = RelationalMemory(mem_slots=mem_slots, head_size=head_size, input_size=k_label + embedding_dim,
                                     num_heads=num_heads, return_all_outputs=True)
        self.lstm2out = nn.Linear(self.hidden_dim, vocab_size)

        # LSTM
        # self.hidden_dim = 512
        # self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, batch_first=True)
        # self.lstm2out = nn.Linear(self.hidden_dim, vocab_size)

        self.init_params()

    def forward(self, inp, hidden, label=None, need_hidden=False):
        """
        Embeds input and applies LSTM, concatenate category vector into each embedding
        :param inp: batch_size * seq_len
        :param label: batch_size, specific label index
        :param hidden: memory size
        :param need_hidden: if return hidden, use for sampling
        """
        assert type(label) == torch.Tensor, 'missing label'
        emb = self.embeddings(inp)  # batch_size * len * embedding_dim

        # cat category vector
        label_onehot = F.one_hot(label, self.k_label).float()  # batch_size * k_label
        label_onehot_ex = label_onehot.unsqueeze(1).expand(-1, inp.size(1), -1)  # batch_size * len * k_label
        label_vec = torch.bmm(label_onehot_ex, self.cat_mat.expand(inp.size(0), -1, -1))  # batch_size * len * embed_dim
        emb = torch.cat((emb, label_vec), dim=-1)  # batch_sie * len * (2 * embed_dim)

        out, hidden = self.lstm(emb, hidden)  # out: batch_size * seq_len * hidden_dim
        out = out.contiguous().view(-1, self.hidden_dim)  # out: (batch_size * len) * hidden_dim
        out = self.lstm2out(out)  # batch_size * seq_len * vocab_size
        out = self.temperature * out  # temperature
        pred = self.softmax(out)

        if need_hidden:
            return pred, hidden
        else:
            return pred

    def step(self, inp, hidden, label=None):
        """
        RelGAN step forward
        :param inp: batch_size
        :param hidden: memory size
        :param label: batch_size, specific label index
        :return: pred, hidden, next_token
            - pred: batch_size * vocab_size, use for adversarial training backward
            - hidden: next hidden
            - next_token: [batch_size], next sentence token
        """
        assert type(label) == torch.Tensor, 'missing label'
        emb = self.embeddings(inp).unsqueeze(1)

        # cat category vector
        label_onehot = F.one_hot(label, self.k_label).float()  # batch_size * k_label
        label_onehot_ex = label_onehot.unsqueeze(1).expand(-1, 1, -1)  # batch_size * 1 * k_label
        label_vec = torch.bmm(label_onehot_ex, self.cat_mat.expand(inp.size(0), -1, -1))  # batch_size * 1 * embed_dim
        emb = torch.cat((emb, label_vec), dim=-1)  # batch_sie * len * (2 * embed_dim)

        out, hidden = self.lstm(emb, hidden)
        gumbel_t = self.add_gumbel(self.lstm2out(out.squeeze(1)))
        next_token = torch.argmax(gumbel_t, dim=1).detach()

        pred = F.softmax(gumbel_t * self.temperature, dim=-1)  # batch_size * vocab_size

        return pred, hidden, next_token

    def sample(self, num_samples, batch_size, one_hot=False, label_i=None,
               start_letter=cfg.start_letter):
        """
        Sample from RelGAN Generator
        - one_hot: if return pred of RelGAN, used for adversarial training
        - label_i: label index
        :return:
            - all_preds: batch_size * seq_len * vocab_size, only use for a batch
            - samples: all samples
        """
        assert type(label_i) == int, 'missing label'
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()
        if one_hot:
            all_preds = torch.zeros(batch_size, self.max_seq_len, self.vocab_size)
            if self.gpu:
                all_preds = all_preds.cuda()

        for b in range(num_batch):
            hidden = self.init_hidden(batch_size)
            inp = torch.LongTensor([start_letter] * batch_size)
            label_t = torch.LongTensor([label_i] * batch_size)
            if self.gpu:
                inp = inp.cuda()
                label_t = label_t.cuda()

            for i in range(self.max_seq_len):
                pred, hidden, next_token = self.step(inp, hidden, label_t)
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
        memory = self.lstm.repackage_hidden(memory)  # detch memory at first
        return memory.cuda() if self.gpu else memory

    @staticmethod
    def add_gumbel(o_t, eps=1e-10, gpu=cfg.CUDA):
        """Add o_t by a vector sampled from Gumbel(0,1)"""
        u = torch.rand(o_t.size())
        if gpu:
            u = u.cuda()
        g_t = -torch.log(-torch.log(u + eps) + eps)
        gumbel_t = o_t + g_t
        return gumbel_t

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if cfg.use_truncated_normal:
                    truncated_normal_(param, std=stddev)
                else:
                    torch.nn.init.normal_(param, std=stddev)
