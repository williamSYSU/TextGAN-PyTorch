# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : BarGAN_G.py
# @Time         : Created at 2019-06-30
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


class BarGAN_G(LSTMGenerator):
    def __init__(self, mem_slots, num_heads, head_size, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx,
                 gpu=False):
        super(BarGAN_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'seqgan'

        self.temperature = 1.0

        # RMC
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.hidden_dim = mem_slots * num_heads * head_size
        self.lstm = RelationalMemory(mem_slots=mem_slots, head_size=head_size, input_size=embedding_dim,
                                     num_heads=num_heads, return_all_outputs=True)
        self.lstm2out = nn.Linear(self.hidden_dim, vocab_size)

        # LSTM
        # self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # self.lstm2out = nn.Linear(hidden_dim, vocab_size)

        self.init_params()

    def step(self, inp, hidden, rebar):
        emb = self.embeddings(inp).unsqueeze(1)
        out, hidden = self.lstm(emb, hidden)
        out = self.lstm2out(out.squeeze(1))

        pred = F.softmax(out, dim=-1)
        gumbel_u = F.softmax(self.temperature * self.add_gumbel_u(out), dim=-1)
        # gumbel_u = F.softmax(self.temperature * self.add_gumbel_u(torch.log(pred)), dim=-1)
        next_token = torch.argmax(gumbel_u, dim=-1).detach()  # use gumbel_u !!!

        if rebar:
            next_token_onehot = F.one_hot(next_token, self.vocab_size).float()
            gumbel_v = F.softmax(self.temperature * self.add_gumbel_v(out, next_token), dim=-1)
            # gumbel_v = F.softmax(self.temperature * self.add_gumbel_v(torch.log(pred), next_token), dim=-1)
            hardlogQ = self.log_likelihood(next_token_onehot, out)
            return out, hidden, pred, next_token, next_token_onehot, gumbel_u, gumbel_v, hardlogQ
        else:
            return out, hidden, next_token

    def sample(self, num_samples, batch_size, rebar=False, start_letter=cfg.start_letter):
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()

        # REBAR
        all_pred = []
        all_gumbel_u = []
        all_gumbel_v = []
        all_hardlogQ = []

        for b in range(num_batch):
            hidden = self.init_hidden(batch_size)
            inp = torch.LongTensor([start_letter] * batch_size)
            if self.gpu:
                inp = inp.cuda()

            for i in range(self.max_seq_len):
                if rebar:
                    out, hidden, pred, next_token, next_token_onehot, gumbel_u, gumbel_v, hardlogQ = self.step(inp,
                                                                                                               hidden,
                                                                                                               rebar)
                    all_pred.append(pred)
                    all_gumbel_u.append(gumbel_u)
                    all_gumbel_v.append(gumbel_v)
                    all_hardlogQ.append(hardlogQ)
                else:
                    out, hidden, next_token = self.step(inp, hidden, rebar)
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token
                inp = next_token

        samples = samples[:num_samples]  # num_samples * seq_len

        if rebar:
            samples_onehot = F.one_hot(samples, self.vocab_size).float()
            all_pred = torch.stack(all_pred, dim=1)
            all_gumbel_u = torch.stack(all_gumbel_u, dim=1)
            all_gumbel_v = torch.stack(all_gumbel_v, dim=1)
            all_hardlogQ = torch.stack(all_hardlogQ, dim=1)
            # all_pred = torch.stack(all_pred, dim=1).view(batch_size * num_batch, -1, self.vocab_size)[:num_samples]
            # all_gumbel_u = torch.stack(all_gumbel_u, dim=1).view(batch_size * num_batch, -1,
            #                                                      self.vocab_size)[:num_samples]
            # all_gumbel_v = torch.stack(all_gumbel_v, dim=1).view(batch_size * num_batch, -1,
            #                                                      self.vocab_size)[:num_samples]
            # all_hardlogQ = torch.stack(all_hardlogQ, dim=1).view(batch_size * num_batch, -1)[:num_samples]
            return samples, samples_onehot, all_pred, all_gumbel_u, all_gumbel_v, all_hardlogQ
        else:
            return samples

    @staticmethod
    def add_gumbel_u(out, eps=1e-10, gpu=cfg.CUDA):
        gumbel_u = torch.zeros(out.size())
        if gpu:
            gumbel_u = gumbel_u.cuda()
        gumbel_u.uniform_(0, 1)
        g_t = -torch.log(-torch.log(gumbel_u + eps) + eps)
        out = out + g_t
        return out

    @staticmethod
    def add_gumbel_v(out, next_token, eps=1e-10, gpu=cfg.CUDA):
        gumbel_v = torch.zeros(out.size())
        if gpu:
            gumbel_v = gumbel_v.cuda()
        gumbel_v.uniform_(0, 1)
        p = torch.exp(out)
        b = gumbel_v[torch.arange(gumbel_v.size(0)), next_token].unsqueeze(1)
        v_1 = -torch.log(-(torch.log(gumbel_v + eps) / (p + eps)) - torch.log(b))
        v_2 = -torch.log(-torch.log(gumbel_v[torch.arange(gumbel_v.size(0)), next_token] + eps) + eps)

        v_1_clo = v_1.clone()
        v_1_clo[torch.arange(gumbel_v.size(0)), next_token] = v_2.clone()
        out = v_1_clo

        return out

    @staticmethod
    def log_likelihood(y, log_y_hat, eps=1e-10):
        """Computes log likelihood.

        Args:
          y: observed data
          log_y_hat: parameters of the variables

        Returns:
          log_likelihood
        """
        return torch.sum(y * torch.log(torch.clamp(F.softmax(log_y_hat, dim=-1), eps, 1)), 1)

    @staticmethod
    def _u_to_v_poly(hidden, u, samples, eps=1e-10):
        """Convert u to tied randomness in v."""
        p = torch.exp(hidden)

        v_k = torch.pow(u, 1 / torch.clamp(p, min=eps))
        v_k = v_k.detach()
        v_k = torch.pow(v_k, p)

        v_true_k = v_k[torch.arange(v_k.size(0)), samples]

        v_i = u / torch.clamp(torch.pow(v_true_k.unsqueeze(1), p), min=eps)
        v_i = v_i.detach()
        v_i = v_i * torch.pow(v_true_k.unsqueeze(1), p)

        v_clo = v_i.clone()
        v_clo[torch.arange(v_clo.size(0)), samples] = v_true_k.clone()

        v = v_clo + (-v_clo + u).detach()
        return v

    def init_hidden(self, batch_size=cfg.batch_size):
        """init RMC memory"""
        memory = self.lstm.initial_state(batch_size)
        memory = self.lstm.repackage_hidden(memory)  # detch memory at first
        return memory.cuda() if self.gpu else memory

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if cfg.use_truncated_normal:
                    truncated_normal_(param, std=stddev)
                else:
                    torch.nn.init.normal_(param, std=stddev)
