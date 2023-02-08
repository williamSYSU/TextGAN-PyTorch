# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : JSDGAN_G.py
# @Time         : Created at 2019/11/17
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.


import torch
import torch.nn.functional as F

from models.generator import LSTMGenerator


class JSDGAN_G(LSTMGenerator):
    def __init__(self, mem_slots, num_heads, head_size, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx,
                 gpu=False):
        super(JSDGAN_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'jsdgan'

        # RMC

    #     self.hidden_dim = mem_slots * num_heads * head_size
    #     self.lstm = RelationalMemory(mem_slots=mem_slots, head_size=head_size, input_size=embedding_dim,
    #                                  num_heads=num_heads, return_all_outputs=True)
    #     self.lstm2out = nn.Linear(self.hidden_dim, vocab_size)
    #
    # def init_hidden(self, batch_size=cfg.batch_size):
    #     """init RMC memory"""
    #     memory = self.lstm.initial_state(batch_size)
    #     memory = self.lstm.repackage_hidden(memory)  # detch memory at first
    #     return memory.cuda() if self.gpu else memory

    def JSD_loss(self, inp, target):
        """
        Returns a JSDGAN loss

        :param inp: batch_size x seq_len, inp should be target with <s> (start letter) prepended
        :param target: batch_size x seq_len
        :return loss: loss to optimize
        """
        batch_size, seq_len = inp.size()
        hidden = self.init_hidden(batch_size)
        pred = self.forward(inp, hidden).view(batch_size, self.max_seq_len, self.vocab_size)
        target_onehot = F.one_hot(target, self.vocab_size).float()  # batch_size * seq_len * vocab_size
        pred = torch.sum(pred * target_onehot, dim=-1)  # batch_size * seq_len

        # calculate probabilities of sentences
        prob_gen = torch.exp(torch.sum(pred, dim=-1).double())  # sum of log prob
        prob_gen = self.min_max_normal(prob_gen).clamp(min=1e-10)
        prob_data = torch.DoubleTensor([1 / batch_size] * prob_gen.size(0))
        if self.gpu:
            prob_data = prob_data.cuda()

        # calculate the reward
        reward = torch.log(1. - torch.div(prob_data, prob_data + prob_gen))  # batch_size

        # check if nan
        if torch.isnan(reward).sum() > 0:
            print('Reward is nan!!!')
            exit(1)

        loss = torch.sum((prob_gen * reward).detach() * torch.sum(pred.double(), dim=-1))

        return loss

    def min_max_normal(self, prob):
        return torch.div(prob - torch.min(prob), torch.clamp(torch.max(prob) - torch.min(prob), min=1e-78))

    def sigmoid_normal(self, prob):
        """push prob either close to 0 or 1"""
        return torch.sigmoid((prob - 0.5) * 20)
