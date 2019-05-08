# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : SeqGAN_G.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import config as cfg
from models.generator import LSTMGenerator
from models.relational_rnn_general import RelationalMemory


class SeqGAN_G(LSTMGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(SeqGAN_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'seqgan'

        self.temperature = 1

        # RMC
        # mem_slots = 1
        # num_heads = 2
        # head_size = 256
        # self.hidden_dim = mem_slots * num_heads * head_size
        # self.lstm = RelationalMemory(mem_slots=mem_slots, head_size=head_size, input_size=embedding_dim,
        #                              num_heads=num_heads, return_all_outputs=True)
        # self.lstm2out = nn.Linear(self.hidden_dim, vocab_size)
        #
        # self.init_params()

    def batchPGLoss(self, inp, target, reward):
        """
        Returns a policy gradient loss

        :param inp: batch_size x seq_len, inp should be target with <s> (start letter) prepended
        :param target: batch_size x seq_len
        :param reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding sentence)
        :return loss: policy loss
        """

        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)  # seq_len x batch_size
        target = target.permute(1, 0)  # seq_len x batch_size
        hidden = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, hidden = self.forward(inp[i], hidden, need_hidden=True)  # origin: F.log_softmax, no_log=False

            # TODO: should hidden be detached from graph (.detach())?
            for j in range(batch_size):
                if cfg.if_reward:
                    loss += -out[j][target.data[i][j]] * reward[j]  # origin: log(P(y_t|Y_1:Y_{t-1})) * Q
                else:
                    loss += out[j][target.data[i][j]] * (1 - reward[j])  # P(y_t|Y_1:Y_{t-1}) * (1 - Q)
                # print('reward: ', reward[j].item(), 1 - reward[j].item())

        return loss / (seq_len * batch_size)

    # def init_hidden(self, batch_size=cfg.batch_size):
    #     """init RMC memory"""
    #     memory = self.lstm.initial_state(batch_size)
    #     return memory.cuda() if self.gpu else memory
