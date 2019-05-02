# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : SeqGAN-william
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


class RelGAN_G(LSTMGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(RelGAN_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'relgan'

        # RMC
        mem_slots = 1
        num_heads = 2
        head_size = 256
        self.hidden_dim = mem_slots * num_heads * head_size
        self.lstm = RelationalMemory(mem_slots=mem_slots, head_size=head_size, input_size=embedding_dim,
                                     num_heads=num_heads, return_all_outputs=True)

        self.lstm2out = nn.Linear(self.hidden_dim, vocab_size)

        self.init_params()

    def init_hidden(self, batch_size=1):
        memory = self.lstm.initial_state(batch_size)

        if self.gpu:
            return memory.cuda()
        else:
            return memory
