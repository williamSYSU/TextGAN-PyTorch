import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init

import config as cfg


class Generator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, oracle_init=False):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.gru = nn.LSTM(embedding_dim, hidden_dim)
        self.gru2out = nn.Linear(hidden_dim, vocab_size)

        # initialise oracle network with N(0,1)
        # otherwise variance of initialisation is very small => high NLL for data sampled from the same model
        if oracle_init:
            for p in self.parameters():
                init.normal_(p, 0, 1)

    def init_hidden(self, batch_size=1):
        # h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))
        h = torch.zeros(1, batch_size, self.hidden_dim)
        c = torch.zeros(1, batch_size, self.hidden_dim)

        if self.gpu:
            return (h.cuda(), c.cuda())
        else:
            return (h, c)

    def forward(self, inp, hidden, no_log=False):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)
        """

        # input dim                                             # batch_size
        emb = self.embeddings(inp)  # batch_size x embedding_dim
        emb = emb.view(1, -1, self.embedding_dim)  # 1 x batch_size x embedding_dim
        # print('size of hidden:', hidden.size())
        out, hidden = self.gru(emb, hidden)  # 1 x batch_size x hidden_dim (out)
        out = self.gru2out(out.view(-1, self.hidden_dim))  # batch_size x vocab_size
        if no_log:
            out = F.softmax(out, dim=1)  # SentiGAN: softmax
        else:
            out = F.log_softmax(out, dim=1)  # origin: log_softmax
        return out, hidden

    def sample(self, num_samples, start_letter=0):
        """
        Samples the network and returns num_samples samples of length max_seq_len.

        Outputs: samples, hidden
            - samples: num_samples x max_seq_length (a sampled sequence in each row)
        """

        samples = torch.zeros(num_samples, self.max_seq_len).type(torch.LongTensor)

        h = self.init_hidden(num_samples)
        inp = autograd.Variable(torch.LongTensor([start_letter] * num_samples))

        if self.gpu:
            samples = samples.cuda()
            inp = inp.cuda()

        # 用LSTM随机生成batch_size个句子，生成下一个词的时候是按多项式分布来选择的，而不是概率最大那个
        for i in range(self.max_seq_len):
            out, h = self.forward(inp, h)  # out: num_samples x vocab_size
            out = torch.multinomial(torch.exp(out), 1)  # num_samples x 1 (sampling from each row)
            samples[:, i] = out.view(-1).data

            inp = out.view(-1)

        return samples

    def batchNLLLoss(self, inp, target):
        """
        Returns the NLL Loss for predicting target sequence.

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len

            inp should be target with <s> (start letter) prepended
        """

        loss_fn = nn.NLLLoss()
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)  # seq_len x batch_size
        target = target.permute(1, 0)  # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            loss += loss_fn(out, target[i])

        return loss  # per batch

    def batchPGLoss(self, inp, target, reward):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)

            inp should be target with <s> (start letter) prepended
        """

        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)  # seq_len x batch_size
        target = target.permute(1, 0)  # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            if cfg.if_reward:
                out, h = self.forward(inp[i], h, no_log=cfg.no_log)  # origin: F.log_softmax, no_log=False
            else:
                out, h = self.forward(inp[i], h, no_log=True)
            # TODO: should h be detached from graph (.detach())?
            for j in range(batch_size):
                if cfg.if_reward:
                    loss += -out[j][target.data[i][j]] * reward[j]  # origin: log(P(y_t|Y_1:Y_{t-1})) * Q
                else:
                    loss += out[j][target.data[i][j]] * (1 - reward[j])  # P(y_t|Y_1:Y_{t-1}) * (1 - Q)
                # print('reward: ', reward[j], 1 - reward[j])

        return loss / batch_size


class Leak_Generator(nn.Module):

    def __init__(self):
        super(Leak_Generator, self).__init__()

    def forward(self, *input):
        pass

    def sample(self, num_samples, start_letter=0):
        pass

    def batchNLLLoss(self, inp, target):
        pass

    def batchPGLoss(self, inp, target, reward):
        pass

    def ini_hidden(self, batch_size=1):
        pass
