import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import config as cfg


class LSTMGenerator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(LSTMGenerator, self).__init__()
        self.name = 'vanilla'

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.gpu = gpu

        self.temperature = 1

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm2out = nn.Linear(hidden_dim, vocab_size)

        self.init_params()

    def forward(self, inp, hidden):
        """
        Embeds input and applies LSTM
        :param inp: batch_size * seq_len
        :param hidden: (h, c)
        :param no_log: if log after Softmax
        """
        emb = self.embeddings(inp)  # batch_size * len * embedding_dim
        if len(inp.size()) == 1:
            emb = emb.unsqueeze(1)  # batch_size * 1 * embedding_dim

        out, hidden = self.lstm(emb, hidden)  # out: batch_size * seq_len * hidden_dim
        if len(inp.size()) == 1:
            out = out.view(-1, self.hidden_dim) # out: batch_size * hidden_dim
        out = self.lstm2out(out)  # batch_size * seq_len * vocab_size

        out = self.temperature * out  # temperature
        out = F.log_softmax(out, dim=-1)

        return out, hidden

    def sample(self, num_samples, batch_size, start_letter=cfg.start_letter):
        """
        Samples the network and returns num_samples samples of length max_seq_len.
        :return samples: num_samples * max_seq_length (a sampled sequence in each row)
        """
        num_batch = num_samples // batch_size + 1

        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()

        h = self.init_hidden(batch_size)

        # 用LSTM随机生成batch_size个句子，生成下一个词的时候是按多项式分布来选择的，而不是概率最大那个
        for b in range(num_batch):
            inp = torch.LongTensor([start_letter] * batch_size)
            if self.gpu:
                inp = inp.cuda()

            for i in range(self.max_seq_len):
                out, h = self.forward(inp, h)  # out: num_samples * vocab_size
                out = torch.multinomial(torch.exp(out), 1)  # num_samples * 1 (sampling from each row)
                samples[b * batch_size:(b + 1) * batch_size, i] = out.view(-1).data

                inp = out.view(-1)

        samples = samples[:num_samples]

        return samples

    def batchNLLLoss(self, inp, target):
        """
        Returns the NLL Loss for predicting target sequence.

        :param inp: batch_size * seq_len, inp should be target with <s> (start letter) prepended
        :param target: batch_size * seq_len
        :return loss: NLL loss
        """
        loss_fn = nn.NLLLoss()
        batch_size, seq_len = inp.size()
        h = self.init_hidden(batch_size)

        target = target.permute(1, 0)
        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[:, i], h)
            loss += loss_fn(out, target[i])

        return loss

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def init_oracle(self):
        for param in self.parameters():
            param.data.normal_(0, 1)

    def init_hidden(self, batch_size=1):
        # h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))
        h = torch.zeros(1, batch_size, self.hidden_dim)
        c = torch.zeros(1, batch_size, self.hidden_dim)

        if self.gpu:
            return h.cuda(), c.cuda()
        else:
            return h, c
