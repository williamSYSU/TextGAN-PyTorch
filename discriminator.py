import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pdb


class Discriminator(nn.Module):
    def __init__(self, embedding_dim, vocab_size, filter_sizes, num_filters, k_label, gpu=False, dropout=0.2):
        super(Discriminator, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, embedding_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(dropout)
        self.feature2out = nn.Linear(sum(num_filters), k_label + 1)

        self.init_parameters()

    def forward(self, x):
        """
        Get final feature of discriminator
        :param x: batch_size * seq_len
        :return: pred: feature of discriminator
        """
        emb = self.embeddings(x).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # tensor: batch_size * total_num_filter
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

        return pred

    def batchClassify(self, inp):
        """
        Get scores of each label
        :param inp: batch_size * seq_len
        :return:
        """
        pred = self.forward(inp)
        pred = self.feature2out(self.dropout(pred))
        return pred

    def get_feature(self, inp):
        return self.forward(inp)

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)


class GRU_Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, k_label, gpu=False, dropout=0.2):
        super(GRU_Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2 * 2 * hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1 + k_label)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2 * 2 * 1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, input, hidden):
        # input dim                                                # batch_size x seq_len
        emb = self.embeddings(input)  # batch_size x seq_len x embedding_dim
        emb = emb.permute(1, 0, 2)  # seq_len x batch_size x embedding_dim
        _, hidden = self.gru(emb, hidden)  # 4 x batch_size x hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()  # batch_size x 4 x hidden_dim
        out = self.gru2hidden(hidden.view(-1, 4 * self.hidden_dim))  # batch_size x 4*hidden_dim
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)  # batch_size x 1
        # out = torch.sigmoid(out)
        return out

    def batchClassify(self, inp):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """

        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return out.view(-1)

    def batchClasssifySenti(self, inp):
        """
        Get scores of each label
        :param inp: batch_size x seq_len
        :return: out: batch_size x (k_label + 1)
        """

        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return out

    def batchBCELoss(self, inp, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.

         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """

        loss_fn = nn.BCELoss()
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return loss_fn(out, target)
