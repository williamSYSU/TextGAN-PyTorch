import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from utils.helpers import truncated_normal_


class CNNDiscriminator(nn.Module):
    def __init__(self, embed_dim, vocab_size, filter_sizes, num_filters, padding_idx, gpu=False,
                 dropout=0.2):
        super(CNNDiscriminator, self).__init__()
        self.embedding_dim = embed_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.feature_dim = sum(num_filters)
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, embed_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 2)
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def forward(self, inp):
        """
        Get final predictions of discriminator
        :param inp: batch_size * seq_len
        :return: pred: batch_size * seq_len * vocab_size
        """
        feature = self.get_feature(inp)
        pred = self.feature2out(self.dropout(feature))

        return pred

    def get_feature(self, inp):
        """
        Get feature vector of given sentences
        :param inp: batch_size * max_seq_len
        :return: batch_size * feature_dim
        """
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # tensor: batch_size * feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

        return pred

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad:
                if cfg.use_truncated_normal:
                    truncated_normal_(param, std=0.1)
                else:
                    torch.nn.init.normal_(param, std=0.1)


class GRUDiscriminator(nn.Module):

    def __init__(self, embedding_dim, vocab_size, hidden_dim, feature_dim, max_seq_len, padding_idx,
                 gpu=False, dropout=0.2):
        super(GRUDiscriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2 * 2 * hidden_dim, feature_dim)
        self.feature2out = nn.Linear(feature_dim, 2)
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2 * 2 * 1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, inp):
        """
        Get final feature of discriminator
        :param inp: batch_size * seq_len
        :return pred: batch_size * seq_len * vocab_size
        """
        feature = self.get_feature(inp)
        pred = self.feature2out(self.dropout(feature))

        return pred

    def get_feature(self, inp):
        """
        Get feature vector of given sentences
        :param inp: batch_size * max_seq_len
        :return: batch_size * feature_dim
        """
        hidden = self.init_hidden(inp.size(0))

        emb = self.embeddings(input)  # batch_size * seq_len * embedding_dim
        emb = emb.permute(1, 0, 2)  # seq_len * batch_size * embedding_dim
        _, hidden = self.gru(emb, hidden)  # 4 * batch_size * hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()  # batch_size * 4 * hidden_dim
        out = self.gru2hidden(hidden.view(-1, 4 * self.hidden_dim))  # batch_size * 4 * hidden_dim
        feature = torch.tanh(out)  # batch_size * feature_dim

        return feature

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad:
                if cfg.use_truncated_normal:
                    truncated_normal_(param, std=0.1)
                else:
                    torch.nn.init.normal_(param, std=0.1)
