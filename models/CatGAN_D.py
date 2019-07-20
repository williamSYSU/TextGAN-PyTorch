# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : CatGAN_D.py
# @Time         : Created at 2019-05-28
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.discriminator import CNNDiscriminator

dis_filter_sizes = [2, 3, 4, 5]
dis_num_filters = [300, 300, 300, 300]


# Classifier
# >>>RelGAN discriminator, shared parameters
class CatGAN_C(CNNDiscriminator):
    def __init__(self, k_label, embed_dim, max_seq_len, num_rep, vocab_size, padding_idx, gpu=False, dropout=0.25):
        super(CatGAN_C, self).__init__(embed_dim, vocab_size, dis_filter_sizes, dis_num_filters, padding_idx,
                                       gpu, dropout)

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.feature_dim = sum(dis_num_filters)
        self.emb_dim_single = int(embed_dim / num_rep)

        self.embeddings = nn.Linear(vocab_size, embed_dim, bias=False)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, self.emb_dim_single), stride=(1, self.emb_dim_single)) for (n, f) in
            zip(dis_num_filters, dis_filter_sizes)
        ])

        # set this tag each time before forwarding, and set None after finishing
        self.dis_or_clas = None  # 'dis' or 'clas'

        # Discriminator part
        self.dis_highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.dis_feature2out = nn.Linear(self.feature_dim, 100)
        # self.dis_out2logits = nn.Linear(100, 1)
        self.dis_out2logits = nn.Linear(100, 1)

        # Classifier part
        self.clas_highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.clas_feature2out = nn.Linear(self.feature_dim, 100)
        self.clas_out2logits = nn.Linear(100, k_label)

        self.dropout = nn.Dropout(dropout)
        self.init_params()

    def forward(self, inp):
        """
        Get logits of discriminator
        :param inp: batch_size * seq_len * vocab_size
        :return logits: [batch_size] (1-D tensor)
        """
        if len(inp.size()) == 2:
            inp = F.one_hot(inp, self.vocab_size).float()
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim

        cons = [F.relu(conv(emb)) for conv in self.convs]  # [batch_size * num_filter * (seq_len-k_h+1) * num_rep]
        pools = [F.max_pool2d(con, (con.size(2), 1)).squeeze(2) for con in cons]  # [batch_size * num_filter * num_rep]
        pred = torch.cat(pools, 1)
        pred = pred.permute(0, 2, 1).contiguous().view(-1, self.feature_dim)  # (batch_size * num_rep) * feature_dim

        assert self.dis_or_clas, 'Need to set dis_or_clas before forward!!'
        if self.dis_or_clas == 'dis':
            dis_highway = self.dis_highway(pred)
            pred = torch.sigmoid(dis_highway) * F.relu(dis_highway) + (
                    1. - torch.sigmoid(dis_highway)) * pred  # dis_highway
            pred = self.dis_feature2out(self.dropout(pred))  # batch_size * num_rep * 100
            # logits = self.dis_out2logits(self.dropout(pred.view(-1, 100))).squeeze(1)  # [batch_size * num_rep]
            logits = self.dis_out2logits(self.dropout(pred)).squeeze(1)  # [batch_size * num_rep]
        else:
            clas_highway = self.clas_highway(pred)
            pred = torch.sigmoid(clas_highway) * F.relu(clas_highway) + (
                    1. - torch.sigmoid(clas_highway)) * pred  # clas_highway
            pred = self.clas_feature2out(self.dropout(pred))  # batch_size * num_rep * 100
            logits = self.clas_out2logits(self.dropout(pred)).squeeze(1)  # [batch_size * num_rep]

        return logits

    def init_params(self):
        for param in self.parameters():
            # if param.requires_grad:
            #     torch.nn.init.uniform_(param, -0.05, 0.05)

            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                torch.nn.init.normal_(param, std=stddev)

    def split_params(self):
        dis_params = list()
        clas_params = list()

        dis_params += self.embeddings.parameters()
        dis_params += self.convs.parameters()
        dis_params += self.dis_highway.parameters()
        dis_params += self.dis_feature2out.parameters()
        dis_params += self.dis_out2logits.parameters()

        clas_params += self.embeddings.parameters()
        clas_params += self.convs.parameters()
        clas_params += self.clas_highway.parameters()
        clas_params += self.clas_feature2out.parameters()
        clas_params += self.clas_out2logits.parameters()

        return dis_params, clas_params
