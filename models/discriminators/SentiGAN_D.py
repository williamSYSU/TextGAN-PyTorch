# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : SentiGAN_D.py
# @Time         : Created at 2019-07-26
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch.nn as nn

from models.discriminator import CNNDiscriminator, CNNClassifier

dis_filter_sizes = [2, 3, 4, 5]
dis_num_filters = [200, 200, 200, 200]

clas_filter_sizes = [2, 3, 4, 5]
clas_num_filters = [200]


class SentiGAN_D(CNNDiscriminator):
    def __init__(self, k_label, embed_dim, vocab_size, padding_idx, gpu=False, dropout=0.2):
        super(SentiGAN_D, self).__init__(embed_dim, vocab_size, dis_filter_sizes, dis_num_filters, padding_idx, gpu,
                                         dropout)

        self.feature2out = nn.Linear(self.feature_dim, k_label + 1)

        self.init_params()


# Classifier
class SentiGAN_C(CNNClassifier):
    def __init__(self, k_label, embed_dim, max_seq_len, num_rep, vocab_size, padding_idx, gpu=False, dropout=0.25):
        super(SentiGAN_C, self).__init__(k_label, embed_dim, max_seq_len, num_rep, vocab_size, clas_filter_sizes,
                                         clas_num_filters, padding_idx, gpu, dropout)

        # Use Glove
        # self.embeddings.from_pretrained(build_embedding_matrix(cfg.dataset))
