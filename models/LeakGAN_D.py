# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : LeakGAN_D.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

from models.discriminator import CNNDiscriminator


class LeakGAN_D(CNNDiscriminator):
    def __init__(self, embedding_dim, vocab_size, filter_sizes, num_filters, k_label, padding_idx, gpu=False,
                 dropout=0.2):
        super(LeakGAN_D, self).__init__(embedding_dim, vocab_size, filter_sizes, num_filters, k_label, padding_idx,
                                        gpu, dropout)
