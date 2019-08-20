# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : CSGAN_D.py
# @Time         : Created at 2019-07-30
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

from models.discriminator import CNNDiscriminator, CNNClassifier

# dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
# dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160]
dis_filter_sizes = [2, 3, 4, 5]
dis_num_filters = [200, 200, 200, 200]
clas_filter_sizes = [2]
clas_num_filters = [300]


class CSGAN_D(CNNDiscriminator):
    def __init__(self, k_label, embed_dim, vocab_size, padding_idx, gpu=False, dropout=0.2):
        super(CSGAN_D, self).__init__(embed_dim, vocab_size, dis_filter_sizes, dis_num_filters, padding_idx, gpu,
                                      dropout)


# Classifier
class CSGAN_C(CNNClassifier):
    def __init__(self, k_label, embed_dim, max_seq_len, num_rep, vocab_size, padding_idx, gpu=False, dropout=0.25):
        super(CSGAN_C, self).__init__(k_label, embed_dim, max_seq_len, num_rep, vocab_size, clas_filter_sizes,
                                      clas_num_filters, padding_idx, gpu, dropout)
