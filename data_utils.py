# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : SentiGAN-william
# @FileName     : data_utils.py
# @Time         : Created at 2019-03-16
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
from models.Oracle import Oracle
import config as cfg
import helpers
import os
import json

K_LABEL = 1
save_path = './save'


def create_senti_oracle():
    for _ in range(1, K_LABEL + 1):
        oracle = Oracle(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size,
                                         cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        oracle = oracle.cuda()

        if cfg.gen_pretrain:
            oracle.load_state_dict(torch.load(cfg.oracle_state_dict_path))
        else:
            torch.save(oracle.state_dict(), cfg.oracle_state_dict_path)

        samples = oracle.sample(5000)
        torch.save(samples, cfg.oracle_samples_path.format(5000))

        large_samples = oracle.sample(10000)
        torch.save(large_samples, cfg.oracle_samples_path.format(10000))

        # count ground truth
        dis = None
        ground_nll_loss = helpers.batchwise_oracle_nll(oracle, dis, oracle, 10000, 64, 20, gpu=cfg.CUDA)
        print('ground nll loss: ', ground_nll_loss)


def clean_amazon_long_sentence():
    data_root = '/home/sysu2018/Documents/william/amazon_dataset/'
    all_files = os.listdir(data_root)

    for file in all_files:
        filename = data_root + file
        if os.path.isdir(filename):
            continue
        print('>>>filename: ', filename)
        save_data = []
        with open(filename, 'r') as fin:
            raw_data = fin.read().strip().split('\n')
            print('original: ', len(raw_data))
            for line in raw_data:
                if len(eval(line)['reviewText'].split()) <= 40:
                    save_data.append(eval(line))
        print('after clean: ', len(save_data))

        save_filename = data_root + 'clean/' + file.lower().split('_5')[0] + '.txt'
        with open(save_filename, 'w') as fout:
            json.dump(save_data, fout)
        print('saved in ', save_filename)


if __name__ == '__main__':
    create_senti_oracle()
    # clean_amazon_long_sentence()
