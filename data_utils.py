# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : SentiGAN-william
# @FileName     : data_utils.py
# @Time         : Created at 2019-03-16
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from models.Oracle import Oracle
import config as cfg
import helpers
import os
import json

save_path = './save'


class GANDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class GenDataIter:
    def __init__(self, samples):
        self.batch_size = cfg.batch_size
        self.max_seq_len = cfg.max_seq_len
        self.start_letter = cfg.start_letter

        self.loader = DataLoader(
            dataset=GANDataset(self.__read_data__(samples)),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True)

    def __read_data__(self, samples):
        """
        input: same as target, but start with start_letter.
        """
        inp, target = self.prepare(samples)
        all_data = [{'input': i, 'target': t} for (i, t) in zip(inp, target)]
        return all_data

    def reset(self, samples):
        self.loader.dataset = GANDataset(self.__read_data__(samples))
        return self.loader

    def randam_batch(self):
        return next(iter(self.loader))

    def prepare(self, samples, gpu=False):
        """Add start_letter to samples as inp, target same as samples"""
        inp = torch.zeros(samples.size())
        target = samples
        inp[:, 0] = self.start_letter
        inp[:, 1:] = target[:, :self.max_seq_len - 1]

        inp, target = inp.long(), target.long()
        if gpu:
            return inp.cuda(), target.cuda()
        return inp, target


class DisDataIter:
    def __init__(self, pos_samples, neg_samples):
        self.batch_size = cfg.batch_size
        self.max_seq_len = cfg.max_seq_len
        self.start_letter = cfg.start_letter

        self.loader = DataLoader(
            dataset=GANDataset(self.__read_data__(pos_samples, neg_samples)),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True)

    def __read_data__(self, pos_samples, neg_samples):
        """
        input: same as target, but start with start_letter.
        """
        inp, target = self.prepare(pos_samples, neg_samples)
        all_data = [{'input': i, 'target': t} for (i, t) in zip(inp, target)]
        return all_data

    def reset(self, pos_samples, neg_samples):
        self.loader.dataset = GANDataset(self.__read_data__(pos_samples, neg_samples))
        return self.loader

    def randam_batch(self):
        return next(iter(self.loader))

    def prepare(self, pos_samples, neg_samples, gpu=False):
        """Build inp and target"""
        inp = torch.cat((pos_samples, neg_samples), dim=0).long()
        target = torch.ones(pos_samples.size(0) + neg_samples.size(0)).long()
        target[pos_samples.size(0):] = 0

        # shuffle
        perm = torch.randperm(target.size(0))
        target = target[perm]
        inp = inp[perm]

        if gpu:
            return inp.cuda(), target.cuda()
        return inp, target


def create_oracle():
    oracle = Oracle(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size,
                    cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
    oracle = oracle.cuda()

    torch.save(oracle.state_dict(), cfg.oracle_state_dict_path)

    large_samples = oracle.sample(cfg.samples_num, cfg.batch_size)
    torch.save(large_samples, cfg.oracle_samples_path.format(cfg.samples_num))

    # count ground truth
    dis = None
    ground_nll_loss = helpers.batchwise_oracle_nll(oracle, dis, oracle, cfg.samples_num, cfg.batch_size,
                                                   cfg.max_seq_len, gpu=cfg.CUDA)
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
    create_oracle()
    # clean_amazon_long_sentence()
