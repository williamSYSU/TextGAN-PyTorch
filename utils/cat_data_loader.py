# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : cat_data_loader.py
# @Time         : Created at 2019-05-31
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import random
from torch.utils.data import Dataset, DataLoader

from utils.text_process import *


class GANDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class CatGenDataIter:
    def __init__(self, samples_list, shuffle=None):
        self.batch_size = cfg.batch_size
        self.max_seq_len = cfg.max_seq_len
        self.start_letter = cfg.start_letter
        self.shuffle = cfg.data_shuffle if not shuffle else shuffle
        if cfg.if_real_data:
            self.word2idx_dict, self.idx2word_dict = load_dict(cfg.dataset)

        self.loader = DataLoader(
            dataset=GANDataset(self.__read_data__(samples_list)),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True)

        self.input = self._all_data_('input')
        self.target = self._all_data_('target')
        self.label = self._all_data_('label')  # from 0 to k-1, different from Discriminator label

    def __read_data__(self, samples_list):
        """
        input: same as target, but start with start_letter.
        """
        inp, target, label = self.prepare(samples_list)
        all_data = [{'input': i, 'target': t, 'label': l} for (i, t, l) in zip(inp, target, label)]
        return all_data

    def random_batch(self):
        """Randomly choose a batch from loader, please note that the data should not be shuffled."""
        idx = random.randint(0, len(self.loader) - 1)
        return list(self.loader)[idx]

    def _all_data_(self, col):
        return torch.cat([data[col].unsqueeze(0) for data in self.loader.dataset.data], 0)

    def prepare(self, samples_list, gpu=False):
        """Add start_letter to samples as inp, target same as samples"""
        all_samples = torch.cat(samples_list, dim=0).long()
        target = all_samples
        inp = torch.zeros(all_samples.size()).long()
        inp[:, 0] = self.start_letter
        inp[:, 1:] = target[:, :self.max_seq_len - 1]

        label = torch.zeros(all_samples.size(0)).long()
        for idx in range(len(samples_list)):
            start = sum([samples_list[i].size(0) for i in range(idx)])
            label[start: start + samples_list[idx].size(0)] = idx

        # shuffle
        perm = torch.randperm(inp.size(0))
        inp = inp[perm].detach()
        target = target[perm].detach()
        label = label[perm].detach()

        if gpu:
            return inp.cuda(), target.cuda(), label.cuda()
        return inp, target, label

    def load_data(self, filename):
        """Load real data from local file"""
        self.tokens = get_tokenlized(filename)
        samples_index = tokens_to_tensor(self.tokens, self.word2idx_dict)
        return self.prepare(samples_index)


class CatClasDataIter:
    """Classifier data loader, handle for multi label data"""

    def __init__(self, samples_list, given_target=None, shuffle=None):
        """
        - samples_list:  list of tensors, [label_0, label_1, ..., label_k]
        """
        self.batch_size = cfg.batch_size
        self.max_seq_len = cfg.max_seq_len
        self.start_letter = cfg.start_letter
        self.shuffle = cfg.data_shuffle if not shuffle else shuffle

        self.loader = DataLoader(
            dataset=GANDataset(self.__read_data__(samples_list, given_target)),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True)

        self.input = self._all_data_('input')
        self.target = self._all_data_('target')

    def __read_data__(self, samples_list, given_target=None):
        inp, target = self.prepare(samples_list, given_target)
        all_data = [{'input': i, 'target': t} for (i, t) in zip(inp, target)]
        return all_data

    def random_batch(self):
        idx = random.randint(0, len(self.loader) - 1)
        return list(self.loader)[idx]
        # return next(iter(self.loader))

    def _all_data_(self, col):
        return torch.cat([data[col].unsqueeze(0) for data in self.loader.dataset.data], 0)

    @staticmethod
    def prepare(samples_list, given_target=None, detach=True, gpu=False):
        """
        Build inp and target
        :param samples_list: list of tensors, [label_0, label_1, ..., label_k]
        :param given_target: given a target, len(samples_list) = 1
        :param detach: if detach input
        :param gpu: if use cuda
        :returns inp, target:
            - inp: sentences
            - target: label index, 0-label_0, 1-label_1, ..., k-label_k
        """
        if len(samples_list) == 1 and given_target is not None:
            inp = samples_list[0]
            if detach:
                inp = inp.detach()
            target = torch.LongTensor([given_target] * inp.size(0))
            if len(inp.size()) == 2:  # samples token, else samples onehot
                inp = inp.long()
        else:
            inp = torch.cat(samples_list, dim=0)  # !!!need .detach()
            if detach:
                inp = inp.detach()
            target = torch.zeros(inp.size(0)).long()
            if len(inp.size()) == 2:  # samples token, else samples onehot
                inp = inp.long()
            for idx in range(1, len(samples_list)):
                start = sum([samples_list[i].size(0) for i in range(idx)])
                target[start: start + samples_list[idx].size(0)] = idx

        # shuffle
        perm = torch.randperm(inp.size(0))
        inp = inp[perm]
        target = target[perm]

        if gpu:
            return inp.cuda(), target.cuda()
        return inp, target
