# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : data_loader.py
# @Time         : Created at 2019-05-31
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.

import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange

import config as cfg
from utils.text_process import (
    tokens_to_tensor,
    get_tokenlized,
    load_dict,
    load_test_dict,
    vectorize_sentence,
)


class GANDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class GenDataIter:
    def __init__(self, samples, if_test_data=False, shuffle=None):
        self.batch_size = cfg.batch_size
        self.max_seq_len = cfg.max_seq_len
        self.start_letter = cfg.start_letter
        self.shuffle = cfg.data_shuffle if not shuffle else shuffle
        if cfg.if_real_data:
            self.word2idx_dict, self.idx2word_dict = load_dict(cfg.dataset)
        if if_test_data:  # used for the classifier
            self.word2idx_dict, self.idx2word_dict = load_test_dict(cfg.dataset)

        self.loader = DataLoader(
            dataset=GANDataset(self.__read_data__(samples)),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True)

        self.input = self._all_data_('input')
        self.target = self._all_data_('target')

    def __read_data__(self, samples):
        """
        input: same as target, but start with start_letter.
        """
        # global all_data
        if isinstance(samples, torch.Tensor):  # Tensor
            inp, target = self.prepare(samples)
            all_data = [{'input': i, 'target': t} for (i, t) in zip(inp, target)]
        elif isinstance(samples, str):  # filename
            inp, target = self.load_data(samples)
            all_data = [{'input': i, 'target': t} for (i, t) in zip(inp, target)]
        else:
            all_data = None
        return all_data

    def random_batch(self):
        """Randomly choose a batch from loader, please note that the data should not be shuffled."""
        idx = random.randint(0, len(self.loader) - 1)
        return list(self.loader)[idx]

    def _all_data_(self, col):
        return torch.cat([data[col].unsqueeze(0) for data in self.loader.dataset.data], 0)

    @staticmethod
    def prepare(samples, gpu=False):
        """Add start_letter to samples as inp, target same as samples"""
        inp = torch.zeros(samples.size()).long()
        target = samples
        inp[:, 0] = cfg.start_letter
        inp[:, 1:] = target[:, :cfg.max_seq_len - 1]

        if gpu:
            return inp.cuda(), target.cuda()
        return inp, target

    def load_data(self, filename):
        """Load real data from local file"""
        self.tokens = get_tokenlized(filename)
        samples_index = tokens_to_tensor(self.tokens, self.word2idx_dict)
        return self.prepare(samples_index)


class DisDataIter:
    def __init__(self, pos_samples, neg_samples, shuffle=None):
        self.batch_size = cfg.batch_size
        self.max_seq_len = cfg.max_seq_len
        self.start_letter = cfg.start_letter
        self.shuffle = cfg.data_shuffle if not shuffle else shuffle

        self.loader = DataLoader(
            dataset=GANDataset(self.__read_data__(pos_samples, neg_samples)),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True)

    def __read_data__(self, pos_samples, neg_samples):
        """
        input: same as target, but start with start_letter.
        """
        inp, target = self.prepare(pos_samples, neg_samples)
        all_data = [{'input': i, 'target': t} for (i, t) in zip(inp, target)]
        return all_data

    def random_batch(self):
        idx = random.randint(0, len(self.loader) - 1)
        return list(self.loader)[idx]

    def prepare(self, pos_samples, neg_samples, gpu=False):
        """Build inp and target"""
        inp = torch.cat((pos_samples, neg_samples), dim=0).long().detach()  # !!!need .detach()
        target = torch.ones(inp.size(0)).long()
        target[pos_samples.size(0):] = 0

        # shuffle
        perm = torch.randperm(inp.size(0))
        inp = inp[perm]
        target = target[perm]

        if gpu:
            return inp.cuda(), target.cuda()
        return inp, target


class DataSupplier:
    def __init__(self, tokenized, labels, w2v, batch_size, batches_per_epoch):
        labels, tokenized = zip(*[
            (label, tokens)
            for label, tokens in zip(labels, tokenized)
            if all(token in w2v.wv for token in tokens)
        ])

        self.labels = torch.tensor(labels, dtype=int)

        self.tokenized = np.array(tokenized)

        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size

        self.w2v = w2v

        self.texts = set(" ".join(tokens[-cfg.target_len:]) for tokens in tokenized)
        print('dataset random texts examples\n', '\n'.join([txt for txt in self.texts][:5]))

    def vectorize_batch(self, tokenized):
        vectors = [
            vectorize_sentence(
                tokens,
                self.w2v,
                target_len=cfg.target_len,
                padding_token = cfg.padding_token,
            )
            for tokens in tokenized
        ]
        vectors = np.stack(vectors, axis=0)
        vectors = torch.tensor(vectors, dtype=torch.float32)
        return vectors

    def __iter__(self):
        permutation = torch.randperm(len(self))
        self.tokenized = self.tokenized[permutation]
        self.labels = self.labels[permutation]

        for _ in range(self.batches_per_epoch):
            index = 0
            index += self.batch_size
            if index > len(self):
                # concatenating beginning of self.vectors
                yield (
                    torch.cat((self.labels[index - self.batch_size: index], self.labels[:index-len(self)])),
                    torch.cat((
                        self.vectorize_batch(self.tokenized[index - self.batch_size: index]),
                        self.vectorize_batch(self.tokenized[:index-len(self)])
                    ))
                )
                index = index % len(self)
            else:
                yield self.labels[index - self.batch_size: index], self.vectorize_batch(self.tokenized[index - self.batch_size: index])


    def __len__(self):
        return len(self.tokenized)

    def is_this_message_in_dataset(self, text):
        return text in self.texts
