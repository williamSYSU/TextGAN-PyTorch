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
        self.samples = samples
        if type(self.samples) == str:  # we received filename
            self.samples = get_tokenlized(self.samples)

        self.shuffle = cfg.data_shuffle if not shuffle else shuffle

        if cfg.if_real_data:
            self.word2idx_dict, self.idx2word_dict = load_dict(cfg.dataset)
        if if_test_data:  # used for the classifier
            self.word2idx_dict, self.idx2word_dict = load_test_dict(cfg.dataset)

        self.loader = DataLoader(
            dataset=GANDataset(self.__read_data__(self.samples)),
            batch_size=cfg.batch_size,
            shuffle=self.shuffle,
            drop_last=True,
        )

        self.input = self._all_data_("input")
        self.target = self._all_data_("target")

    def __read_data__(self, samples):
        """
        input: same as target, but start with start_letter.
        """
        if isinstance(samples[0], str) or isinstance(
            samples[0][0], str
        ):  # list of strings
            # we directly generated string, skip NLL
            return [
                {"input": i, "target": t}
                for i, t in zip(torch.zeros(2), torch.zeros(2))
            ]
        if isinstance(samples[0], list):
            # need to transform to indexes
            samples = tokens_to_tensor(samples, self.word2idx_dict)
        inp, target = self.prepare_for_NLL(samples)
        all_data = [{"input": i, "target": t} for (i, t) in zip(inp, target)]
        return all_data

    def random_batch(self):
        """Randomly choose a batch from loader, please note that the data should not be shuffled."""
        idx = random.randint(0, len(self.loader) - 1)
        return list(self.loader)[idx]

    def _all_data_(self, col):
        return torch.cat(
            [data[col].unsqueeze(0) for data in self.loader.dataset.data], 0
        )

    @property
    def tokens(self):
        """Returns samples in form of list of tensors, if input tensor,
        or list of tokens in case if input string."""
        if type(self.samples[0]) == str:  # we have list of strings
            return [smpl.split() for smpl in self.samples]
        return list(self.samples)

    @staticmethod
    def prepare_for_NLL(samples, gpu=False):
        """Add start_letter to samples as inp, target same as samples"""
        inp = torch.zeros(samples.size()).long()
        target = samples
        inp[:, 0] = cfg.start_letter
        inp[:, 1:] = target[:, : cfg.max_seq_len - 1]

        if gpu:
            return inp.cuda(), target.cuda()
        return inp, target


class DisDataIter:
    def __init__(self, pos_samples, neg_samples, shuffle=None):
        self.shuffle = cfg.data_shuffle if not shuffle else shuffle

        self.loader = DataLoader(
            dataset=GANDataset(self.__read_data__(pos_samples, neg_samples)),
            batch_size=cfg.batch_size,
            shuffle=self.shuffle,
            drop_last=True,
        )

    def __read_data__(self, pos_samples, neg_samples):
        inp, target = self.prepare(pos_samples, neg_samples)
        all_data = [{"input": i, "target": t} for (i, t) in zip(inp, target)]
        return all_data

    def random_batch(self):
        idx = random.randint(0, len(self.loader) - 1)
        return list(self.loader)[idx]

    def prepare(self, pos_samples, neg_samples, gpu=False):
        """Build inp and target"""
        inp = (
            torch.cat((pos_samples, neg_samples), dim=0).long().detach()
        )  # !!!need .detach()
        target = torch.ones(inp.size(0)).long()
        target[pos_samples.size(0) :] = 0

        # shuffle
        perm = torch.randperm(inp.size(0))
        inp = inp[perm]
        target = target[perm]

        if gpu:
            return inp.cuda(), target.cuda()
        return inp, target


class DataSupplier:
    def __init__(self, tokenized, labels, w2v, batch_size, batches_per_epoch):
        labels, tokenized = zip(
            *[
                (label, tokens)
                for label, tokens in zip(labels, tokenized)
                if all(token in w2v.wv for token in tokens)
            ]
        )
        self.labels = torch.tensor(labels, dtype=int)
        self.tokenized = np.array(tokenized)
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.w2v = w2v
        self.texts = set(" ".join(tokens[-cfg.target_len :]) for tokens in tokenized)
        print(
            "dataset random texts examples\n",
            "\n".join([txt for txt in self.texts][:5]),
        )

    def vectorize_batch(self, tokenized):
        vectors = [
            vectorize_sentence(
                tokens,
                self.w2v,
                target_len=cfg.target_len,
                padding_token=cfg.padding_token,
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
                    torch.cat(
                        (
                            self.labels[index - self.batch_size : index],
                            self.labels[: index - len(self)],
                        )
                    ),
                    torch.cat(
                        (
                            self.vectorize_batch(
                                self.tokenized[index - self.batch_size : index]
                            ),
                            self.vectorize_batch(self.tokenized[: index - len(self)]),
                        )
                    ),
                )
                index = index % len(self)
            else:
                yield self.labels[
                    index - self.batch_size : index
                ], self.vectorize_batch(self.tokenized[index - self.batch_size : index])

    def __len__(self):
        return self.batches_per_epoch

    def is_message_in_dataset(self, text):
        return text in self.texts
