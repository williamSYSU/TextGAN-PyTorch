# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : data_loader.py
# @Time         : Created at 2019-05-31
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.

import random
from torch.utils.data import Dataset, DataLoader

import config as cfg
from utils.text_process import *


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
    def __init__(self, tokenized, labels, w2v, verbose, batch_size, batches_per_epoch):
        self.verbose=verbose


        labels, tokenized = zip(*[
            (label, tokens)
            for label, tokens in zip(labels, tokenized)
            if all(token in w2v.wv for token in tokens)
        ])

        self.labels = torch.tensor(labels, dtype=int)

        self.vectors = [vectorize_sentence(tokens, w2v, target_len=cfg.max_seq_len, padding_token = cfg.padding_token) for tokens in tokenized]
        self.vectors = np.stack(vectors, axis=0)
        self.vectors = torch.tensor(vectors, dtype=torch.float32)

        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size

        self.texts = set(" ".join(text[-cfg.max_seq_len:]) for text in texts)
        if self.verbose:
            print('texts examples', [txt for txt in self.texts][:3])


    def __iter__(self):
        batch_iterator = trange(self.batches_per_epoch) if self.verbose else range(self.batches_per_epoch)
        permutation = torch.randperm(len(self))
        self.vectors = self.vectors[permutation]
        self.labels = self.labels[permutation]

        for _ in batch_iterator:
            index = 0
            index += self.batch_size
            if index > len(self):
                # concatenating beginning of self.vectors
                yield (torch.cat((self.labels[index - self.batch_size: index], self.labels[:index-len(self)])).to(device),
                torch.cat((self.vectors[index - self.batch_size: index], self.vectors[:index-len(self)])).to(device))
                index = index % len(self)
            else:
                yield self.labels[index - self.batch_size: index].to(device), self.vectors[index - self.batch_size: index].to(device)


    def __len__(self):
        return len(self.vectors)

    def is_this_message_in_dataset(self, text):
        return text in self.texts
