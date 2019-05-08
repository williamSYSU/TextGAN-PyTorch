import torch
from torch.autograd import Variable
from math import ceil
from fuzzywuzzy.fuzz import ratio
from functools import partial
import time
import numpy as np

import config as cfg


class Signal:
    def __init__(self, signal_file):
        self.signal_file = signal_file
        self.pre_sig = True
        self.adv_sig = True

        self.update()

    def update(self):
        signal_dict = self.read_signal(self.signal_file)
        self.pre_sig = signal_dict['pre_sig']
        self.adv_sig = signal_dict['adv_sig']

    def read_signal(self, signal_file):
        with open(signal_file, 'r') as fin:
            return eval(fin.read())


def prepare_generator_batch(samples, start_letter=cfg.start_letter, gpu=False):
    """
    Takes samples (a batch) and returns

    Inputs: samples, start_letter, cuda
        - samples: batch_size x seq_len (Tensor with a sample in each row)

    Returns: inp, target
        - inp: batch_size x seq_len (same as target, but with start_letter prepended)
        - target: batch_size x seq_len (Variable same as samples)
    """

    batch_size, seq_len = samples.size()

    inp = torch.zeros(batch_size, seq_len)
    target = samples
    inp[:, 0] = start_letter
    inp[:, 1:] = target[:, :seq_len - 1]

    inp = inp.long()
    target = target.long()

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target


def prepare_discriminator_data(pos_samples, neg_samples, gpu=False):
    """
    Takes positive (target) samples, negative (gen) samples and prepares inp and target loader for discriminator.

    Inputs: pos_samples, neg_samples
        - pos_samples: pos_size x seq_len
        - neg_samples: neg_size x seq_len

    Returns: inp, target
        - inp: (pos_size + neg_size) x seq_len
        - target: pos_size + neg_size (boolean 1/0)
    """

    inp = torch.cat((pos_samples, neg_samples), dim=0).long()
    target = torch.ones(pos_samples.size(0) + neg_samples.size(0)).long()
    target[pos_samples.size(0):] = 0

    # shuffle
    perm = torch.randperm(target.size(0))
    target = target[perm]
    inp = inp[perm]

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target


def prepare_senti_discriminator_data(pos_samples_list, neg_samples_list, k_label):
    """
    Prepare multi-class loader for discriminator
    :param pos_samples_list: list: k_label x sample_size or k_label x (k_label x sample_size)
    :param neg_samples_list: list: k_label x sample_size
    :param k_label:
    :param gpu:
    :return:
        - inp: (pos_size + neg_size) x seq_len
        - target: pos_size + neg_size, 0: neg, 1,2,...: pos
    """
    pos_samples = torch.cat(pos_samples_list, dim=0)
    neg_samples = torch.cat(neg_samples_list, dim=0)

    # initial
    inp = torch.cat((pos_samples, neg_samples), dim=0).long()
    target = torch.zeros(pos_samples.size(0) + neg_samples.size(0)).long()
    s_size = pos_samples_list[0].size(0)
    for i in range(k_label):
        target[s_size * i:s_size * (i + 1)] = i + 1

    # shuffle
    perm = torch.randperm(target.size(0))
    inp = inp[perm]
    target = target[perm]

    return inp, target


def batchwise_sample(gen, dis, num_samples, batch_size):
    """
    Sample num_samples samples batch_size samples at a time from gen.
    Does not require gpu since gen.sample() takes care of that.
    """

    samples = []
    for i in range(int(ceil(num_samples / float(batch_size)))):
        samples.append(gen.sample(batch_size, dis))

    return torch.cat(samples, 0)[:num_samples]


def batchwise_oracle_nll(gen, dis, oracle, num_samples, batch_size, max_seq_len, start_letter=cfg.start_letter,
                         gpu=False):
    # s = batchwise_sample(gen, dis, num_samples, batch_size)   # origin
    s = gen.sample(num_samples, batch_size, start_letter)
    oracle_nll = 0
    for i in range(0, num_samples, batch_size):
        inp, target = prepare_generator_batch(s[i:i + batch_size], start_letter, gpu)
        oracle_loss = oracle.batchNLLLoss(inp, target)
        oracle_nll += oracle_loss.data.item()

    return oracle_nll / (num_samples // batch_size)


def batchwise_gen_nll(gen, dis, samples, batch_size, max_seq_len, gpu=False):
    num_samples = len(samples)
    gen_nll = 0
    for i in range(0, num_samples, batch_size):
        target = samples[i:i + batch_size]
        if gpu:
            target = target.cuda()
        gen_loss = gen.batchNLLLoss(target, dis) / max_seq_len
        gen_nll += gen_loss.data.item()

    return gen_nll / (num_samples // batch_size)


def rec_func(samples, s_i):
    map_dist = partial(ratio, s_i)
    res = list(map(map_dist, samples))


def sent_distance(samples):
    s_size = len(samples)
    total_dist = 0

    # map_func = partial(rec_func, samples)
    # res = list(map(map_func, samples))

    for i in range(s_size):
        # t0 = time.time()

        # for j in range(s_size):
        #     total_dist += ratio(samples[i],samples[j])

        map_dist = partial(ratio, samples[i])
        res = list(map(map_dist, samples))

        total_dist += np.mean(res)

        # t1 = time.time()
        # print('time-sent distence: ', t1 - t0)

    return total_dist / s_size
