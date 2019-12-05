# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : data_utils.py
# @Time         : Created at 2019-03-16
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.
from time import strftime, localtime

import torch.nn as nn

from metrics.nll import NLL
from models.Oracle import Oracle
from utils.data_loader import GenDataIter
from utils.text_process import *


def create_multi_oracle(number):
    for i in range(number):
        print('Creating Oracle %d...' % i)
        oracle = Oracle(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size,
                        cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        if cfg.CUDA:
            oracle = oracle.cuda()
        large_samples = oracle.sample(cfg.samples_num, 4 * cfg.batch_size)
        small_samples = oracle.sample(cfg.samples_num // 2, 4 * cfg.batch_size)

        torch.save(oracle.state_dict(), cfg.multi_oracle_state_dict_path.format(i))
        torch.save(large_samples, cfg.multi_oracle_samples_path.format(i, cfg.samples_num))
        torch.save(small_samples, cfg.multi_oracle_samples_path.format(i, cfg.samples_num // 2))

        oracle_data = GenDataIter(large_samples)
        mle_criterion = nn.NLLLoss()
        groud_truth = NLL.cal_nll(oracle, oracle_data.loader, mle_criterion)
        print('Oracle %d Groud Truth: %.4f' % (i, groud_truth))


def create_specific_oracle(from_a, to_b, num=1, save_path='../pretrain/'):
    for i in range(num):
        while True:
            oracle = Oracle(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size,
                            cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
            if cfg.CUDA:
                oracle = oracle.cuda()

            big_samples = oracle.sample(cfg.samples_num, 8 * cfg.batch_size)
            small_samples = oracle.sample(cfg.samples_num // 2, 8 * cfg.batch_size)

            oracle_data = GenDataIter(big_samples)
            mle_criterion = nn.NLLLoss()
            groud_truth = NLL.cal_nll(oracle, oracle_data.loader, mle_criterion)

            if from_a <= groud_truth <= to_b:
                dir_path = save_path + 'oracle_data_gt{:.2f}_{}'.format(groud_truth,
                                                                        strftime("%m%d_%H%M%S", localtime()))
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                print('save ground truth: ', groud_truth)
                # prefix = 'oracle{}_lstm_gt{:.2f}_{}'.format(i, groud_truth, strftime("%m%d", localtime()))
                prefix = dir_path + '/oracle_lstm'
                torch.save(oracle.state_dict(), '{}.pt'.format(prefix))
                torch.save(big_samples, '{}_samples_{}.pt'.format(prefix, cfg.samples_num))
                torch.save(small_samples, '{}_samples_{}.pt'.format(prefix, cfg.samples_num // 2))
                break


def create_many_oracle(from_a, to_b, num=1, save_path='../pretrain/'):
    for i in range(num):
        while True:
            oracle = Oracle(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size,
                            cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
            if cfg.CUDA:
                oracle = oracle.cuda()

            big_samples = oracle.sample(cfg.samples_num, 8 * cfg.batch_size)
            small_samples = oracle.sample(cfg.samples_num // 2, 8 * cfg.batch_size)

            oracle_data = GenDataIter(big_samples)
            mle_criterion = nn.NLLLoss()
            groud_truth = NLL.cal_nll(oracle, oracle_data.loader, mle_criterion)

            if from_a <= groud_truth <= to_b:
                print('save ground truth: ', groud_truth)
                prefix = 'oracle_lstm'
                torch.save(oracle.state_dict(), save_path + '{}.pt'.format(prefix))
                torch.save(big_samples, save_path + '{}_samples_{}.pt'.format(prefix, cfg.samples_num))
                torch.save(small_samples, save_path + '{}_samples_{}.pt'.format(prefix, cfg.samples_num // 2))
                break


def _save(data, filename):
    with open(filename, 'w') as fout:
        for d in data:
            fout.write(d['reviewText'] + '\n')
            fout.write(str(d['overall']) + '\n')


def _count(filename):
    with open(filename, 'r') as fin:
        data = fin.read().strip().split('\n')
        return len(data) / 2


def clean_amazon_long_sentence():
    data_root = '/home/sysu2018/Documents/william/amazon_dataset/'
    all_files = os.listdir(data_root)

    print('|\ttype\t|\torigin\t|\tclean_40\t|\tclean_20\t|\tfinal_40\t|\tfinal_20\t|')
    print('|----------|----------|----------|----------|----------|----------|')
    for file in all_files:
        filename = data_root + file
        if os.path.isdir(filename):
            continue

        clean_save_40 = []
        clean_save_20 = []
        final_save_40 = []
        final_save_20 = []
        with open(filename, 'r') as fin:
            raw_data = fin.read().strip().split('\n')
            for line in raw_data:
                review = eval(line)['reviewText']
                if len(review.split()) <= 40:
                    clean_save_40.append(eval(line))
                    if len(review.split('.')) <= 2:  # one sentence
                        final_save_40.append(eval(line))

                if len(review.split()) <= 20:
                    clean_save_20.append(eval(line))
                    if len(review.split('.')) <= 2:  # one sentence
                        final_save_20.append(eval(line))

        save_filename = data_root + 'clean_40/' + file.lower().split('_5')[0] + '.txt'
        _save(clean_save_40, save_filename)
        # a = _count(save_filename)
        save_filename = data_root + 'clean_20/' + file.lower().split('_5')[0] + '.txt'
        _save(clean_save_20, save_filename)
        # b = _count(save_filename)
        save_filename = data_root + 'final_40/' + file.lower().split('_5')[0] + '.txt'
        _save(final_save_40, save_filename)
        # c = _count(save_filename)
        save_filename = data_root + 'final_20/' + file.lower().split('_5')[0] + '.txt'
        _save(final_save_20, save_filename)
        # d = _count(save_filename)

        print('|\t%s\t|\t%d\t|\t%d\t|\t%d\t|\t%d\t|\t%d\t|' % (
            file.lower().split('_5')[0], len(raw_data),
            len(clean_save_40), len(clean_save_20),
            len(final_save_40), len(final_save_20)))
        # print('|\t%s\t|\t%d\t|\t%d\t|\t%d\t|\t%d\t|\t%d\t|' % (
        #     file.lower().split('_5')[0], len(raw_data), a, b, c, d))


def mean(x, y):
    return round((2 * x * y) / (x + y), 3)


def mean_list(x, y):
    res = []
    for i, j in zip(x, y):
        res.append(round(mean(i, j), 3))
    return res


if __name__ == '__main__':
    pass
