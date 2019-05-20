# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : SentiGAN-william
# @FileName     : data_utils.py
# @Time         : Created at 2019-03-16
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

from torch.utils.data import Dataset, DataLoader

from models.Oracle import Oracle
from utils.text_process import *

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
        self.shuffle = cfg.data_shuffle
        if cfg.if_real_data:
            self.word_index_dict, self.index_word_dict = load_dict(cfg.dataset)

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

    def reset(self, samples):
        self.loader.dataset = GANDataset(self.__read_data__(samples))
        self.input = self._all_data_('input')
        self.target = self._all_data_('target')
        return self.loader

    def randam_batch(self):
        return next(iter(self.loader))

    def _all_data_(self, col):
        return torch.cat([data[col].unsqueeze(0) for data in self.loader.dataset.data], 0)

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

    def load_data(self, filename):
        """Load real data from local file"""
        tokens = get_tokenlized(filename)
        samples_index = tokens_to_tensor(tokens, self.word_index_dict)
        return self.prepare(samples_index)


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
    # dis = None
    # ground_nll_loss = helpers.batchwise_oracle_nll(oracle, dis, oracle, cfg.samples_num, cfg.batch_size,
    #                                                cfg.max_seq_len, gpu=cfg.CUDA)
    # print('ground nll loss: ', ground_nll_loss)


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


if __name__ == '__main__':
    # create_oracle()
    # clean_amazon_long_sentence()
    pass
