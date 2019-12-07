# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : text_process.py
# @Time         : Created at 2019-05-14
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import nltk
import numpy as np
import os
import torch

import config as cfg


def get_tokenlized(file):
    """tokenlize the file"""
    tokenlized = list()
    with open(file) as raw:
        for text in raw:
            text = nltk.word_tokenize(text.lower())
            tokenlized.append(text)
    return tokenlized


def get_word_list(tokens):
    """get word set"""
    word_set = list()
    for sentence in tokens:
        for word in sentence:
            word_set.append(word)
    return list(set(word_set))


def get_dict(word_set):
    """get word2idx_dict and idx2word_dict"""
    word2idx_dict = dict()
    idx2word_dict = dict()

    index = 2
    word2idx_dict[cfg.padding_token] = str(cfg.padding_idx)  # padding token
    idx2word_dict[str(cfg.padding_idx)] = cfg.padding_token
    word2idx_dict[cfg.start_token] = str(cfg.start_letter)  # start token
    idx2word_dict[str(cfg.start_letter)] = cfg.start_token

    for word in word_set:
        word2idx_dict[word] = str(index)
        idx2word_dict[str(index)] = word
        index += 1
    return word2idx_dict, idx2word_dict


def text_process(train_text_loc, test_text_loc=None):
    """get sequence length and dict size"""
    train_tokens = get_tokenlized(train_text_loc)
    if test_text_loc is None:
        test_tokens = list()
    else:
        test_tokens = get_tokenlized(test_text_loc)
    word_set = get_word_list(train_tokens + test_tokens)
    word2idx_dict, idx2word_dict = get_dict(word_set)

    if test_text_loc is None:
        sequence_len = len(max(train_tokens, key=len))
    else:
        sequence_len = max(len(max(train_tokens, key=len)), len(max(test_tokens, key=len)))

    return sequence_len, len(word2idx_dict)


# ============================================
def init_dict(dataset):
    """
    Initialize dictionaries of dataset, please note that '0': padding_idx, '1': start_letter.
    Finally save dictionary files locally.
    """
    tokens = get_tokenlized('dataset/{}.txt'.format(dataset))
    word_set = get_word_list(tokens)
    word2idx_dict, idx2word_dict = get_dict(word_set)

    with open('dataset/{}_wi_dict.txt'.format(dataset), 'w') as dictout:
        dictout.write(str(word2idx_dict))
    with open('dataset/{}_iw_dict.txt'.format(dataset), 'w') as dictout:
        dictout.write(str(idx2word_dict))

    print('total tokens: ', len(word2idx_dict))


def load_dict(dataset):
    """Load dictionary from local files"""
    iw_path = 'dataset/{}_iw_dict.txt'.format(dataset)
    wi_path = 'dataset/{}_wi_dict.txt'.format(dataset)

    if not os.path.exists(iw_path) or not os.path.exists(iw_path):  # initialize dictionaries
        init_dict(dataset)

    with open(iw_path, 'r') as dictin:
        idx2word_dict = eval(dictin.read().strip())
    with open(wi_path, 'r') as dictin:
        word2idx_dict = eval(dictin.read().strip())

    return word2idx_dict, idx2word_dict


def load_test_dict(dataset):
    """Build test data dictionary, extend from train data. For the classifier."""
    word2idx_dict, idx2word_dict = load_dict(dataset)  # train dict
    # tokens = get_tokenlized('dataset/testdata/{}_clas_test.txt'.format(dataset))
    tokens = get_tokenlized('dataset/testdata/{}_test.txt'.format(dataset))
    word_set = get_word_list(tokens)
    index = len(word2idx_dict)  # current index

    # extend dict with test data
    for word in word_set:
        if word not in word2idx_dict:
            word2idx_dict[word] = str(index)
            idx2word_dict[str(index)] = word
            index += 1
    return word2idx_dict, idx2word_dict


def tensor_to_tokens(tensor, dictionary):
    """transform Tensor to word tokens"""
    tokens = []
    for sent in tensor:
        sent_token = []
        for word in sent.tolist():
            if word == cfg.padding_idx:
                break
            sent_token.append(dictionary[str(word)])
        tokens.append(sent_token)
    return tokens


def tokens_to_tensor(tokens, dictionary):
    """transform word tokens to Tensor"""
    global i
    tensor = []
    for sent in tokens:
        sent_ten = []
        for i, word in enumerate(sent):
            if word == cfg.padding_token:
                break
            sent_ten.append(int(dictionary[str(word)]))
        while i < cfg.max_seq_len - 1:
            sent_ten.append(cfg.padding_idx)
            i += 1
        tensor.append(sent_ten[:cfg.max_seq_len])
    return torch.LongTensor(tensor)


def padding_token(tokens):
    """pad sentences with padding_token"""
    global i
    pad_tokens = []
    for sent in tokens:
        sent_token = []
        for i, word in enumerate(sent):
            if word == cfg.padding_token:
                break
            sent_token.append(word)
        while i < cfg.max_seq_len - 1:
            sent_token.append(cfg.padding_token)
            i += 1
        pad_tokens.append(sent_token)
    return pad_tokens


def write_tokens(filename, tokens):
    """Write word tokens to a local file (For Real data)"""
    with open(filename, 'w') as fout:
        for sent in tokens:
            fout.write(' '.join(sent))
            fout.write('\n')


def write_tensor(filename, tensor):
    """Write Tensor to a local file (For Oracle data)"""
    with open(filename, 'w') as fout:
        for sent in tensor:
            fout.write(' '.join([str(i) for i in sent.tolist()]))
            fout.write('\n')


def process_cat_text():
    import random

    dataset = 'mr'

    test_ratio = 0.3
    seq_len = 15

    pos_file = 'dataset/{}/{}{}_cat1.txt'.format(dataset, dataset, seq_len)
    neg_file = 'dataset/{}/{}{}_cat0.txt'.format(dataset, dataset, seq_len)
    pos_sent = open(pos_file, 'r').readlines()
    neg_sent = open(neg_file, 'r').readlines()

    pos_len = int(test_ratio * len(pos_sent))
    neg_len = int(test_ratio * len(neg_sent))

    random.shuffle(pos_sent)
    random.shuffle(neg_sent)

    all_sent_test = pos_sent[:pos_len] + neg_sent[:neg_len]
    all_sent_train = pos_sent[pos_len:] + neg_sent[neg_len:]
    random.shuffle(all_sent_test)
    random.shuffle(all_sent_train)

    f_pos_train = open('dataset/{}{}_cat1.txt'.format(dataset, seq_len), 'w')
    f_neg_train = open('dataset/{}{}_cat0.txt'.format(dataset, seq_len), 'w')
    f_pos_test = open('dataset/testdata/{}{}_cat1_test.txt'.format(dataset, seq_len), 'w')
    f_neg_test = open('dataset/testdata/{}{}_cat0_test.txt'.format(dataset, seq_len), 'w')

    for p_s in pos_sent[:pos_len]:
        f_pos_test.write(p_s)
    for n_s in neg_sent[:neg_len]:
        f_neg_test.write(n_s)
    for p_s in pos_sent[pos_len:]:
        f_pos_train.write(p_s)
    for n_s in neg_sent[neg_len:]:
        f_neg_train.write(n_s)

    with open('dataset/testdata/{}{}_test.txt'.format(dataset, seq_len), 'w') as fout:
        for sent in all_sent_test:
            fout.write(sent)
    with open('dataset/{}{}.txt'.format(dataset, seq_len), 'w') as fout:
        for sent in all_sent_train:
            fout.write(sent)

    f_pos_train.close()
    f_neg_train.close()
    f_pos_test.close()
    f_neg_test.close()


def combine_amazon_text():
    cat0_name = 'app'
    cat1_name = 'book'
    root_path = 'dataset/'
    cat0_train = open(root_path + cat0_name + '.txt', 'r').readlines()
    cat0_test = open(root_path + cat0_name + '_test.txt', 'r').readlines()
    cat1_train = open(root_path + cat1_name + '.txt', 'r').readlines()
    cat1_test = open(root_path + cat1_name + '_test.txt', 'r').readlines()

    with open(root_path + 'amazon_{}_{}.txt'.format(cat0_name, cat1_name), 'w') as fout:
        for sent in cat0_train:
            fout.write(sent)
        for sent in cat1_train:
            fout.write(sent)
    with open(root_path + 'testdata/amazon_{}_{}_test.txt'.format(cat0_name, cat1_name), 'w') as fout:
        for sent in cat0_test:
            fout.write(sent)
        for sent in cat1_test:
            fout.write(sent)


def extend_clas_train_data():
    data_name = 'mr'
    dataset = 'mr20'
    neg_filter_file = 'dataset/{}/{}_cat0.txt'.format(data_name, dataset)  # include train and test for generator
    pos_filter_file = 'dataset/{}/{}_cat1.txt'.format(data_name, dataset)
    neg_test_file = 'dataset/testdata/{}_cat0_test.txt'.format(dataset)
    pos_test_file = 'dataset/testdata/{}_cat1_test.txt'.format(dataset)
    neg_all_file = 'dataset/{}/{}_cat0.txt'.format(data_name, data_name)
    pos_all_file = 'dataset/{}/{}_cat1.txt'.format(data_name, data_name)

    neg_filter = open(neg_filter_file, 'r').readlines()
    pos_filter = open(pos_filter_file, 'r').readlines()
    neg_test = open(neg_test_file, 'r').readlines()
    pos_test = open(pos_test_file, 'r').readlines()
    neg_all = open(neg_all_file, 'r').readlines()
    pos_all = open(pos_all_file, 'r').readlines()

    # print('neg filter:', len(neg_filter))
    # print('neg test:', len(neg_test))
    # print('neg all:', len(neg_all))
    # print('pos filter:', len(pos_filter))
    # print('pos test:', len(pos_test))
    # print('pos all:', len(pos_all))

    print('neg before:', len(neg_test))
    for line in neg_all:
        if line not in neg_filter:
            neg_test.append(line)
    print('neg after:', len(neg_test))

    print('pos before:', len(pos_test))
    for line in pos_all:
        if line not in pos_filter:
            pos_test.append(line)
    print('pos after:', len(pos_test))

    with open('dataset/testdata/{}_cat0_clas_test.txt'.format(dataset), 'w') as fout:
        for line in neg_test:
            fout.write(line)
    with open('dataset/testdata/{}_cat1_clas_test.txt'.format(dataset), 'w') as fout:
        for line in pos_test:
            fout.write(line)
    with open('dataset/testdata/{}_clas_test.txt'.format(dataset), 'w') as fout:
        for line in neg_test:
            fout.write(line)
        for line in pos_test:
            fout.write(line)


def load_word_vec(path, word2idx_dict=None, type='glove'):
    """Load word embedding from local file"""
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    if type == 'glove':
        word2vec_dict = {}
        for line in fin:
            tokens = line.rstrip().split()
            if word2idx_dict is None or tokens[0] in word2idx_dict.keys():
                word2vec_dict[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    elif type == 'word2vec':
        import gensim
        word2vec_dict = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    else:
        raise NotImplementedError('No such type: %s' % type)
    return word2vec_dict


def build_embedding_matrix(dataset):
    """Load or build Glove embedding matrix."""
    embed_filename = 'dataset/glove_embedding_300d_{}.pt'.format(dataset)
    if os.path.exists(embed_filename):
        print('Loading embedding:', embed_filename)
        embedding_matrix = torch.load(embed_filename)
    else:
        print('Loading Glove word vectors...')
        word2idx_dict, _ = load_dict(dataset)
        embedding_matrix = np.random.random((len(word2idx_dict) + 2, 300))  # 2 for padding token and start token
        fname = '../glove.42B.300d.txt'  # Glove file
        # fname = '../GoogleNews-vectors-negative300.bin' # Google Word2Vec file
        word2vec_dict = load_word_vec(fname, word2idx_dict=word2idx_dict, type='glove')
        print('Building embedding matrix:', embed_filename)
        for word, i in word2idx_dict.items():
            if word in word2vec_dict:
                # words not found in embedding index will be randomly initialized.
                embedding_matrix[int(i)] = word2vec_dict[word]
        embedding_matrix = torch.FloatTensor(embedding_matrix)
        torch.save(embedding_matrix, embed_filename)
    return embedding_matrix


if __name__ == '__main__':
    os.chdir('../')
    # process_cat_text()
    # load_test_dict('mr15')
    # extend_clas_train_data()
    pass
