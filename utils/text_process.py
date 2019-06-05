# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : text_process.py
# @Time         : Created at 2019-05-14
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import nltk
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
    """get word_index_dict and index_word_dict"""
    word_index_dict = dict()
    index_word_dict = dict()

    index = 2
    word_index_dict[cfg.padding_token] = str(cfg.padding_idx)
    index_word_dict[str(cfg.padding_idx)] = cfg.padding_token
    word_index_dict[cfg.start_token] = str(cfg.start_letter)
    index_word_dict[str(cfg.start_letter)] = cfg.start_token

    for word in word_set:
        word_index_dict[word] = str(index)
        index_word_dict[str(index)] = word
        index += 1
    return word_index_dict, index_word_dict


def text_precess(train_text_loc, test_text_loc=None):
    """get sequence length and dict size"""
    train_tokens = get_tokenlized(train_text_loc)
    if test_text_loc is None:
        test_tokens = list()
    else:
        test_tokens = get_tokenlized(test_text_loc)
    word_set = get_word_list(train_tokens + test_tokens)
    [word_index_dict, index_word_dict] = get_dict(word_set)

    if test_text_loc is None:
        sequence_len = len(max(train_tokens, key=len))
    else:
        sequence_len = max(len(max(train_tokens, key=len)), len(max(test_tokens, key=len)))

    # with open(oracle_file, 'w') as outfile:
    #     outfile.write(text_to_code(tokens, word_index_dict, seq_len))

    return sequence_len, len(word_index_dict) + 1


# ========================================================================
def init_dict():
    """
    Initialize dictionaries of dataset, please note that '0': padding_idx, '1': start_letter.
    Finally save dictionary files locally.
    """
    # image_coco
    tokens = get_tokenlized('dataset/image_coco.txt')
    tokens.extend(get_tokenlized('dataset/testdata/image_coco_test.txt'))
    word_set = get_word_list(tokens)
    word_index_dict, index_word_dict = get_dict(word_set)

    with open('dataset/image_coco_wi_dict.txt', 'w') as dictout:
        dictout.write(str(word_index_dict))
    with open('dataset/image_coco_iw_dict.txt', 'w') as dictout:
        dictout.write(str(index_word_dict))

    # emnlp
    tokens = get_tokenlized('dataset/emnlp_news.txt')
    tokens.extend(get_tokenlized('dataset/testdata/emnlp_news_test.txt'))
    word_set = get_word_list(tokens)
    word_index_dict, index_word_dict = get_dict(word_set)

    with open('dataset/emnlp_news_wi_dict.txt', 'w') as dictout:
        dictout.write(str(word_index_dict))
    with open('dataset/emnlp_news_iw_dict.txt', 'w') as dictout:
        dictout.write(str(index_word_dict))


def load_dict(dataset):
    """Load dictionary from local files"""
    iw_path = 'dataset/{}_iw_dict.txt'.format(dataset)
    wi_path = 'dataset/{}_wi_dict.txt'.format(dataset)

    if not os.path.exists(iw_path) or not os.path.exists(iw_path):  # initialize dictionaries
        init_dict()

    with open(iw_path, 'r') as dictin:
        index_word_dict = eval(dictin.read().strip())
    with open(wi_path, 'r') as dictin:
        word_index_dict = eval(dictin.read().strip())

    return word_index_dict, index_word_dict


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
