# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : SentiGAN-william
# @FileName     : config.py
# @Time         : Created at 2019-03-18
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import os
from datetime import datetime

# =====Program=====
if_test = False
if_reward = True
CUDA = True
multi_gpu = False
if_save = True
gen_pretrain = False
dis_pretrain = False

seq_update = True  # True，是否是更新整个序列
no_log = True  # False，是否取消log操作

# =====Basic Train=====
samples_num = 5000  # 10000
MLE_train_epoch = 100  # 100
ADV_train_epoch = 150  # 50
k_label = 2  # num of labels
batch_size = 32  # 32
max_seq_len = 20  # 20
start_letter = 0
vocab_size = 5000  # 5000

# =====Generator=====
ADV_g_step = 1  # 1
rollout_num = 32  # 3
gen_embed_dim = 32  # 32
gen_hidden_dim = 32  # 32

# =====Discriminator=====
d_step = 50  # 50
d_epoch = 5 if k_label == 2 else 3  # 3
ADV_d_step = 2  # 5
ADV_d_epoch = 5 if k_label == 2 else 3  # 3
dis_embed_dim = 64  # 64
dis_hidden_dim = 64  # 64

# =====Save Model and samples=====
# oracle_samples_path = './oracle_samples.trc'
# oracle_state_dict_path = './oracle_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
oracle_samples_path = './save/oracle{}_samples_NUM{}_lstm.pkl'
oracle_state_dict_path = './save/oracle{}_EMB32_HID32_VOC5000_SEQ20_lstm.pkl'

pretrain_root = './pretrain/NUM{}/'.format(samples_num)
pretrained_gen_path = pretrain_root + 'gen{}_MLE_pretrain_EMB32_HID32_VOC5000_SEQ20_lstm.pkl'
pretrained_dis_path = pretrain_root + 'dis_pretrain_K{}_EMB64_HID64_VOC5000_SEQ20_lstm.pkl'

tips = ''

# =====log=====
log_filename = './log/log_{}'.format(datetime.now().strftime('%m%d_%H%M'))
if os.path.exists(log_filename + '.txt'):
    i = 2
    while True:
        if not os.path.exists(log_filename + '_%d' % i + '.txt'):
            log_filename = log_filename + '_%d' % i
            break
        i += 1
# print('***Log in:', log_filename, '***')
if not if_test:
    log = open(log_filename + '.txt', 'w')
    dict_file = open(log_filename + '_dict.txt', 'w')

'''Create dir for model'''
dir_list = ['save', 'savefig', 'log', 'pretrain', 'log/tensor_log', 'dataset',
            'pretrain/NUM{}'.format(samples_num)]
for d in dir_list:
    if not os.path.exists(d):
        os.mkdir(d)

'''Automatically choose GPU or CPU'''
device_dict = {
    -1: 'cpu',
    0: 'cuda:0',
    1: 'cuda:1',
    2: 'cuda:2',
    3: 'cuda:3',
}

if torch.cuda.is_available():
    os.system('nvidia-smi -q -d Utilization | grep Gpu > log/gpu')
    util_gpu = [int(line.strip().split()[2]) for line in open('log/gpu', 'r')]

    gpu_count = torch.cuda.device_count()
    device = util_gpu.index(min(util_gpu))
else:
    device = -1
# device = device_dict[device]
# device=3
torch.cuda.set_device(device)


# print('***Current Device: ', device_dict[device], '***')


def init_param(opt):
    global MLE_train_epoch, ADV_train_epoch, k_label, samples_num, batch_size, ADV_g_step, \
        rollout_num, d_step, d_epoch, ADV_d_step, ADV_d_epoch, vocab_size, max_seq_len, \
        start_letter, gen_embed_dim, gen_hidden_dim, dis_embed_dim, dis_hidden_dim, \
        CUDA, if_save, if_test, if_reward, gen_pretrain, dis_pretrain, log_filename, tips, \
        device, seq_update, no_log

    MLE_train_epoch = opt.mle_epoch
    ADV_train_epoch = opt.adv_epoch
    k_label = opt.k_label
    samples_num = opt.samples_num
    batch_size = opt.batch_size
    ADV_g_step = opt.adv_g_step
    rollout_num = opt.rollout_num
    d_step = opt.d_step
    d_epoch = opt.d_epoch
    ADV_d_step = opt.adv_d_step
    ADV_d_epoch = opt.adv_d_epoch

    vocab_size = opt.vocab_size
    max_seq_len = opt.max_seq_len
    start_letter = opt.start_letter
    gen_embed_dim = opt.gen_embed_dim
    gen_hidden_dim = opt.gen_hidden_dim
    dis_embed_dim = opt.dis_embed_dim
    dis_hidden_dim = opt.dis_hidden_dim

    device = opt.device
    seq_update = opt.seq_update
    no_log = opt.no_log
    CUDA = True if opt.cuda == 1 else False
    if_save = True if opt.if_save == 1 else False
    if_test = True if opt.if_test == 1 else False
    if_reward = True if opt.if_reward == 1 else False
    gen_pretrain = True if opt.gen_pretrain == 1 else False
    dis_pretrain = True if opt.dis_pretrain == 1 else False
    log_filename = opt.log_file
    tips = opt.tips


''' hvd command
mpirun -np 3 \
    -H localhost:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
'''
