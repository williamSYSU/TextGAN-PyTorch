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
if_reward = True
CUDA = True
multi_gpu = False
if_save = True
oracle_pretrain = True  # True
gen_pretrain = False
dis_pretrain = False

seq_update = True  # True，是否是更新整个序列
no_log = False  # False，是否取消log操作。False: 有log，在算NLL loss时使用；True: 无log，采样时使用

run_model = 'relgan'

# =====Basic Train=====
samples_num = 10000  # 10000
MLE_train_epoch = 300  # 8
ADV_train_epoch = 1  # 150
inter_epoch = 1  # 10
k_label = 1  # num of labels
batch_size = 64  # 64
max_seq_len = 20  # 20
start_letter = 1
padding_idx = 0
vocab_size = 5000  # 5000
gen_lr = 0.01  # 0.01
dis_lr = 5e-5  # 5e-5 for CNN dis

# =====Generator=====
ADV_g_step = 1  # 1
rollout_num = 4  # 4
gen_embed_dim = 32  # 32
gen_hidden_dim = 32  # 32
goal_size = 16
step_size = 4

# =====Discriminator=====
d_step = 0  # 5
d_epoch = 3  # 3
ADV_d_step = 3  # 3
ADV_d_epoch = 3  # 3
if max_seq_len == 20:
    dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
    # gen_lr = 0.0015
    dis_embed_dim = 64
    dis_hidden_dim = 64
    goal_out_size = sum(dis_num_filters)
    # goal_out_size = 300
elif max_seq_len == 40:
    dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40]
    dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160, 160]
    # gen_lr = 0.0005
    dis_embed_dim = 64
    dis_hidden_dim = 64
    goal_out_size = sum(dis_num_filters)
    # goal_out_size = 300
else:
    exit(0)

# =====Save Model and samples=====
oracle_samples_path = './save/oracle_samples_NUM{}_lstm.pkl'
oracle_state_dict_path = './save/oracle_EMB32_HID32_VOC5000_SEQ20_lstm.pkl'

pretrain_root = './pretrain_generator/NUM{}/'.format(samples_num)
pretrained_gen_path = pretrain_root + 'gen_MLE_pretrain_EMB32_HID32_VOC5000_SEQ20_leak.pkl'
pretrained_dis_path = pretrain_root + 'dis_pretrain_EMB64_HID64_VOC5000_SEQ20_leak.pkl'
signal_file = 'run_signal.txt'

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

'''Create dir for model'''
dir_list = ['save', 'savefig', 'log', 'pretrain_generator', 'log/tensor_log', 'save/log', 'dataset',
            'pretrain_generator/NUM{}'.format(samples_num)]
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
# device=3
torch.cuda.set_device(device)
print('device: ', device)


def init_param(opt):
    global MLE_train_epoch, ADV_train_epoch, k_label, samples_num, batch_size, ADV_g_step, \
        rollout_num, d_step, d_epoch, ADV_d_step, ADV_d_epoch, vocab_size, max_seq_len, \
        start_letter, gen_embed_dim, gen_hidden_dim, dis_embed_dim, dis_hidden_dim, \
        CUDA, if_save, if_reward, gen_pretrain, dis_pretrain, log_filename, tips, \
        device, seq_update, no_log, oracle_pretrain, gen_lr, dis_lr, inter_epoch, run_model

    run_model = opt.run_model
    MLE_train_epoch = opt.mle_epoch
    ADV_train_epoch = opt.adv_epoch
    inter_epoch = opt.inter_epoch
    batch_size = opt.batch_size
    ADV_g_step = opt.adv_g_step
    rollout_num = opt.rollout_num
    d_step = opt.d_step
    d_epoch = opt.d_epoch
    ADV_d_step = opt.adv_d_step
    ADV_d_epoch = opt.adv_d_epoch
    gen_lr = opt.gen_lr
    dis_lr = opt.dis_lr

    device = opt.device
    CUDA = True if opt.cuda == 1 else False
    oracle_pretrain = True if opt.ora_pretrain == 1 else False
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
