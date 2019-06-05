# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : run_seqgan.py
# @Time         : Created at 2019-05-27
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import sys
from subprocess import call

import os

# Job id and gpu_id
if len(sys.argv) > 2:
    job_id = int(sys.argv[1])
    gpu_id = str(sys.argv[2])
    print('job_id: {}, gpu_id: {}'.format(job_id, gpu_id))
elif len(sys.argv) > 1:
    job_id = int(sys.argv[1])
    gpu_id = 0
    print('job_id: {}, missing gpu_id (use default {})'.format(job_id, gpu_id))
else:
    job_id = 0
    gpu_id = 0
    print('Missing argument: job_id and gpu_id. Use default job_id: {}, gpu_id: {}'.format(job_id, gpu_id))

# Executables
executable = 'python'

# =====Program=====
if_test = int(False)
run_model = 'seqgan'
CUDA = int(True)
if_real_data = [int(False), int(True)]
data_shuffle = int(False)
use_truncated_normal = int(False)
oracle_pretrain = int(True)
gen_pretrain = int(False)
dis_pretrain = int(False)

# =====Oracle  or Real=====
dataset = ['oracle', 'image_coco']
model_type = 'vanilla'
loss_type = 'JS'
vocab_size = [5000, 6613]
temperature = 1

# =====Basic Train=====
samples_num = 10000
MLE_train_epoch = 120
ADV_train_epoch = 200
batch_size = 64
max_seq_len = 20
gen_lr = 0.01
dis_lr = 1e-4
pre_log_step = 5
adv_log_step = 1

# =====Generator=====
ADV_g_step = 1
rollout_num = 32
gen_embed_dim = 32
gen_hidden_dim = 32

# =====Discriminator=====
d_step = 50
d_epoch = 3
ADV_d_step = 5
ADV_d_epoch = 3
dis_embed_dim = 64
dis_hidden_dim = 64

# =====Run=====
rootdir = '../'
scriptname = 'main.py'
cwd = os.path.dirname(os.path.abspath(__file__))

args = [
    # Program
    '--if_test', if_test,
    '--run_model', run_model,
    '--dataset', dataset[job_id],
    '--if_real_data', if_real_data[job_id],
    '--model_type', model_type,
    '--loss_type', loss_type,
    '--cuda', CUDA,
    # '--device', gpu_id,   # comment for auto GPU
    '--shuffle', data_shuffle,
    '--use_truncated_normal', use_truncated_normal,

    # Basic Train
    '--samples_num', samples_num,
    '--vocab_size', vocab_size[job_id],
    '--mle_epoch', MLE_train_epoch,
    '--adv_epoch', ADV_train_epoch,
    '--batch_size', batch_size,
    '--max_seq_len', max_seq_len,
    '--gen_lr', gen_lr,
    '--dis_lr', dis_lr,
    '--pre_log_step', pre_log_step,
    '--adv_log_step', adv_log_step,
    '--temperature', temperature,
    '--ora_pretrain', oracle_pretrain,
    '--gen_pretrain', gen_pretrain,
    '--dis_pretrain', dis_pretrain,

    # Generator
    '--adv_g_step', ADV_g_step,
    '--rollout_num', rollout_num,
    '--gen_embed_dim', gen_embed_dim,
    '--gen_hidden_dim', gen_hidden_dim,

    # Discriminator
    '--d_step', d_step,
    '--d_epoch', d_epoch,
    '--adv_d_step', ADV_d_step,
    '--adv_d_epoch', ADV_d_epoch,
    '--dis_embed_dim', dis_embed_dim,
    '--dis_hidden_dim', dis_hidden_dim,

    # Log
    '--tips', 'vanilla SeqGAN',
]

args = list(map(str, args))
my_env = os.environ.copy()
call([executable, scriptname] + args, env=my_env, cwd=rootdir)
