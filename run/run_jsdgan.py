# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : run_jsdgan.py
# @Time         : Created at 2019/11/29
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
run_model = 'jsdgan'
CUDA = int(True)
if_real_data = [int(False), int(True), int(True)]
data_shuffle = int(False)
gen_init = 'normal'
oracle_pretrain = int(True)
gen_pretrain = int(False)
tips = 'JSDGAN experiments'

# =====Oracle  or Real=====
dataset = ['oracle', 'image_coco', 'emnlp_news']
model_type = 'vanilla'
loss_type = 'JS'
vocab_size = [5000, 4683, 5256]
temperature = 1

# =====Basic Train=====
samples_num = 10000
MLE_train_epoch = 0  # no pre-training
ADV_train_epoch = 500
batch_size = 64
max_seq_len = 20
gen_lr = 0.01
pre_log_step = 20
adv_log_step = 5

# =====Generator=====
ADV_g_step = 1
gen_embed_dim = 32
gen_hidden_dim = 32

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
    # '--device', 0,  # comment for auto GPU
    '--shuffle', data_shuffle,
    '--gen_init', gen_init,
    '--tips', tips.format(run_model),

    # Basic Train
    '--samples_num', samples_num,
    '--vocab_size', vocab_size[job_id],
    '--mle_epoch', MLE_train_epoch,
    '--adv_epoch', ADV_train_epoch,
    '--batch_size', batch_size,
    '--max_seq_len', max_seq_len,
    '--gen_lr', gen_lr,
    '--pre_log_step', pre_log_step,
    '--adv_log_step', adv_log_step,
    '--temperature', temperature,
    '--ora_pretrain', oracle_pretrain,
    '--gen_pretrain', gen_pretrain,

    # Generator
    '--adv_g_step', ADV_g_step,
    '--gen_embed_dim', gen_embed_dim,
    '--gen_hidden_dim', gen_hidden_dim,
]

args = list(map(str, args))
my_env = os.environ.copy()
call([executable, scriptname] + args, env=my_env, cwd=rootdir)
