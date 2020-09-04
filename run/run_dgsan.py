# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : run_dgsan.py
# @Time         : Created at 2020/4/21
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
executable = '/home/zhiwei/.virtualenvs/zhiwei/bin/python'  # specify your own python interpreter path here
rootdir = '../'
scriptname = 'main.py'

# ===Program===
if_test = int(False)
run_model = 'dgsan'
CUDA = int(True)
oracle_pretrain = int(True)
gen_pretrain = int(False)
dis_pretrain = int(False)
MLE_train_epoch = 0
ADV_train_epoch = 200
tips = 'DGSAN experiments'

# ===Oracle  or Real===
if_real_data = [int(False), int(True), int(True)]
dataset = ['oracle', 'image_coco', 'emnlp_news']
vocab_size = [5000, 0, 0]

# ===Basic Param===
data_shuffle = int(False)
model_type = 'vanilla'
gen_init = 'truncated_normal'
samples_num = 10000
batch_size = 64
max_seq_len = 20
gen_lr = 1e-2
pre_log_step = 10
adv_log_step = 5

# ===Generator===
gen_embed_dim = 32
gen_hidden_dim = 32

# ===Metrics===
use_nll_oracle = int(True)
use_nll_gen = int(True)
use_nll_div = int(True)
use_bleu = int(True)
use_self_bleu = int(True)
use_ppl = int(False)

args = [
    # Program
    '--if_test', if_test,
    '--run_model', run_model,
    '--cuda', CUDA,
    # '--device', gpu_id,  # comment for auto GPU
    '--ora_pretrain', oracle_pretrain,
    '--gen_pretrain', gen_pretrain,
    '--dis_pretrain', dis_pretrain,
    '--mle_epoch', MLE_train_epoch,
    '--adv_epoch', ADV_train_epoch,
    '--tips', tips,

    # Oracle or Real
    '--if_real_data', if_real_data[job_id],
    '--dataset', dataset[job_id],
    '--vocab_size', vocab_size[job_id],

    # Basic Param
    '--shuffle', data_shuffle,
    '--model_type', model_type,
    '--gen_init', gen_init,
    '--samples_num', samples_num,
    '--batch_size', batch_size,
    '--max_seq_len', max_seq_len,
    '--gen_lr', gen_lr,
    '--pre_log_step', pre_log_step,
    '--adv_log_step', adv_log_step,

    # Generator
    '--gen_embed_dim', gen_embed_dim,
    '--gen_hidden_dim', gen_hidden_dim,

    # Metrics
    '--use_nll_oracle', use_nll_oracle,
    '--use_nll_gen', use_nll_gen,
    '--use_nll_div', use_nll_div,
    '--use_bleu', use_bleu,
    '--use_self_bleu', use_self_bleu,
    '--use_ppl', use_ppl,
]

args = list(map(str, args))
my_env = os.environ.copy()
call([executable, scriptname] + args, env=my_env, cwd=rootdir)
