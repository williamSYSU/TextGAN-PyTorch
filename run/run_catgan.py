# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : run_catgan.py
# @Time         : Created at 2019-08-04
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
rootdir = '../'

# ===Program===
# CatGAN: Catgory text generation model
# EvoGAN: General text generation model
run_model = ['catgan', 'catgan', 'catgan', 'evogan', 'evogan', 'evogan']
MLE_train_epoch = 1
loss_type = 'ragan'
mu_type = 'ragan rsgan'
eval_type = 'Ra'
ADV_train_epoch = 2000

# === Real data===
if_real_data = [int(False), int(True), int(True), int(False), int(True), int(True)]
dataset = ['oracle', 'mr15', 'amazon_app_book', 'oracle', 'image_coco', 'emnlp_news']
temp_adpt = 'exp'
temperature = [1, 100, 100, 1, 100, 100]
tips = '{} experiments'

# === Basic Param ===
if_test = int(True)
ora_pretrain = int(True)
gen_pretrain = int(False)
data_shuffle = int(False)
gen_init = 'truncated_normal'
dis_init = 'uniform'
model_type = 'vanilla'
samples_num = 10000
n_parent = 1
ADV_d_step = 3
ADV_d_epoch = 1
mem_slots = 1
num_heads = 2
head_size = [512, 512, 512, 256, 256, 256]
pre_log_step = 10
adv_log_step = 20

# === EvoGAN Param ===
d_out_mean = int(True)
lambda_fq = 1.0
lambda_fd = 0.001
eval_b_num = 8

args = [
    # Compare Param
    '--device', gpu_id,
    '--run_model', run_model[job_id],
    '--mle_epoch', MLE_train_epoch,
    '--ora_pretrain', ora_pretrain,
    '--gen_pretrain', gen_pretrain,
    '--loss_type', loss_type,
    '--mu_type', mu_type,
    '--eval_type', eval_type,
    '--adv_epoch', ADV_train_epoch,
    '--tips', tips.format(run_model[job_id]),
    # Basic Param
    '--if_test', if_test,
    '--if_real_data', if_real_data[job_id],
    '--dataset', dataset[job_id],
    '--shuffle', data_shuffle,
    '--gen_init', gen_init,
    '--dis_init', dis_init,
    '--model_type', model_type,
    '--samples_num', samples_num,
    '--adv_d_step', ADV_d_step,
    '--adv_d_epoch', ADV_d_epoch,
    '--temp_adpt', temp_adpt,
    '--temperature', temperature[job_id],
    '--mem_slots', mem_slots,
    '--num_heads', num_heads,
    '--head_size', head_size[job_id],
    '--pre_log_step', pre_log_step,
    '--adv_log_step', adv_log_step,
    # EvoGAN Param
    '--d_out_mean', d_out_mean,
    '--n_parent', n_parent,
    '--lambda_fq', lambda_fq,
    '--lambda_fd', lambda_fd,
    '--eval_b_num', eval_b_num,
]

args = list(map(str, args))
my_env = os.environ.copy()
call([executable, 'main.py'] + args, env=my_env, cwd=rootdir)
