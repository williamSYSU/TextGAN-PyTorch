# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : run_senti_review.py
# @Time         : Created at 2019-08-04
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import sys
from subprocess import call

import os

if len(sys.argv) == 2:
    device = int(sys.argv[1])
else:
    device = 0

# Executables
executable = '/home/zhiwei/.virtualenvs/zhiwei/bin/python'
rootdir = '../'

num_group = 10  # run num groups of exp
run_model = 'evocatgan'  # catgan, evocatgan

# === Compare Param ===
MLE_train_epoch = 200
gen_pretrain = [1, 1]
loss_type = 'nsgan'
mu_type = 'nsgan rsgan'
eval_type = 'Ra'
ADV_train_epoch = [1500, 1500]

# === Real data===
if_real_data = int(True)
dataset = 'mr15'  # mr15, br15, cr15
temp_adpt = 'exp'
temperature = [100, 1000]
tips = '[Real data-Sentiment Reviews (MR, BR, CR)] EvoCatGAN with temp{}, dataset={}, with head_size=512'.format(
    temperature, dataset)

# === Basic Param ===
if_test = int(False)
ora_pretrain = int(True)
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
head_size = 512
pre_log_step = 10
adv_log_step = 20

# === EvoGAN Param ===
use_all_real_fake = int(False)
use_population = int(False)
d_out_mean = int(True)
lambda_fq = 1.0
lambda_fd = 0.0
eval_b_num = 8

for i in range(num_group * len(ADV_train_epoch)):
    job_id = i % len(ADV_train_epoch)
    args = [
        # Compare Param
        '--device', device,
        '--run_model', run_model,
        '--mle_epoch', MLE_train_epoch,
        '--ora_pretrain', ora_pretrain,
        '--gen_pretrain', gen_pretrain[job_id],
        '--loss_type', loss_type,
        '--mu_type', mu_type,
        '--eval_type', eval_type,
        '--adv_epoch', ADV_train_epoch[job_id],
        '--tips', tips,
        # Basic Param
        '--if_test', if_test,
        '--if_real_data', if_real_data,
        '--dataset', dataset,
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
        '--head_size', head_size,
        '--pre_log_step', pre_log_step,
        '--adv_log_step', adv_log_step,
        # EvoGAN Param
        '--use_all_real_fake', use_all_real_fake,
        '--use_population', use_population,
        '--d_out_mean', d_out_mean,
        '--n_parent', n_parent,
        '--lambda_fq', lambda_fq,
        '--lambda_fd', lambda_fd,
        '--eval_b_num', eval_b_num,
    ]

    args = list(map(str, args))
    my_env = os.environ.copy()
    call([executable, 'main.py'] + args, env=my_env, cwd=rootdir)
