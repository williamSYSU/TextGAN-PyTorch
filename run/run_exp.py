# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : run_exp.py
# @Time         : Created at 2019-07-26
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

from subprocess import call

import os

# Executables
executable = '/home/zhiwei/.virtualenvs/zhiwei/bin/python'
rootdir = '../'

num_group = 4  # run num groups of exp
run_model = 'evogan'
device = 3

# === Compare Param ===
MLE_train_epoch = 150
n_parent = [1, 2, 4, 8]
ora_pretrain = int(True)
gen_pretrain = int(True)
loss_type = ['nsgan', 'nsgan', 'nsgan', 'nsgan']
mu_type = ['nsgan rsgan', 'nsgan rsgan', 'nsgan rsgan', 'nsgan rsgan']
eval_type = 'Ra'
ADV_train_epoch = 2500
tips = '[Compare Exp] EvoGAN (seq_len=20), compare n_parent. [Current] n_parent = {}. [Param] eval_type = Ra, loss_type = nsgan, mu_type = nsgan rsgan'
# tips = '[Compare Exp] EvoGAN (seq_len=20), compare evo and no_evo, set eval_type = Ra. [Current] loss_type = {}, mu_type = {}'

# === Basic Param ===
if_test = int(False)
if_real_data = int(False)
data_shuffle = int(False)
gen_init = 'truncated_normal'
dis_init = 'uniform'
model_type = 'vanilla'
samples_num = 10000
vocab_size = 5000
ADV_d_step = 3
ADV_d_epoch = 1
temp_adpt = 'exp'
temperature = 1
mem_slots = 1
num_heads = 2
head_size = 256
pre_log_step = 10
adv_log_step = 20

# === EvoGAN Param ===
use_all_real_fake = int(False)
use_population = int(False)
d_out_mean = int(True)
lambda_fq = 1.0
lambda_fd = 0.0
eval_b_num = 8

for i in range(num_group * len(loss_type)):
    job_id = i % len(loss_type)
    args = [
        # Compare Param
        '--device', device,
        '--run_model', run_model,
        '--mle_epoch', MLE_train_epoch,
        '--ora_pretrain', ora_pretrain,
        '--gen_pretrain', gen_pretrain,
        '--loss_type', loss_type[job_id],
        '--mu_type', mu_type[job_id],
        '--eval_type', eval_type,
        '--adv_epoch', ADV_train_epoch,
        '--tips', tips.format(n_parent[job_id]),
        # Basic Param
        '--if_test', if_test,
        '--if_real_data', if_real_data,
        '--shuffle', data_shuffle,
        '--gen_init', gen_init,
        '--dis_init', dis_init,
        '--model_type', model_type,
        '--samples_num', samples_num,
        '--vocab_size', vocab_size,
        '--adv_d_step', ADV_d_step,
        '--adv_d_epoch', ADV_d_epoch,
        '--temp_adpt', temp_adpt,
        '--temperature', temperature,
        '--mem_slots', mem_slots,
        '--num_heads', num_heads,
        '--head_size', head_size,
        '--pre_log_step', pre_log_step,
        '--adv_log_step', adv_log_step,
        # EvoGAN Param
        '--use_all_real_fake', use_all_real_fake,
        '--use_population', use_population,
        '--d_out_mean', d_out_mean,
        '--n_parent', n_parent[job_id],
        '--lambda_fq', lambda_fq,
        '--lambda_fd', lambda_fd,
        '--eval_b_num', eval_b_num,
    ]

    args = list(map(str, args))
    my_env = os.environ.copy()
    call([executable, 'main.py'] + args, env=my_env, cwd=rootdir)
