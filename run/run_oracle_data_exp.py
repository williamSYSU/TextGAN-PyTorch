# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : run_exp.py
# @Time         : Created at 2019-07-26
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
devices = str(device)

num_group = 5  # run num groups of exp
run_model = 'evogan'

# === Compare Param ===
MLE_train_epoch = 150
n_parent = 1
ora_pretrain = int(True)
gen_pretrain = int(True)
loss_type = 'ragan'
mu_type = 'rsgan ragan'
eval_type = 'Ra'
temperature = [1, 5]
ADV_train_epoch = [2000, 2000]
tips = '[Oracle data-Temperature] EvoGAN, temp = {}'

# === Basic Param ===
if_test = int(False)
if_real_data = int(False)
data_shuffle = int(False)
gen_init = 'truncated_normal'
dis_init = 'uniform'
model_type = 'vanilla'
samples_num = 10000
vocab_size = 5000
ADV_d_step = 5
ADV_d_epoch = 1
temp_adpt = 'exp'
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

for i in range(num_group * len(ADV_train_epoch)):
    job_id = i % len(ADV_train_epoch)
    args = [
        # Compare Param
        '--device', device,
        # '--devices', devices,
        '--run_model', run_model,
        '--mle_epoch', MLE_train_epoch,
        '--ora_pretrain', ora_pretrain,
        '--gen_pretrain', gen_pretrain,
        '--loss_type', loss_type,
        '--mu_type', mu_type,
        '--eval_type', eval_type,
        '--adv_epoch', ADV_train_epoch[job_id],
        '--tips', tips.format(temperature[job_id]),
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
