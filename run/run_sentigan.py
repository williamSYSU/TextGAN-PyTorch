# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : run_sentigan.py
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
rootdir = '../'

# ===Program===
run_model = 'sentigan'
MLE_train_epoch = 120
PRE_clas_epoch = 5
samples_num = 10000
gen_pretrain = 1
ADV_train_epoch = 100

# ===Oracle or Real===
if_real_data = [int(False), int(True), int(True)]
dataset = ['oracle', 'mr15', 'amazon_app_book']
tips = 'SentiGAN experiments'

# === Basic Param ===
if_test = int(False)
ora_pretrain = int(True)
data_shuffle = int(False)
gen_init = 'normal'
dis_init = 'uniform'
model_type = 'vanilla'
loss_type = 'JS'
n_parent = 1
ADV_d_step = 5
ADV_d_epoch = 1
mem_slots = 1
num_heads = 2
head_size = 256
pre_log_step = 10
adv_log_step = 1

args = [
    # Compare Param
    '--device', gpu_id,
    '--run_model', run_model,
    '--mle_epoch', MLE_train_epoch,
    '--clas_pre_epoch', PRE_clas_epoch,
    '--ora_pretrain', ora_pretrain,
    '--gen_pretrain', gen_pretrain,
    '--loss_type', loss_type,
    '--adv_epoch', ADV_train_epoch,
    '--tips', tips,

    # Basic Param
    '--if_test', if_test,
    '--if_real_data', if_real_data[job_id],
    '--dataset', dataset,
    '--shuffle', data_shuffle,
    '--gen_init', gen_init,
    '--dis_init', dis_init,
    '--model_type', model_type,
    '--samples_num', samples_num,
    '--adv_d_step', ADV_d_step,
    '--adv_d_epoch', ADV_d_epoch,
    '--mem_slots', mem_slots,
    '--num_heads', num_heads,
    '--head_size', head_size,
    '--pre_log_step', pre_log_step,
    '--adv_log_step', adv_log_step,
]

args = list(map(str, args))
my_env = os.environ.copy()
call([executable, 'main.py'] + args, env=my_env, cwd=rootdir)
