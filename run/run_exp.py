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

# Executables
executable = '/home/zhiwei/.virtualenvs/zhiwei/bin/python'
rootdir = '../'

run_model = 'evogan'
device = 1

loss_type = 'rsgan'
eval_type = 'nll'

MLE_train_epoch = 150
lambda_fq = 1.0
lambda_fd = 0.0

ora_pretrain = [0, 1, 1]
gen_pretrain = [0, 1, 1]
mu_type = ['rsgan', 'rsgan', 'rsgan nsgan']
ADV_train_epoch = [0, 3000, 3000]

for i in range(15):
    job_id = i % 3
    args = [
        '--device', device,
        '--run_model', run_model,
        '--ora_pretrain', ora_pretrain[job_id],
        '--gen_pretrain', gen_pretrain[job_id],
        '--lambda_fq', lambda_fq,
        '--lambda_fd', lambda_fd,
        '--loss_type', loss_type,
        '--mu_type', mu_type[job_id],
        '--eval_type', eval_type,
        '--mle_epoch', MLE_train_epoch,
        '--adv_epoch', ADV_train_epoch[job_id],
    ]

    args = list(map(str, args))
    my_env = os.environ.copy()
    call([executable, 'main.py'] + args, env=my_env, cwd=rootdir)
