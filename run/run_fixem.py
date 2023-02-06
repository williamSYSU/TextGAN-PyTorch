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
scriptname = 'main.py'

# ===Program===
# EvoGAN: General text generation model
if_test = int(False)
run_model = ['fixemgan', 'cat_fixemgan', 'fixemgan', 'cat_fixemgan', 'fixemgan', 'fixemgan', 'fixemgan', 'cat_fixemgan', 'fixemgan']
k_label = 2
CUDA = int(True)
batch_size = 32
noise_size = 1000
max_epochs = 20
batches_per_epoch = 200
tips = '{} experiments'

# ===Oracle or Real===
if_real_data = [int(True), int(True), int(True), int(True), int(True), int(True), int(True), int(False), int(False)]
dataset = ['image_coco', 'mr20', 'mr20', 'mr15', 'mr15', 'amazon_app_book', 'emnlp_news', 'oracle', 'oracle']
w2v_embedding_size = 512 #low on ram #hyperparam
w2v_window = 5
w2v_min_count = 30
w2v_workers = 30
w2v_samples_num = 5_000_000
vocab_size = 5000

# ===CatGAN Param===
loss_type = 'fixem'
oracle_train_samples_num = 100_000

# === Basic Param ===
data_shuffle = int(False)
model_type = 'fixem'
gen_init = 'truncated_normal'
dis_init = 'uniform'
batch_size = 64
target_len = [16, 20, 20, 16, 16, 40, 48, 20, 20] # architechture requires to be divisible by 4

# ===Generator===
generator_complexity = 768 #hyperparam

# ===Discriminator===
discriminator_complexity = 512 #hyperparam

# ===Metrics===
use_nll_oracle = int(True)
use_nll_gen = int(False)
use_nll_div = int(False)
use_bleu = int(True)
use_self_bleu = int(True)
use_clas_acc = int(True)
use_ppl = int(False)

args = [
    # Program
    '--if_test', if_test,
    '--run_model', run_model[job_id],
    '--k_label', k_label,
    '--cuda', CUDA,
    # '--device', gpu_id,   # comment for auto GPU
    '--tips', tips.format(run_model[job_id]),

    # Oracle or Real
    '--if_real_data', if_real_data[job_id],
    '--dataset', dataset[job_id],
    '--vocab_size', vocab_size,

    # W2V embeddings
    '--w2v_embedding_size', w2v_embedding_size,
    '--w2v_window', w2v_window,
    '--w2v_min_count', w2v_min_count,
    '--w2v_workers', w2v_workers,
    '--w2v_samples_num', w2v_samples_num,

    # FixemGAN Param
    '--loss_type', loss_type,
    '--max_epochs', max_epochs,
    '--batches_per_epoch', batches_per_epoch,
    '--noise_size', noise_size,
    '--target_len', target_len[job_id],
    '--oracle_train_samples_num', oracle_train_samples_num,

    # Generator
    '--generator_complexity', generator_complexity,

    # Discriminator
    '--discriminator_complexity', discriminator_complexity,

    # Metrics
    '--use_nll_oracle', use_nll_oracle,
    '--use_nll_gen', use_nll_gen,
    '--use_nll_div', use_nll_div,
    '--use_bleu', use_bleu,
    '--use_self_bleu', use_self_bleu,
    '--use_clas_acc', use_clas_acc,
    '--use_ppl', use_ppl,
]

args = list(map(str, args))
my_env = os.environ.copy()
call([executable, scriptname] + args, env=my_env, cwd=rootdir)
