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
# CatGAN: Catgory text generation model
# EvoGAN: General text generation model
if_test = int(False)
run_model = ['fixemgan', 'fixemgan', 'fixemgan', 'cat_fixemgan', 'cat_fixemgan', 'cat_fixemgan']
k_label = 2
CUDA = int(True)
batch_size = 32
noise_size = 1000
max_epochs = 20
batches_per_epoch = 200
tips = '{} experiments'

# ===Oracle or Real===
if_real_data = [int(False), int(True), int(True), int(False), int(True), int(True)]
dataset = ['mr15', 'amazon_app_book', 'image_coco', 'emnlp_news']
w2v_embedding_size = [100, 100, 100, 100, 100, 100]
w2v_window = 5
w2v_min_count = 30
w2v_workers = 30
vocab_size = [5000, 0, 0, 5000, 0, 0]

# ===CatGAN Param===
n_parent = 1
loss_type = 'fixem'
mu_type = 'ragan rsgan'
eval_type = 'Ra'
temp_adpt = 'exp'
d_out_mean = int(False)

# === Basic Param ===
data_shuffle = int(False)
model_type = 'fixem'
gen_init = 'truncated_normal'
dis_init = 'uniform'
samples_num = 10000
batch_size = 64
max_seq_len = [16, 40, 20, 16, 52]
gen_lr = 0.01
gen_adv_lr = 1e-4
dis_lr = 1e-4
pre_log_step = 10
adv_log_step = 20

# ===Generator===
ADV_g_step = 1
gen_embed_dim = 32
gen_hidden_dim = 32
mem_slots = 1
num_heads = 2
head_size = [512, 512, 512, 256, 256, 256]

# ===Discriminator===
ADV_d_step = 3
dis_embed_dim = 64
dis_hidden_dim = 64
num_rep = 64

# ===Metrics===
use_nll_oracle = int(True)
use_nll_gen = int(True)
use_nll_div = int(True)
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
    '--vocab_size', vocab_size[job_id],

    # W2V embeddings
    '--w2v_embedding_size', w2v_embedding_size[job_id],
    '--w2v_window', w2v_window,
    '--w2v_min_count', w2v_min_count,
    '--w2v_workers', w2v_workers,

    # CatGAN Param
    '--n_parent', n_parent,
    '--loss_type', loss_type,
    '--mu_type', mu_type,
    '--eval_type', eval_type,
    '--temp_adpt', temp_adpt,
    '--d_out_mean', d_out_mean,

    # Basic Param
    '--shuffle', data_shuffle,
    '--model_type', model_type,
    '--gen_init', gen_init,
    '--dis_init', dis_init,
    '--samples_num', samples_num,
    '--batch_size', batch_size,
    '--max_seq_len', max_seq_len[job_id],
    '--gen_lr', gen_lr,
    '--gen_adv_lr', gen_adv_lr,
    '--dis_lr', dis_lr,
    '--pre_log_step', pre_log_step,
    '--adv_log_step', adv_log_step,

    # Generator
    '--adv_g_step', ADV_g_step,
    '--gen_embed_dim', gen_embed_dim,
    '--gen_hidden_dim', gen_hidden_dim,
    '--mem_slots', mem_slots,
    '--num_heads', num_heads,
    '--head_size', head_size[job_id],

    # Discriminator
    '--adv_d_step', ADV_d_step,
    '--dis_embed_dim', dis_embed_dim,
    '--dis_hidden_dim', dis_hidden_dim,
    '--num_rep', num_rep,

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
