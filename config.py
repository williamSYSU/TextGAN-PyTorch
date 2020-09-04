# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : config.py
# @Time         : Created at 2019-03-18
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.
import time
from time import strftime, localtime

import os
import re
import torch

# ===Program===
if_test = False
CUDA = True
multi_gpu = False
if_save = True
data_shuffle = False  # False
oracle_pretrain = True  # True
gen_pretrain = False
dis_pretrain = False
clas_pretrain = False

run_model = 'catgan'  # seqgan, leakgan, maligan, jsdgan, relgan, evogan, sentigan, catgan, dpgan, dgsan, cot
k_label = 2  # num of labels, >=2
gen_init = 'truncated_normal'  # normal, uniform, truncated_normal
dis_init = 'uniform'  # normal, uniform, truncated_normal

# ===CatGAN===
n_parent = 1
eval_b_num = 8  # >= n_parent*ADV_d_step
max_bn = 1 if eval_b_num > 1 else eval_b_num
lambda_fq = 1.0
lambda_fd = 0.0
d_out_mean = True
freeze_dis = False
freeze_clas = False
use_all_real_fake = False
use_population = False

# ===Oracle or Real, type===
if_real_data = False  # if use real data
dataset = 'oracle'  # oracle, image_coco, emnlp_news, amazon_app_book, amazon_app_movie, mr15
model_type = 'vanilla'  # vanilla, RMC (custom)
loss_type = 'rsgan'  # rsgan lsgan ragan vanilla wgan hinge, for Discriminator (CatGAN)
mu_type = 'ragan'  # rsgan lsgan ragan vanilla wgan hinge
eval_type = 'Ra'  # standard, rsgan, nll, nll-f1, Ra, bleu3, bleu-f1
d_type = 'Ra'  # S (Standard), Ra (Relativistic_average)
vocab_size = 5000  # oracle: 5000, coco: 4683, emnlp: 5256, amazon_app_book: 6418, mr15: 6289
max_seq_len = 20  # oracle: 20, coco: 37, emnlp: 51, amazon_app_book: 40
ADV_train_epoch = 2000  # SeqGAN, LeakGAN-200, RelGAN-3000
extend_vocab_size = 0  # plus test data, only used for Classifier

temp_adpt = 'exp'  # no, lin, exp, log, sigmoid, quad, sqrt
mu_temp = 'exp'  # lin exp log sigmoid quad sqrt
evo_temp_step = 1
temperature = 1

# ===Basic Train===
samples_num = 10000  # 10000, mr15: 2000,
MLE_train_epoch = 150  # SeqGAN-80, LeakGAN-8, RelGAN-150
PRE_clas_epoch = 10
inter_epoch = 15  # LeakGAN-10
batch_size = 64  # 64
start_letter = 1
padding_idx = 0
start_token = 'BOS'
padding_token = 'EOS'
gen_lr = 0.01  # 0.01
gen_adv_lr = 1e-4  # RelGAN-1e-4
dis_lr = 1e-4  # SeqGAN,LeakGAN-1e-2, RelGAN-1e-4
clas_lr = 1e-3
clip_norm = 5.0

pre_log_step = 10
adv_log_step = 20

train_data = 'dataset/' + dataset + '.txt'
test_data = 'dataset/testdata/' + dataset + '_test.txt'
cat_train_data = 'dataset/' + dataset + '_cat{}.txt'
cat_test_data = 'dataset/testdata/' + dataset + '_cat{}_test.txt'

# ===Metrics===
use_nll_oracle = True
use_nll_gen = True
use_nll_div = True
use_bleu = True
use_self_bleu = False
use_clas_acc = True
use_ppl = False

# ===Generator===
ADV_g_step = 1  # 1
rollout_num = 16  # 4
gen_embed_dim = 32  # 32
gen_hidden_dim = 32  # 32
goal_size = 16  # LeakGAN-16
step_size = 4  # LeakGAN-4

mem_slots = 1  # RelGAN-1
num_heads = 2  # RelGAN-2
head_size = 256  # RelGAN-256

# ===Discriminator===
d_step = 5  # SeqGAN-50, LeakGAN-5
d_epoch = 3  # SeqGAN,LeakGAN-3
ADV_d_step = 5  # SeqGAN,LeakGAN,RelGAN-5
ADV_d_epoch = 3  # SeqGAN,LeakGAN-3

dis_embed_dim = 64
dis_hidden_dim = 64
num_rep = 64  # RelGAN

# ===log===
log_time_str = strftime("%m%d_%H%M_%S", localtime())
log_filename = strftime("log/log_%s" % log_time_str)
if os.path.exists(log_filename + '.txt'):
    i = 2
    while True:
        if not os.path.exists(log_filename + '_%d' % i + '.txt'):
            log_filename = log_filename + '_%d' % i
            break
        i += 1
log_filename = log_filename + '.txt'

# Automatically choose GPU or CPU
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    os.system('nvidia-smi -q -d Utilization > gpu')
    with open('gpu', 'r') as _tmpfile:
        util_gpu = list(map(int, re.findall(r'Gpu\s+:\s*(\d+)\s*%', _tmpfile.read())))
    os.remove('gpu')
    if len(util_gpu):
        device = util_gpu.index(min(util_gpu))
    else:
        device = 0
else:
    device = -1
# device=0
# print('device: ', device)

if multi_gpu:
    devices = '0,1'
    devices = list(map(int, devices.split(',')))
    device = devices[0]
    torch.cuda.set_device(device)
    os.environ['CUDA_VISIBLE_DIVICES'] = ','.join(map(str, devices))
else:
    devices = str(device)
    torch.cuda.set_device(device)

# ===Save Model and samples===
save_root = 'save/{}/{}/{}_{}_dt-{}_lt-{}_mt-{}_et-{}_sl{}_temp{}_lfd{}_T{}/'.format(time.strftime("%Y%m%d"),
                                                                                     dataset, run_model, model_type,
                                                                                     d_type,
                                                                                     loss_type,
                                                                                     '+'.join(
                                                                                         [m[:2] for m in
                                                                                          mu_type.split()]),
                                                                                     eval_type, max_seq_len,
                                                                                     temperature, lambda_fd,
                                                                                     log_time_str)
save_samples_root = save_root + 'samples/'
save_model_root = save_root + 'models/'

oracle_state_dict_path = 'pretrain/oracle_data/oracle_lstm.pt'
oracle_samples_path = 'pretrain/oracle_data/oracle_lstm_samples_{}.pt'
multi_oracle_state_dict_path = 'pretrain/oracle_data/oracle{}_lstm.pt'
multi_oracle_samples_path = 'pretrain/oracle_data/oracle{}_lstm_samples_{}.pt'

pretrain_root = 'pretrain/{}/'.format(dataset if if_real_data else 'oracle_data')
pretrained_gen_path = pretrain_root + 'gen_MLE_pretrain_{}_{}_sl{}_sn{}.pt'.format(run_model, model_type, max_seq_len,
                                                                                   samples_num)
pretrained_dis_path = pretrain_root + 'dis_pretrain_{}_{}_sl{}_sn{}.pt'.format(run_model, model_type, max_seq_len,
                                                                               samples_num)
pretrained_clas_path = pretrain_root + 'clas_pretrain_{}_{}_sl{}_sn{}.pt'.format(run_model, model_type, max_seq_len,
                                                                                 samples_num)
signal_file = 'run_signal.txt'

tips = ''

if samples_num == 5000 or samples_num == 2000:
    assert 'c' in run_model, 'warning: samples_num={}, run_model={}'.format(samples_num, run_model)


# Init settings according to parser
def init_param(opt):
    global run_model, model_type, loss_type, CUDA, device, data_shuffle, samples_num, vocab_size, \
        MLE_train_epoch, ADV_train_epoch, inter_epoch, batch_size, max_seq_len, start_letter, padding_idx, \
        gen_lr, gen_adv_lr, dis_lr, clip_norm, pre_log_step, adv_log_step, train_data, test_data, temp_adpt, \
        temperature, oracle_pretrain, gen_pretrain, dis_pretrain, ADV_g_step, rollout_num, gen_embed_dim, \
        gen_hidden_dim, goal_size, step_size, mem_slots, num_heads, head_size, d_step, d_epoch, \
        ADV_d_step, ADV_d_epoch, dis_embed_dim, dis_hidden_dim, num_rep, log_filename, save_root, \
        signal_file, tips, save_samples_root, save_model_root, if_real_data, pretrained_gen_path, \
        pretrained_dis_path, pretrain_root, if_test, dataset, PRE_clas_epoch, oracle_samples_path, \
        pretrained_clas_path, n_parent, mu_type, eval_type, d_type, eval_b_num, lambda_fd, d_out_mean, \
        lambda_fq, freeze_dis, freeze_clas, use_all_real_fake, use_population, gen_init, dis_init, \
        multi_oracle_samples_path, k_label, cat_train_data, cat_test_data, evo_temp_step, devices, \
        use_nll_oracle, use_nll_gen, use_nll_div, use_bleu, use_self_bleu, use_clas_acc, use_ppl

    if_test = True if opt.if_test == 1 else False
    run_model = opt.run_model
    k_label = opt.k_label
    dataset = opt.dataset
    model_type = opt.model_type
    loss_type = opt.loss_type
    mu_type = opt.mu_type
    eval_type = opt.eval_type
    d_type = opt.d_type
    if_real_data = True if opt.if_real_data == 1 else False
    CUDA = True if opt.cuda == 1 else False
    device = opt.device
    devices = opt.devices
    data_shuffle = opt.shuffle
    gen_init = opt.gen_init
    dis_init = opt.dis_init

    n_parent = opt.n_parent
    eval_b_num = opt.eval_b_num
    lambda_fq = opt.lambda_fq
    lambda_fd = opt.lambda_fd
    d_out_mean = opt.d_out_mean
    freeze_dis = opt.freeze_dis
    freeze_clas = opt.freeze_clas
    use_all_real_fake = opt.use_all_real_fake
    use_population = opt.use_population

    samples_num = opt.samples_num
    vocab_size = opt.vocab_size
    MLE_train_epoch = opt.mle_epoch
    PRE_clas_epoch = opt.clas_pre_epoch
    ADV_train_epoch = opt.adv_epoch
    inter_epoch = opt.inter_epoch
    batch_size = opt.batch_size
    max_seq_len = opt.max_seq_len
    start_letter = opt.start_letter
    padding_idx = opt.padding_idx
    gen_lr = opt.gen_lr
    gen_adv_lr = opt.gen_adv_lr
    dis_lr = opt.dis_lr
    clip_norm = opt.clip_norm
    pre_log_step = opt.pre_log_step
    adv_log_step = opt.adv_log_step
    temp_adpt = opt.temp_adpt
    evo_temp_step = opt.evo_temp_step
    temperature = opt.temperature
    oracle_pretrain = True if opt.ora_pretrain == 1 else False
    gen_pretrain = True if opt.gen_pretrain == 1 else False
    dis_pretrain = True if opt.dis_pretrain == 1 else False

    ADV_g_step = opt.adv_g_step
    rollout_num = opt.rollout_num
    gen_embed_dim = opt.gen_embed_dim
    gen_hidden_dim = opt.gen_hidden_dim
    goal_size = opt.goal_size
    step_size = opt.step_size
    mem_slots = opt.mem_slots
    num_heads = opt.num_heads
    head_size = opt.head_size

    d_step = opt.d_step
    d_epoch = opt.d_epoch
    ADV_d_step = opt.adv_d_step
    ADV_d_epoch = opt.adv_d_epoch
    dis_embed_dim = opt.dis_embed_dim
    dis_hidden_dim = opt.dis_hidden_dim
    num_rep = opt.num_rep

    use_nll_oracle = True if opt.use_nll_oracle == 1 else False
    use_nll_gen = True if opt.use_nll_gen == 1 else False
    use_nll_div = True if opt.use_nll_div == 1 else False
    use_bleu = True if opt.use_bleu == 1 else False
    use_self_bleu = True if opt.use_self_bleu == 1 else False
    use_clas_acc = True if opt.use_clas_acc == 1 else False
    use_ppl = True if opt.use_ppl == 1 else False

    log_filename = opt.log_file
    signal_file = opt.signal_file
    tips = opt.tips

    # CUDA device
    if multi_gpu:
        if type(devices) == str:
            devices = list(map(int, devices.split(',')))
        device = devices[0]
        torch.cuda.set_device(device)
        os.environ['CUDA_VISIBLE_DIVICES'] = ','.join(map(str, devices))
    else:
        devices = str(device)
        torch.cuda.set_device(device)

    # Save path
    save_root = 'save/{}/{}/{}_{}_dt-{}_lt-{}_mt-{}_et-{}_sl{}_temp{}_lfd{}_T{}/'.format(time.strftime("%Y%m%d"),
                                                                                         dataset, run_model, model_type,
                                                                                         d_type,
                                                                                         loss_type,
                                                                                         '+'.join(
                                                                                             [m[:2] for m in
                                                                                              mu_type.split()]),
                                                                                         eval_type, max_seq_len,
                                                                                         temperature, lambda_fd,
                                                                                         log_time_str)

    save_samples_root = save_root + 'samples/'
    save_model_root = save_root + 'models/'

    train_data = 'dataset/' + dataset + '.txt'
    test_data = 'dataset/testdata/' + dataset + '_test.txt'
    cat_train_data = 'dataset/' + dataset + '_cat{}.txt'
    cat_test_data = 'dataset/testdata/' + dataset + '_cat{}_test.txt'

    if max_seq_len == 40:
        oracle_samples_path = 'pretrain/oracle_data/oracle_lstm_samples_{}_sl40.pt'
        multi_oracle_samples_path = 'pretrain/oracle_data/oracle{}_lstm_samples_{}_sl40.pt'

    pretrain_root = 'pretrain/{}/'.format(dataset if if_real_data else 'oracle_data')
    pretrained_gen_path = pretrain_root + 'gen_MLE_pretrain_{}_{}_sl{}_sn{}.pt'.format(run_model, model_type,
                                                                                       max_seq_len, samples_num)
    pretrained_dis_path = pretrain_root + 'dis_pretrain_{}_{}_sl{}_sn{}.pt'.format(run_model, model_type, max_seq_len,
                                                                                   samples_num)
    pretrained_clas_path = pretrain_root + 'clas_pretrain_{}_{}_sl{}_sn{}.pt'.format(run_model, model_type, max_seq_len,
                                                                                     samples_num)

    # Assertion
    assert k_label >= 2, 'Error: k_label = {}, which should be >=2!'.format(k_label)
    assert eval_b_num >= n_parent * ADV_d_step, 'Error: eval_b_num = {}, which should be >= n_parent * ADV_d_step ({})!'.format(
        eval_b_num, n_parent * ADV_d_step)

    # Create Directory
    dir_list = ['save', 'savefig', 'log', 'pretrain', 'dataset',
                'pretrain/{}'.format(dataset if if_real_data else 'oracle_data')]
    if not if_test:
        dir_list.extend([save_root, save_samples_root, save_model_root])
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)
