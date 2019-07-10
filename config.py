# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : config.py
# @Time         : Created at 2019-03-18
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.

from time import strftime, localtime

import os
import re
import torch

# =====Program=====
if_test = False
CUDA = True
if_save = True
data_shuffle = False  # False
oracle_pretrain = True  # True
gen_pretrain = False
dis_pretrain = False

run_model = 'relgan'  # seqgan, leakgan, relgan
use_truncated_normal = True

# =====Oracle or Real, type=====
if_real_data = False  # if use real data
dataset = 'oracle'  # oracle, image_coco, emnlp_news
model_type = 'vanilla'  # vanilla, noRMC, noGumbel (custom)
loss_type = 'RSGAN'  # standard, JS, KL, hinge, tv, LS, RSGAN (for RelGAN)
vocab_size = 5000  # oracle: 5000, coco: 6613, emnlp: 5255

temp_adpt = 'exp'  # no, lin, exp, log, sigmoid, quad, sqrt (for RelGAN)
temperature = 2

# =====Basic Train=====
samples_num = 10000  # 10000
MLE_train_epoch = 150  # SeqGAN-80, LeakGAN-8, RelGAN-150
ADV_train_epoch = 3000  # SeqGAN, LeakGAN-200, RelGAN-3000
inter_epoch = 10  # LeakGAN-10
batch_size = 64  # 64
max_seq_len = 20  # 20
start_letter = 1
padding_idx = 0
start_token = 'BOS'
padding_token = 'EOS'
gen_lr = 0.01  # 0.01
gen_adv_lr = 1e-4  # RelGAN-1e-4
dis_lr = 1e-4  # SeqGAN,LeakGAN-1e-2, RelGAN-1e-4
clip_norm = 5.0

pre_log_step = 10
adv_log_step = 20

train_data = 'dataset/' + dataset + '.txt'
test_data = 'dataset/testdata/' + dataset + '_test.txt'

# =====Generator=====
ADV_g_step = 1  # 1
rollout_num = 4  # 4
gen_embed_dim = 32  # 32
gen_hidden_dim = 32  # 32
goal_size = 16  # LeakGAN-16
step_size = 4  # LeakGAN-4

mem_slots = 1  # RelGAN-1
num_heads = 2  # RelGAN-2
head_size = 256  # RelGAN-256

# =====Discriminator=====
d_step = 5  # SeqGAN-50, LeakGAN-5
d_epoch = 3  # SeqGAN,LeakGAN-3
ADV_d_step = 5  # SeqGAN,LeakGAN,RelGAN-5
ADV_d_epoch = 3  # SeqGAN,LeakGAN-3

dis_embed_dim = 64
dis_hidden_dim = 64
num_rep = 64  # RelGAN

# =====log=====
log_filename = strftime("log/log_%m%d_%H%M", localtime())
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
# device=1
# print('device: ', device)
torch.cuda.set_device(device)

# =====Save Model and samples=====
save_root = 'save/{}_{}_{}_{}_glr{}_temp{}_T{}/'.format(run_model, model_type, dataset, loss_type, gen_lr,
                                                        temperature, strftime("%m%d-%H%M", localtime()))
save_samples_root = save_root + 'samples/'
save_model_root = save_root + 'models/'

oracle_state_dict_path = 'pretrain/oracle_data/oracle_lstm.pt'
oracle_samples_path = 'pretrain/oracle_data/oracle_lstm_samples_{}.pt'

pretrain_root = 'pretrain/{}/'.format('real_data' if if_real_data else 'oracle_data')
pretrained_gen_path = pretrain_root + 'gen_MLE_pretrain_{}_{}.pt'.format(run_model, model_type)
pretrained_dis_path = pretrain_root + 'dis_pretrain_{}_{}.pt'.format(run_model, model_type)
signal_file = 'run_signal.txt'

tips = ''


# Init settings according to parser
def init_param(opt):
    global run_model, model_type, loss_type, CUDA, device, data_shuffle, samples_num, vocab_size, \
        MLE_train_epoch, ADV_train_epoch, inter_epoch, batch_size, max_seq_len, start_letter, padding_idx, \
        gen_lr, gen_adv_lr, dis_lr, clip_norm, pre_log_step, adv_log_step, train_data, test_data, temp_adpt, \
        temperature, oracle_pretrain, gen_pretrain, dis_pretrain, ADV_g_step, rollout_num, gen_embed_dim, \
        gen_hidden_dim, goal_size, step_size, mem_slots, num_heads, head_size, d_step, d_epoch, \
        ADV_d_step, ADV_d_epoch, dis_embed_dim, dis_hidden_dim, num_rep, log_filename, save_root, \
        signal_file, tips, save_samples_root, save_model_root, if_real_data, pretrained_gen_path, \
        pretrained_dis_path, pretrain_root, if_test, use_truncated_normal, dataset

    if_test = True if opt.if_test == 1 else False
    run_model = opt.run_model
    dataset = opt.dataset
    model_type = opt.model_type
    loss_type = opt.loss_type
    if_real_data = True if opt.if_real_data == 1 else False
    CUDA = True if opt.cuda == 1 else False
    device = opt.device
    data_shuffle = opt.shuffle
    use_truncated_normal = True if opt.use_truncated_normal == 1 else False

    samples_num = opt.samples_num
    vocab_size = opt.vocab_size
    MLE_train_epoch = opt.mle_epoch
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
    train_data = opt.train_data
    test_data = opt.test_data
    temp_adpt = opt.temp_adpt
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

    log_filename = opt.log_file
    signal_file = opt.signal_file
    tips = opt.tips

    # CUDA device
    torch.cuda.set_device(device)

    # Save path
    save_root = 'save/{}_{}_{}_{}_glr{}_temp{}_T{}/'.format(run_model, model_type, dataset, loss_type, gen_lr,
                                                            temperature, strftime("%m%d-%H%M", localtime()))
    save_samples_root = save_root + 'samples/'
    save_model_root = save_root + 'models/'

    train_data = 'dataset/' + dataset + '.txt'
    test_data = 'dataset/testdata/' + dataset + '_test.txt'

    pretrain_root = 'pretrain/{}/'.format('real_data' if if_real_data else 'oracle_data')
    pretrained_gen_path = pretrain_root + 'gen_MLE_pretrain_{}_{}.pt'.format(run_model, model_type)
    pretrained_dis_path = pretrain_root + 'dis_pretrain_{}_{}.pt'.format(run_model, model_type)

    # Create Directory
    dir_list = ['save', 'savefig', 'log', 'pretrain', 'save/log', 'dataset',
                'pretrain/oracle_data', 'pretrain/real_data']
    if not if_test:
        dir_list.extend([save_root, save_samples_root, save_model_root])
    for d in dir_list:
        if not os.path.exists(d):
            os.mkdir(d)
