from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb
from tqdm import tqdm
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import generator
import discriminator
import helpers
import rollout

import config as cfg


def _print(content):
    print(content, end='')
    sys.stdout.flush()
    if not cfg.if_test:
        cfg.log.write(content)


def train_generator_MLE(gen, gen_opt, oracle, real_data_samples, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    for epoch in range(epochs):
        _print('epoch %d : ' % (epoch + 1))
        total_loss = 0

        for i in range(0, cfg.samples_num, cfg.batch_size):
            inp, target = helpers.prepare_generator_batch(real_data_samples[i:i + cfg.batch_size],
                                                          start_letter=cfg.start_letter,
                                                          gpu=cfg.CUDA)
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()

            if (i / cfg.batch_size) % ceil(
                    ceil(cfg.samples_num / float(cfg.batch_size)) / 10.) == 0:  # roughly every 10% of an epoch
                _print('.')

        # each loss in a batch is loss per sample
        total_loss = total_loss / ceil(cfg.samples_num / float(cfg.batch_size)) / cfg.max_seq_len

        # sample from generator and compute oracle NLL
        oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, cfg.samples_num, cfg.batch_size, cfg.max_seq_len,
                                                   start_letter=cfg.start_letter, gpu=cfg.CUDA)

        _print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f\n' % (total_loss, oracle_loss))

        res_dict['gen_mle_nll'].append(total_loss)
        res_dict['gen_oracle_nll'].append(oracle_loss)


def train_generator_PG(gen, gen_opt, oracle, dis, g_step, current_k):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """

    rollout_func = rollout.ROLLOUT(gen, cfg.CUDA)
    total_pg_loss = 0
    for step in range(g_step):
        s = gen.sample(cfg.batch_size * 2)  # 64 works best
        # input is same as target, but with start_letter prepended
        inp, target = helpers.prepare_generator_batch(s, start_letter=cfg.start_letter, gpu=cfg.CUDA)

        if cfg.seq_update:
            '''PG loss for total sequence'''
            rewards = rollout_func.get_reward(target, cfg.rollout_num, dis, current_k)  # reward with MC search

            gen_opt.zero_grad()
            pg_loss = gen.batchPGLoss(inp, target, rewards)
            total_pg_loss += pg_loss.data.item()
            pg_loss.backward()
            gen_opt.step()
        else:
            '''PG loss for each token, update generator in each step with PG'''
            for idx in range(1, cfg.max_seq_len + 1):
                rewards = rollout_func.get_token_reward(target, cfg.rollout_num, dis, current_k, idx)

                gen_opt.zero_grad()
                pg_loss = gen.batchPGLoss(inp, target, rewards)
                total_pg_loss += pg_loss.data.item()
                pg_loss.backward()
                gen_opt.step()

    # sample from generator and compute oracle NLL
    oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, cfg.samples_num, cfg.batch_size, cfg.max_seq_len,
                                               start_letter=cfg.start_letter, gpu=cfg.CUDA)

    _print(' oracle_sample_NLL = %.4f' % oracle_loss)
    _print(' PG_LOSS = %.4f\n' % (total_pg_loss / g_step))

    res_dict['gen_oracle_nll'].append(oracle_loss)
    res_dict['gen_pg_loss'].append(total_pg_loss / g_step)


def train_discriminator(dis, dis_opt, gen_list, oracle_list, oracle_samples_list, d_steps, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    # generating a small validation set before training (using oracle and generator)
    pos_val = []
    neg_val = []
    for i in range(cfg.k_label):
        pos_val.append(oracle_list[i].sample(100))
        neg_val.append(gen_list[i].sample(100))

    val_inp, val_target = helpers.prepare_senti_discriminator_data(pos_val, neg_val, cfg.k_label, gpu=cfg.CUDA)

    # loss_fn = nn.BCELoss()
    loss_fn = nn.CrossEntropyLoss()
    for d_step in range(d_steps):
        gen_samples_list = []
        for i in range(cfg.k_label):
            gen_samples_list.append(gen_list[i].sample(cfg.samples_num))

        dis_inp, dis_target = helpers.prepare_senti_discriminator_data(oracle_samples_list, gen_samples_list,
                                                                       cfg.k_label, gpu=cfg.CUDA)
        for epoch in range(epochs):
            _print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1))
            total_loss = 0
            total_acc = 0
            train_size = 2 * cfg.k_label * cfg.samples_num
            global_step = 0
            for i in range(0, train_size, cfg.batch_size):
                inp, target = dis_inp[i:i + cfg.batch_size], dis_target[i:i + cfg.batch_size]

                dis_opt.zero_grad()
                out = dis.batchClassify(inp)

                # loss = loss_fn(out, target.float())
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                # if global_step % 10 == 0:
                #     print('out:', F.softmax(out))
                #     print('target:', target)
                #     # print('acc_num:', torch.sum((out > 0.5) == (target > 0.5)).data.item())
                #     print('acc_num:', torch.sum((out.argmax(dim=-1) == target)).data.item())
                total_loss += loss.data.item()
                total_acc += torch.sum((out.argmax(dim=-1) == target)).data.item()
                # total_acc += torch.sum((out > 0.5) == (target > 0.5)).data.item()

                if (i / cfg.batch_size) % ceil(ceil(train_size / float(
                        cfg.batch_size)) / 10.) == 0:  # roughly every 10% of an epoch
                    _print('.')

            total_loss /= ceil(train_size / float(cfg.batch_size))
            total_acc /= float(train_size)

            val_pred = dis.batchClassify(val_inp)
            val_size = 2 * cfg.k_label * 100
            val_acc = torch.sum((val_pred.argmax(dim=-1) == val_target)).data.item() / val_size
            _print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f\n' % (
                total_loss, total_acc, val_acc))

            res_dict['dis_train_loss'].append(total_loss)
            res_dict['dis_train_acc'].append(total_acc)
            res_dict['dis_val_acc'].append(val_acc)


def tmp_test():
    print('k_label:', cfg.k_label)
    print('ADV_train_epoch:', cfg.ADV_train_epoch)
    print('samples_num:', cfg.samples_num)
    print('tips:', cfg.tips)


# MAIN
if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--mle_epoch', default=cfg.MLE_train_epoch, type=int)
    parser.add_argument('--adv_epoch', default=cfg.ADV_train_epoch, type=int)
    parser.add_argument('--k_label', default=cfg.k_label, type=int)
    parser.add_argument('--samples_num', default=cfg.samples_num, type=int)
    parser.add_argument('--batch_size', default=cfg.batch_size, type=int)
    parser.add_argument('--adv_g_step', default=cfg.ADV_g_step, type=int)
    parser.add_argument('--rollout_num', default=cfg.rollout_num, type=int)
    parser.add_argument('--d_step', default=cfg.d_step, type=int)
    parser.add_argument('--d_epoch', default=cfg.d_epoch, type=int)
    parser.add_argument('--adv_d_step', default=cfg.ADV_d_step, type=int)
    parser.add_argument('--adv_d_epoch', default=cfg.ADV_d_epoch, type=int)

    parser.add_argument('--vocab_size', default=cfg.vocab_size, type=int)
    parser.add_argument('--max_seq_len', default=cfg.max_seq_len, type=int)
    parser.add_argument('--start_letter', default=cfg.start_letter, type=int)
    parser.add_argument('--gen_embed_dim', default=cfg.gen_embed_dim, type=int)
    parser.add_argument('--gen_hidden_dim', default=cfg.gen_hidden_dim, type=int)
    parser.add_argument('--dis_embed_dim', default=cfg.dis_embed_dim, type=int)
    parser.add_argument('--dis_hidden_dim', default=cfg.dis_hidden_dim, type=int)

    parser.add_argument('--cuda', default=cfg.CUDA, type=int)
    parser.add_argument('--device', default=cfg.device, type=int)
    parser.add_argument('--if_save', default=cfg.if_save, type=int)
    parser.add_argument('--if_test', default=cfg.if_test, type=int)
    parser.add_argument('--if_reward', default=cfg.if_reward, type=int)
    parser.add_argument('--gen_pretrain', default=cfg.gen_pretrain, type=int)
    parser.add_argument('--dis_pretrain', default=cfg.dis_pretrain, type=int)
    parser.add_argument('--seq_update', default=cfg.seq_update, type=int)
    parser.add_argument('--no_log', default=cfg.no_log, type=int)
    parser.add_argument('--log_file', default=cfg.log_filename, type=str)
    parser.add_argument('--tips', default=cfg.tips, type=str)
    opt = parser.parse_args()
    cfg.init_param(opt)

    _print(100 * '=' + '\n')
    _print('> training arguments:\n')
    for arg in vars(opt):
        _print('>>> {0}: {1}\n'.format(arg, getattr(opt, arg)))
    _print(100 * '=' + '\n')

    # oracle = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    # oracle.load_state_dict(torch.load(oracle_state_dict_path))
    # oracle_samples = torch.load(oracle_samples_path).type(torch.LongTensor)
    # a new oracle can be generated by passing oracle_init=True in the generator constructor
    # samples for the new oracle can be generated using helpers.batchwise_sample()

    # =====Experiment data dict=====
    res_dict = {}
    key_name = ['gen_oracle_nll', 'gen_mle_nll', 'gen_pg_loss',
                'dis_train_loss', 'dis_train_acc', 'dis_val_acc']
    for k in key_name:
        res_dict[k] = []

    # =====Load Oracle=====
    oracle_list = []
    oracle_samples_list = []
    gen_list = []
    for i in range(1, cfg.k_label + 1):
        # oracle
        oracle = generator.Generator(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                                     gpu=cfg.CUDA)
        oracle.load_state_dict(torch.load(cfg.oracle_state_dict_path.format(i)))
        oracle_list.append(oracle)
        oracle_samples_list.append(torch.load(cfg.oracle_samples_path.format(i, cfg.samples_num)))
        # generator
        gen_list.append(generator.Generator(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                                            oracle_init=True, gpu=cfg.CUDA))

    dis = discriminator.Discriminator(cfg.dis_embed_dim, cfg.vocab_size, cfg.dis_filter_sizes, cfg.dis_num_filters,
                                      cfg.k_label, gpu=cfg.CUDA)
    # dis = discriminator.GRU_Discriminator(cfg.dis_embed_dim, cfg.dis_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
    #                                   cfg.k_label, gpu=cfg.CUDA)

    if cfg.CUDA:
        for i in range(cfg.k_label):
            oracle_list[i] = oracle_list[i].cuda()
            oracle_samples_list[i] = oracle_samples_list[i].cuda()
            gen_list[i] = gen_list[i].cuda()

        dis = dis.cuda()

    # =================test=================
    if cfg.if_test:
        print('Begin test...')
        tmp_test()
        print('End test...')
        exit(0)
    # =================end test=================

    # ==========GENERATOR MLE TRAINING==========
    _print('\nStarting Generator MLE Training...\n')
    gen_optimizer_list = []
    for i in range(cfg.k_label):
        if cfg.gen_pretrain:
            _print('Load MLE pretrain generator: {}\n'.format(cfg.pretrained_gen_path.format(i)))
            gen_list[i].load_state_dict(torch.load(cfg.pretrained_gen_path.format(i)))
            if cfg.no_log:
                gen_optimizer = optim.RMSprop(gen_list[i].parameters(), lr=1e-2)
            else:
                gen_optimizer = optim.Adam(gen_list[i].parameters(), lr=1e-2)
            gen_optimizer_list.append(gen_optimizer)
        else:
            _print('Generator {} MLE training...\n'.format(i + 1))
            gen_optimizer = optim.Adam(gen_list[i].parameters(), lr=1e-2)
            gen_optimizer_list.append(gen_optimizer)

            train_generator_MLE(gen_list[i], gen_optimizer, oracle_list[i], oracle_samples_list[i], cfg.MLE_train_epoch)

            if cfg.if_save:
                torch.save(gen_list[i].state_dict(), cfg.pretrained_gen_path.format(i))
                print('Save MLE pretrain generator: {}\n'.format(cfg.pretrained_gen_path.format(i)))

    # ==========PRETRAIN DISCRIMINATOR==========
    _print('\nStarting Discriminator Training...\n')
    dis_optimizer = optim.Adam(dis.parameters())
    if cfg.dis_pretrain:
        _print('Load pretrain discriminator: {}\n'.format(cfg.pretrained_dis_path.format(cfg.k_label)))
        dis.load_state_dict(torch.load(cfg.pretrained_dis_path.format(cfg.k_label)))
    else:
        train_discriminator(dis, dis_optimizer, gen_list, oracle_list, oracle_samples_list, cfg.d_step, cfg.d_epoch)

        if cfg.if_save:
            torch.save(dis.state_dict(), cfg.pretrained_dis_path.format(cfg.k_label))
            print('Save pretrain discriminator: {}\n'.format(cfg.pretrained_dis_path.format(cfg.k_label)))

    # ==========ADVERSARIAL TRAINING==========
    _print('\nStarting Adversarial Training...\n')
    for i in range(cfg.k_label):
        oracle_loss = helpers.batchwise_oracle_nll(gen_list[i], oracle_list[i], cfg.samples_num, cfg.batch_size,
                                                   cfg.max_seq_len, start_letter=cfg.start_letter, gpu=cfg.CUDA)
        _print('Generator %d: Initial Oracle Sample Loss : %.4f\n' % (i + 1, oracle_loss))

    for epoch in range(cfg.ADV_train_epoch):
        _print('\n--------\nEPOCH %d\n--------\n' % (epoch + 1))
        # TRAIN GENERATOR
        for i in range(cfg.k_label):
            _print('\nAdversarial Training Generator %d : \n' % (i + 1))
            train_generator_PG(gen_list[i], gen_optimizer_list[i], oracle_list[i], dis, cfg.ADV_g_step, i)

        # TRAIN DISCRIMINATOR
        _print('\nAdversarial Training Discriminator : \n')
        train_discriminator(dis, dis_optimizer, gen_list, oracle_list, oracle_samples_list, cfg.ADV_d_step,
                            cfg.ADV_d_epoch)

    # ==========Save and close==========
    cfg.dict_file.write(str(res_dict))
    # close log file
    try:
        cfg.log.close()
        cfg.dict_file.close()
    except:
        pass
