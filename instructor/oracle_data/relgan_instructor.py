# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : relgan_instructor.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import time

from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from utils.data_utils import GenDataIter
from instructor.oracle_data.instructor import BasicInstructor
from models.RelGAN_G import RelGAN_G
from models.RelGAN_D import RelGAN_D
from utils.helpers import get_fixed_temperature


class RelGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(RelGANInstructor, self).__init__(opt)

        # generator, discriminator
        self.gen = RelGAN_G(cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim, cfg.gen_hidden_dim,
                            cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, cfg.temperature, gpu=cfg.CUDA)
        self.dis = RelGAN_D(cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.vocab_size, cfg.dis_filter_sizes,
                            cfg.dis_num_filters, cfg.padding_idx, gpu=cfg.CUDA)

        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_adv_lr)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)

        # Criterion
        self.mle_criterion = nn.NLLLoss()
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.dis_criterion = nn.CrossEntropyLoss()  # For SeqGAN CNN Discriminator

        # DataLoader
        self.gen_data = GenDataIter(self.gen.sample(cfg.batch_size, cfg.batch_size))

    def _run(self):
        # ==========PRE-TRAINING (GENERATOR)==========
        if not cfg.gen_pretrain:
            self._print('\nStarting Generator MLE Training...\n')
            self.pretrain_generator(cfg.MLE_train_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
                print('Save pretrain_generator discriminator: {}\n'.format(cfg.pretrained_dis_path))

        oracle_nll, gen_nll = self.cal_metrics()
        self._print('Initial generator: oracle_NLL = %.4f, gen_NLL = %.4f\n' % (oracle_nll, gen_nll))

        # # ==========ADVERSARIAL TRAINING==========
        self._print('\nStarting Adversarial Training...\n')
        progress = tqdm(range(cfg.ADV_train_epoch))
        for adv_epoch in progress:
            self.sig.update()
            if self.sig.adv_sig:
                g_loss = self.adv_train_generator(cfg.ADV_g_step)  # Generator
                d_loss = self.adv_train_discriminator(cfg.ADV_d_step)  # Discriminator
                self.update_temperature(adv_epoch, cfg.ADV_train_epoch)  # update temperature

                progress.set_description('g_loss: %.4f, d_loss: %.4f,' % (g_loss, d_loss))

                # TEST
                if adv_epoch % cfg.adv_log_step == 0:
                    oracle_nll, gen_nll = self.cal_metrics()
                    self._print(
                        '[ADV] epoch %d: g_loss: %.4f, d_loss: %.4f, oracle_NLL = %.4f, gen_NLL = %.4f,\n' % (
                            adv_epoch, g_loss, d_loss, oracle_nll, gen_nll))

                    if cfg.if_save and not cfg.if_test:
                        self._save('ADV', adv_epoch)
            else:
                self._print('\n>>> Stop by adv_signal! Finishing adversarial training...\n')
                progress.close()
                break

    def _test(self):
        print('>>> Begin test...')

        # t0 = time.time()
        # gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True).cuda()
        # t1 = time.time()
        # print('sample time: ', t1 - t0)
        #
        # t0 = time.time()
        # d_out_fake = self.dis(gen_samples)
        # t1 = time.time()
        # print('dis time: ', t1 - t0)
        #
        # t0 = time.time()
        # self.adv_train_generator(1)
        # t1 = time.time()
        # print('adv gen time: ', t1 - t0)
        #
        # t0 = time.time()
        # self.adv_train_discriminator(5)
        # t1 = time.time()
        # print('adv dis time: ', t1 - t0)
        # self._run()

        # oracle_nll, gen_nll = self.cal_metrics()
        # print(oracle_nll,gen_nll)
        # self.adv_train_discriminator(1)

        # self.gen.load_state_dict(torch.load(cfg.pretrained_gen_path))
        #
        # oracle_nll, gen_nll = self.cal_metrics()
        # self._print('Initial generator: oracle_NLL = %.4f, gen_NLL = %.4f\n' % (oracle_nll, gen_nll))
        #
        # real_samples = self.oracle_data.randam_batch()['target']
        # gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size)
        #
        # print(real_samples)
        # print(gen_samples)

        t0 = time.time()
        # for _ in range(10):
        #     self.gen.sample(cfg.batch_size, cfg.batch_size)
        # self.adv_train_generator(10)
        # self.adv_train_discriminator(10)
        # for _ in range(10):
        while True:
            t0 = time.time()

            # real_samples = self.oracle_data.randam_batch()['target']
            gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True)
            # if cfg.CUDA:
            #     real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            # real_samples = F.one_hot(real_samples, cfg.vocab_size).float()
            #
            # # =====Train=====
            # d_out_real = self.dis(real_samples)
            # d_out_fake = self.dis(gen_samples)
            # # loss = torch.sum(d_out_fake-d_out_real)
            # loss = torch.sum(d_out_fake)
            #
            # t0 = time.time()
            # self.optimize(self.gen_opt, loss)
            # t1 = time.time()
            # print('opt time: ', t1 - t0)
            t1 = time.time()
            print('time: ', t1 - t0)
        t1 = time.time()
        # print((t1 - t0) / 10)

        # self._run()
        pass

    def pretrain_generator(self, epochs):
        """
        Max Likelihood Pre-training for the generator
        """
        t0 = time.time()
        for epoch in range(epochs):
            self.sig.update()
            if self.sig.pre_sig:
                # =====Train=====
                pre_loss = self.train_gen_epoch(self.gen, self.oracle_data.loader, self.mle_criterion, self.gen_opt)

                # =====Test=====
                if epoch % cfg.pre_log_step == 0:
                    oracle_nll, gen_nll = self.cal_metrics()
                    t1 = time.time()
                    self._print(
                        '[MLE-GEN] epoch %d : pre_loss = %.4f, oracle_NLL = %.4f, gen_NLL = %.4f, time = %.4f\n' % (
                            epoch, pre_loss, oracle_nll, gen_nll, t1 - t0))
                    t0 = time.time()

                    if cfg.if_save and not cfg.if_test:
                        self._save('MLE', epoch)
            else:
                self._print('\n>>> Stop by pre signal, skip to adversarial training...')
                break

    def adv_train_generator(self, g_step):
        total_loss = 0
        for step in range(g_step):
            real_samples = self.oracle_data.randam_batch()['target']
            gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True)
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            real_samples = F.one_hot(real_samples, cfg.vocab_size).float()

            # =====Train=====
            d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
            g_loss = self.adv_criterion(d_out_fake - d_out_real, torch.ones_like(d_out_fake))

            self.optimize(self.gen_adv_opt, g_loss, self.gen)
            total_loss += g_loss.item()

        return total_loss / g_step if g_step != 0 else 0

    def adv_train_discriminator(self, d_step):
        total_loss = 0
        for step in range(d_step):
            real_samples = self.oracle_data.randam_batch()['target']
            gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True)
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            real_samples = F.one_hot(real_samples, cfg.vocab_size).float()

            # =====Train=====
            d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
            d_loss = self.adv_criterion(d_out_real - d_out_fake, torch.ones_like(d_out_real))

            self.optimize(self.dis_opt, d_loss, self.dis)
            total_loss += d_loss.item()

        return total_loss / d_step if d_step != 0 else 0

    def update_temperature(self, i, N):
        self.gen.temperature = get_fixed_temperature(cfg.temperature, i, N, cfg.temp_adpt)
