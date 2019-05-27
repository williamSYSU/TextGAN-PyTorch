# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : leakgan_instructor.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.
import time

import torch
import torch.nn as nn
import torch.optim as optim

import config as cfg
from instructor.oracle_data.instructor import BasicInstructor
from models.LeakGAN_D import LeakGAN_D
from models.LeakGAN_G import LeakGAN_G
from utils import rollout
from utils.data_utils import GenDataIter, DisDataIter


class LeakGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(LeakGANInstructor, self).__init__(opt)

        # generator, discriminator
        self.gen = LeakGAN_G(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                             cfg.padding_idx, cfg.goal_size, cfg.goal_out_size, cfg.step_size, cfg.CUDA)
        self.dis = LeakGAN_D(cfg.dis_embed_dim, cfg.vocab_size, cfg.dis_filter_sizes, cfg.dis_num_filters,
                             cfg.padding_idx, gpu=cfg.CUDA)
        self.init_model()

        # optimizer
        mana_params, work_params = self.gen.split_params()
        mana_opt = optim.Adam(mana_params, lr=cfg.gen_lr)
        work_opt = optim.Adam(work_params, lr=cfg.gen_lr)

        self.gen_opt = [mana_opt, work_opt]
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)

        # Criterion
        self.mle_criterion = nn.NLLLoss()
        self.dis_criterion = nn.CrossEntropyLoss()

        # DataLoader
        self.gen_data = GenDataIter(self.gen.sample(cfg.batch_size, cfg.batch_size, self.dis))

    def _run(self):
        for inter_num in range(cfg.inter_epoch):
            self._print('\n>>> Interleaved Round %d...\n' % inter_num)
            self.sig.update()  # update signal
            if self.sig.pre_sig:
                # ==========DISCRIMINATOR PRE-TRAINING==========
                self._print('\nStarting Discriminator Training...\n')
                if not cfg.dis_pretrain:
                    self.train_discriminator(cfg.d_step, cfg.d_epoch)
                    if cfg.if_save:
                        torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)
                        print('Save pretrain_generator discriminator: {}\n'.format(cfg.pretrained_dis_path))

                # ==========GENERATOR MLE TRAINING==========
                self._print('\nStarting Generator MLE Training...\n')
                if not cfg.gen_pretrain:
                    self.pretrain_generator(cfg.MLE_train_epoch)
                    if cfg.if_save:
                        torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
                        print('Save MLE pretrain_generator gen: {}\n'.format(cfg.pretrained_gen_path))
            else:
                self._print('\n>>> Stop by pre_signal! Skip to adversarial training...\n')
                break

        # ==========ADVERSARIAL TRAINING==========
        self._print('\nStarting Adversarial Training...\n')

        oracle_nll, gen_nll = self.cal_metrics()
        self._print('Initial generator: oracle_NLL = %.4f, gen_NLL = %.4f\n' % (oracle_nll, gen_nll))

        for epoch in range(cfg.ADV_train_epoch):
            self._print('\n-----\nADV EPOCH %d\n-----\n' % epoch)
            self.sig.update()
            if self.sig.adv_sig:
                # TRAIN GENERATOR
                self.adv_train_generator(cfg.ADV_g_step)
                # TRAIN DISCRIMINATOR
                self.train_discriminator(cfg.ADV_d_step, cfg.ADV_d_epoch, 'ADV')
            else:
                self._print('\n>>> Stop by adv_signal! Finishing adversarial training...\n')
                break

    def _test(self):
        print('>>> Begin test...')

        self._run()
        # self.cal_metrics()
        pass

    def pretrain_generator(self, epochs):
        """
        Max Likelihood Pretraining for the gen

        - gen_opt: [mana_opt, work_opt]
        """
        t0 = time.time()
        for epoch in range(epochs):
            self.sig.update()
            if self.sig.pre_sig:
                pre_mana_loss = 0
                pre_work_loss = 0

                # =====Train=====
                for i, data in enumerate(self.gen_data.loader):
                    inp, target = data['input'], data['target']
                    if cfg.CUDA:
                        inp, target = inp.cuda(), target.cuda()

                    mana_loss, work_loss = self.gen.pretrain_loss(target, self.dis)
                    self.optimize_multi(self.gen_opt, [mana_loss, work_loss])
                    pre_mana_loss += mana_loss.data.item()
                    pre_work_loss += work_loss.data.item()
                pre_mana_loss = pre_mana_loss / len(self.gen_data.loader)
                pre_work_loss = pre_work_loss / len(self.gen_data.loader)

                # =====Test=====
                if epoch % cfg.pre_log_step == 0:
                    oracle_nll, gen_nll = self.cal_metrics()
                    t1 = time.time()
                    self._print('[MLE-GEN] epoch %d : pre_mana_loss = %.4f, pre_work_loss = %.4f, '
                                'oracle_NLL = %.4f, gen_NLL = %.4f, time = %.4f\n' % (
                                    epoch, pre_mana_loss, pre_work_loss, oracle_nll, gen_nll, t1 - t0))
                    t0 = time.time()

                    if cfg.if_save:
                        torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
            else:
                self._print('\n>>> Stop by pre signal, skip to adversarial training...')
                break

    def adv_train_generator(self, g_step, current_k=0):
        """
        The gen is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """

        rollout_func = rollout.ROLLOUT(self.gen, cfg.CUDA)
        adv_mana_loss = 0
        adv_work_loss = 0
        for step in range(g_step):
            with torch.no_grad():
                gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, self.dis,
                                              train=True)  # !!! train=True, the only place
                inp, target = self.gen_data.prepare(gen_samples, gpu=cfg.CUDA)

            # =====Train=====
            rewards = rollout_func.get_reward_leakgan(target, cfg.rollout_num, self.dis,
                                                      current_k).cpu()  # reward with MC search
            mana_loss, work_loss = self.gen.adversarial_loss(target, rewards, self.dis)

            # update parameters
            self.optimize_multi(self.gen_opt, [mana_loss, work_loss])
            adv_mana_loss += mana_loss.data.item()
            adv_work_loss += work_loss.data.item()
        # =====Test=====
        oracle_nll, gen_nll = self.cal_metrics()

        self._print('[ADV-GEN] adv_mana_loss = %.4f, adv_work_loss = %.4f, oracle_NLL = %.4f, gen_NLL = %.4f,\n' % (
            adv_mana_loss / g_step, adv_work_loss / g_step, oracle_nll, gen_nll))

    def train_discriminator(self, d_step, d_epoch, phrase='MLE'):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
        Samples are drawn d_step times, and the discriminator is trained for d_epoch d_epoch.
        """
        # prepare loader for validate
        with torch.no_grad():
            pos_val = self.oracle.sample(cfg.samples_num, 4 * cfg.batch_size)
            neg_val = self.gen.sample(cfg.samples_num, cfg.batch_size, self.dis)
            dis_val_data = DisDataIter(pos_val, neg_val)

        for step in range(d_step):
            # prepare loader for training
            with torch.no_grad():
                pos_samples = self.oracle_samples
                neg_samples = self.gen.sample(cfg.samples_num, cfg.batch_size, self.dis)
                dis_train_data = DisDataIter(pos_samples, neg_samples)

            for epoch in range(d_epoch):
                # =====Train=====
                d_loss, train_acc = self.train_dis_epoch(self.dis, dis_train_data.loader, self.dis_criterion,
                                                         self.dis_opt)

                # =====Test=====
                _, eval_acc = self.eval_dis(self.dis, dis_val_data.loader, self.dis_criterion)
                self._print('[%s-DIS] d_step %d, d_epoch %d: d_loss = %.4f, train_acc = %.4f, eval_acc = %.4f,\n' % (
                    phrase, step, epoch, d_loss, train_acc, eval_acc))

    def cal_metrics(self):
        oracle_nll = self.eval_gen(self.oracle,
                                   self.gen_data.reset(self.gen.sample(cfg.samples_num, cfg.batch_size, self.dis)),
                                   self.mle_criterion)

        gen_nll = 0
        for data in self.oracle_data.loader:
            inp, target = data['input'], data['target']
            if cfg.CUDA:
                inp, target = inp.cuda(), target.cuda()
            loss = self.gen.batchNLLLoss(target, self.dis)
            gen_nll += loss.item()
        gen_nll /= len(self.oracle_data.loader)

        return oracle_nll, gen_nll
