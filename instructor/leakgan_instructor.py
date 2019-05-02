# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : leakgan_instructor.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.
from math import ceil
import time

import torch
import torch.optim as optim
import torch.nn as nn

import helpers
import rollout
import config as cfg

from instructor.instructor import BasicInstructor
from models.LeakGAN_D import LeakGAN_D
from models.LeakGAN_G import LeakGAN_G


class LeakGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(LeakGANInstructor, self).__init__(opt)

        # generator, discriminator
        self.gen = LeakGAN_G(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                             cfg.padding_idx, cfg.goal_size, cfg.goal_out_size, cfg.step_size, cfg.CUDA)
        self.dis = LeakGAN_D(cfg.dis_embed_dim, cfg.vocab_size, cfg.dis_filter_sizes, cfg.dis_num_filters,
                             cfg.k_label, cfg.padding_idx, gpu=cfg.CUDA)
        self.init_model()

        # optimizer
        mana_params, work_params = self.gen.split_params()
        mana_opt = optim.Adam(mana_params, lr=cfg.gen_lr)
        work_opt = optim.Adam(work_params, lr=cfg.gen_lr)

        self.gen_opt = [mana_opt, work_opt]
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)

    def _run(self):
        oracle_samples = self.oracle.sample(cfg.samples_num, cfg.batch_size)
        for inter_num in range(cfg.inter_epoch):
            self._print('\n>>> Interleaved Round %d...\n' % inter_num)
            self.sig.update()  # update signal
            if self.sig.pre_sig:
                # ==========DISCRIMINATOR PRE-TRAINING==========
                self._print('\nStarting Discriminator Training...\n')
                if not cfg.dis_pretrain:
                    self.train_discriminator(cfg.d_step, cfg.d_epoch)

                    if cfg.if_save:
                        torch.save(self.dis.state_dict(), cfg.pretrained_dis_path.format(cfg.k_label))
                        print('Save pretrain_generator discriminator: {}\n'.format(
                            cfg.pretrained_dis_path.format(cfg.k_label)))

                # ==========GENERATOR MLE TRAINING==========
                self._print('\nStarting Generator MLE Training...\n')
                if not cfg.gen_pretrain:
                    self._print('Generator MLE training...\n')
                    self.pretrain_generator(oracle_samples, cfg.MLE_train_epoch)

                    if cfg.if_save:
                        torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
                        print('Save MLE pretrain_generator gen: {}\n'.format(cfg.pretrained_gen_path))
            else:
                self._print('\n>>> Stop by pre_signal! Skip to adversarial training...\n')
                break

        # ==========ADVERSARIAL TRAINING==========
        self._print('\nStarting Adversarial Training...\n')

        oracle_nll = self.get_nll(self.gen.sample(cfg.samples_num, cfg.batch_size, self.dis), self.oracle)

        self._print('Generator: Initial Oracle Sample NLL : %.4f\n' % oracle_nll)

        for epoch in range(cfg.ADV_train_epoch):
            self._print('\n--------\nEPOCH %d\n--------\n' % (epoch + 1))
            self.sig.update()
            if self.sig.adv_sig:
                # TRAIN GENERATOR
                self._print('\nAdversarial Training Generator: \n')
                self.adversarial_train_generator(cfg.ADV_g_step)

                # TRAIN DISCRIMINATOR
                self._print('\nAdversarial Training Discriminator : \n')
                self.train_discriminator(cfg.ADV_d_step, cfg.ADV_d_epoch)
            else:
                self._print('\n>>> Stop by adv_signal! Finishing adversarial training...\n')
                break

    def pretrain_generator(self, real_data_samples, epochs):
        """
        Max Likelihood Pretraining for the gen

        - gen_opt: [mana_opt, work_opt]
        """
        for epoch in range(epochs):
            self.sig.update()
            if self.sig.pre_sig:
                self._print('epoch %d : ' % (epoch + 1))
                pre_mana_loss = 0
                pre_work_loss = 0

                t0 = time.time()
                for i in range(0, cfg.samples_num, cfg.batch_size):
                    # =====Train=====
                    self.gen.train()
                    self.dis.train()

                    inp, target = helpers.prepare_generator_batch(real_data_samples[i:i + cfg.batch_size],
                                                                  start_letter=cfg.start_letter, gpu=cfg.CUDA)

                    mana_loss, work_loss = self.gen.pretrain_loss(target, self.dis)

                    # update parameters
                    self.optimize_multi(self.gen_opt, [mana_loss, work_loss])

                    pre_mana_loss += mana_loss.data.item()
                    pre_work_loss += work_loss.data.item()

                    if (i / cfg.batch_size) % ceil(
                            ceil(cfg.samples_num / float(cfg.batch_size)) / 10.) == 0:  # roughly every 10% of an epoch
                        self._print('.')

                # each loss in a batch is loss per sample
                pre_mana_loss = pre_mana_loss / ceil(cfg.samples_num / float(cfg.batch_size))
                pre_work_loss = pre_work_loss / ceil(cfg.samples_num / float(cfg.batch_size))

                # =====Test=====
                oracle_nll, gen_nll = self.eval_gen()

                t1 = time.time()

                self._print(
                    ' pre_mana_loss = %.4f, pre_work_loss = %.4f, oracle_NLL = %.4f, gen_NLL = %.4f, '
                    'time = %.4f\n' % (
                        pre_mana_loss, pre_work_loss, oracle_nll, gen_nll, t1 - t0))

            else:
                self._print('\n>>> Stop by pre signal, skip to adversarial training...')
                break

    def adversarial_train_generator(self, g_step, current_k=0):
        """
        The gen is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """

        rollout_func = rollout.ROLLOUT(self.gen, cfg.CUDA)
        adv_mana_loss = 0
        adv_work_loss = 0

        for step in range(g_step):
            with torch.no_grad():
                gen_s = self.gen.sample(cfg.batch_size, cfg.batch_size, self.dis,
                                        train=True)  # !!! train=True, the only place
                inp, target = helpers.prepare_generator_batch(gen_s, start_letter=cfg.start_letter, gpu=cfg.CUDA)

            # =====Train=====
            self.gen.train()
            self.dis.train()

            rewards = rollout_func.get_reward_leakgan(target, cfg.rollout_num, self.dis,
                                                      current_k).cpu()  # reward with MC search

            mana_loss, work_loss = self.gen.adversarial_loss(target, rewards, self.dis)

            # update parameters
            self.optimize_multi(self.gen_opt, [mana_loss, work_loss])

            adv_mana_loss += mana_loss.data.item()
            adv_work_loss += work_loss.data.item()

        # =====Test=====
        oracle_loss, inverse_nll = self.eval_gen()

        self._print(' adv_mana_loss = %.4f, adv_work_loss = %.4f, oracle_NLL = %.4f\n' % (
            adv_mana_loss / g_step, adv_work_loss / g_step, oracle_loss))
        self._print('inverse_NLL = %.4f\n' % inverse_nll)

    def train_discriminator(self, d_steps, epochs):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
        Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
        """
        # prepare data for validate
        with torch.no_grad():
            pos_val = self.oracle.sample(cfg.samples_num, cfg.batch_size)
            neg_val = self.gen.sample(cfg.samples_num, cfg.batch_size, self.dis)

            val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val)

        # loss_fn = nn.BCELoss()
        loss_fn = nn.CrossEntropyLoss()
        for d_step in range(d_steps):
            # prepare data for training
            with torch.no_grad():
                oracle_samples = self.oracle.sample(cfg.samples_num, cfg.batch_size)
                gen_samples = self.gen.sample(cfg.samples_num, cfg.batch_size, self.dis)
                dis_inp, dis_target = helpers.prepare_discriminator_data(oracle_samples, gen_samples)

            # training
            for epoch in range(epochs):
                self._print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1))
                total_loss = 0
                total_acc = 0
                train_size = 2 * cfg.samples_num

                for i in range(0, train_size, cfg.batch_size):
                    # =====Train=====
                    self.dis.train()
                    self.gen.eval()

                    inp, target = dis_inp[i:i + cfg.batch_size], dis_target[i:i + cfg.batch_size]

                    if cfg.CUDA:
                        inp = inp.cuda()
                        target = target.cuda()

                    out = self.dis.batchClassify(inp)
                    loss = loss_fn(out, target)

                    self.optimize(self.dis_opt, loss)

                    total_loss += loss.item()
                    total_acc += torch.sum((out.argmax(dim=-1) == target)).item()

                    if (i / cfg.batch_size) % ceil(ceil(train_size / float(
                            cfg.batch_size)) / 10.) == 0:  # roughly every 10% of an epoch
                        self._print('.')

                total_loss /= ceil(train_size / float(cfg.batch_size))
                total_acc /= float(train_size)

                # =====Test=====
                val_acc = self.eval_dis(val_inp, val_target)

                self._print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f\n' % (
                    total_loss, total_acc, val_acc))

    def eval_gen(self):
        self.gen.eval()
        self.dis.eval()
        with torch.no_grad():
            # sample from gen and compute oracle NLL
            gen_samples = self.gen.sample(cfg.samples_num, cfg.batch_size, self.dis)
            oracle_nll = self.get_nll(gen_samples, self.oracle)

            # real_samples = self.oracle.sample(cfg.samples_num, cfg.batch_size)
            # gen_nll = self.get_nll(real_samples, self.gen)
            gen_nll = 0

        return oracle_nll, gen_nll
