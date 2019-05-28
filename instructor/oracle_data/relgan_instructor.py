# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : relgan_instructor.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import config as cfg
from instructor.oracle_data.instructor import BasicInstructor
from models.RelGAN_D import RelGAN_D
from models.RelGAN_G import RelGAN_G
from utils.data_utils import GenDataIter
from utils.helpers import get_fixed_temperature


class RelGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(RelGANInstructor, self).__init__(opt)

        # generator, discriminator
        self.gen = RelGAN_G(cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim, cfg.gen_hidden_dim,
                            cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        self.dis = RelGAN_D(cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.vocab_size, cfg.padding_idx,
                            gpu=cfg.CUDA)

        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_adv_lr)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)

        # Criterion
        self.mle_criterion = nn.NLLLoss()

        # DataLoader
        self.gen_data = GenDataIter(self.gen.sample(cfg.batch_size, cfg.batch_size))

    def _run(self):
        # ==========PRE-TRAINING (GENERATOR)==========
        if not cfg.gen_pretrain:
            self._print('\nStarting Generator MLE Training...\n')
            self.pretrain_generator(cfg.MLE_train_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
                print('Save pre-trained generator: {}\n'.format(cfg.pretrained_gen_path))

        oracle_nll, gen_nll, self_nll = self.cal_metrics()
        self._print('Initial generator: oracle_NLL = %.4f, gen_NLL = %.4f, self_NLL = %.4f,\n' % (
            oracle_nll, gen_nll, self_nll))

        # # ==========ADVERSARIAL TRAINING==========
        self._print('\nStarting Adversarial Training...\n')
        progress = tqdm(range(cfg.ADV_train_epoch))
        for adv_epoch in progress:
            self.sig.update()
            if self.sig.adv_sig:
                g_loss = self.adv_train_generator(cfg.ADV_g_step)  # Generator
                d_loss = self.adv_train_discriminator(cfg.ADV_d_step)  # Discriminator
                self.update_temperature(adv_epoch, cfg.ADV_train_epoch)  # update temperature

                progress.set_description(
                    'g_loss: %.4f, d_loss: %.4f, temperature: %.4f' % (g_loss, d_loss, self.gen.temperature))

                # TEST
                if adv_epoch % cfg.adv_log_step == 0:
                    oracle_nll, gen_nll, self_nll = self.cal_metrics()
                    self._print(
                        '[ADV] epoch %d: g_loss: %.4f, d_loss: %.4f, oracle_NLL = %.4f, gen_NLL = %.4f, self_NLL = %.4f,\n' % (
                            adv_epoch, g_loss, d_loss, oracle_nll, gen_nll, self_nll))

                    if cfg.if_save and not cfg.if_test:
                        self._save('ADV', adv_epoch)
            else:
                self._print('\n>>> Stop by adv_signal! Finishing adversarial training...\n')
                progress.close()
                break

    def _test(self):
        print('>>> Begin test...')

        self._run()
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
                    oracle_nll, gen_nll, self_nll = self.cal_metrics()
                    t1 = time.time()
                    self._print(
                        '[MLE-GEN] epoch %d : pre_loss = %.4f, oracle_NLL = %.4f, gen_NLL = %.4f, self_NLL = %.4f, time = %.4f,\n' % (
                            epoch, pre_loss, oracle_nll, gen_nll, self_nll, t1 - t0))
                    t0 = time.time()

                    if cfg.if_save and not cfg.if_test:
                        self._save('MLE', epoch)
            else:
                self._print('\n>>> Stop by pre signal, skip to adversarial training...')
                break
        if cfg.if_save and not cfg.if_test:
            self._save('MLE', epoch)

    def adv_train_generator(self, g_step):
        total_loss = 0
        for step in range(g_step):
            real_samples = self.oracle_data.random_batch()['target']
            gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True)
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            real_samples = F.one_hot(real_samples, cfg.vocab_size).float()

            # =====Train=====
            d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
            g_loss, _ = self.get_losses(d_out_real, d_out_fake)

            self.optimize(self.gen_adv_opt, g_loss, self.gen)
            total_loss += g_loss.item()

        return total_loss / g_step if g_step != 0 else 0

    def adv_train_discriminator(self, d_step):
        total_loss = 0
        for step in range(d_step):
            real_samples = self.oracle_data.random_batch()['target']
            gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True)
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            real_samples = F.one_hot(real_samples, cfg.vocab_size).float()

            # =====Train=====
            d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
            _, d_loss = self.get_losses(d_out_real, d_out_fake)

            self.optimize(self.dis_opt, d_loss, self.dis)
            total_loss += d_loss.item()

        return total_loss / d_step if d_step != 0 else 0

    def get_losses(self, d_out_real, d_out_fake):
        loss_type = cfg.loss_type
        bce_loss = nn.BCEWithLogitsLoss()

        if loss_type == 'standard':  # the non-satuating GAN loss
            d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
            d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
            d_loss = d_loss_real + d_loss_fake

            g_loss = bce_loss(d_out_fake, torch.ones_like(d_out_fake))

        elif loss_type == 'JS':  # the vanilla GAN loss
            d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
            d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
            d_loss = d_loss_real + d_loss_fake

            g_loss = -d_loss_fake

        elif loss_type == 'KL':  # the GAN loss implicitly minimizing KL-divergence
            d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
            d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
            d_loss = d_loss_real + d_loss_fake

            g_loss = torch.mean(-d_out_fake)

        elif loss_type == 'hinge':  # the hinge loss
            d_loss_real = torch.mean(nn.ReLU(1.0 - d_out_real))
            d_loss_fake = torch.mean(nn.ReLU(1.0 + d_out_fake))
            d_loss = d_loss_real + d_loss_fake

            g_loss = -torch.mean(d_out_fake)

        elif loss_type == 'tv':  # the total variation distance
            d_loss = torch.mean(nn.Tanh(d_out_fake) - nn.Tanh(d_out_real))
            g_loss = torch.mean(-nn.Tanh(d_out_fake))

        elif loss_type == 'RSGAN':  # relativistic standard GAN
            d_loss = bce_loss(d_out_real - d_out_fake, torch.ones_like(d_out_real))
            g_loss = bce_loss(d_out_fake - d_out_real, torch.ones_like(d_out_fake))

        else:
            raise NotImplementedError("Divergence '%s' is not implemented" % loss_type)

        return g_loss, d_loss

    def update_temperature(self, i, N):
        self.gen.temperature = get_fixed_temperature(cfg.temperature, i, N, cfg.temp_adpt)

    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False):
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()
