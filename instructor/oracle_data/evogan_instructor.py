# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : evogan_instructor.py
# @Time         : Created at 2019-07-09
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import config as cfg
from instructor.oracle_data.instructor import BasicInstructor
from models.EvoGAN_D import EvoGAN_D
from models.EvoGAN_G import EvoGAN_G
from utils.data_loader import GenDataIter
from utils.helpers import get_fixed_temperature, get_losses


class EvoGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(EvoGANInstructor, self).__init__(opt)

        # generator, discriminator
        self.gen = EvoGAN_G(cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim, cfg.gen_hidden_dim,
                            cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        self.dis = EvoGAN_D(cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.vocab_size, cfg.padding_idx,
                            gpu=cfg.CUDA)

        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_adv_lr)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)

        # Criterion
        self.mle_criterion = nn.NLLLoss()
        self.cross_entro_cri = nn.CrossEntropyLoss()

        # DataLoader
        self.gen_data = GenDataIter(self.gen.sample(cfg.batch_size, cfg.batch_size))

    def _run(self):
        # =====PRE-TRAINING (GENERATOR)=====
        if not cfg.gen_pretrain:
            self.log.info('Starting Generator MLE Training...')
            self.pretrain_generator(cfg.MLE_train_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
                print('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path))

        # # =====ADVERSARIAL TRAINING=====
        self.log.info('Starting Adversarial Training...')
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
                    # if adv_epoch > 50 and adv_epoch % cfg.adv_log_step == 0:
                    self.log.info('[ADV] epoch %d: g_loss: %.4f, d_loss: %.4f, %s' % (
                        adv_epoch, g_loss, d_loss, self.cal_metrics(fmt_str=True)))

                    if cfg.if_save and not cfg.if_test:
                        self._save('ADV', adv_epoch)
            else:
                self.log.info('>>> Stop by adv_signal! Finishing adversarial training...')
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
        for epoch in range(epochs):
            self.sig.update()
            if self.sig.pre_sig:
                # =====Train=====
                pre_loss = self.train_gen_epoch(self.gen, self.oracle_data.loader, self.mle_criterion, self.gen_opt)

                # =====Test=====
                if epoch % cfg.pre_log_step == 0:
                    self.log.info(
                        '[MLE-GEN] epoch %d : pre_loss = %.4f, %s' % (epoch, pre_loss, self.cal_metrics(fmt_str=True)))

                    if cfg.if_save and not cfg.if_test:
                        self._save('MLE', epoch)
            else:
                self.log.info('>>> Stop by pre signal, skip to adversarial training...')
                break
        if cfg.if_save and not cfg.if_test:
            self._save('MLE', epoch)

    def adv_train_generator(self, g_step):
        total_loss = 0
        for step in range(g_step):
            real_samples = F.one_hot(self.oracle_data.random_batch()['target'], cfg.vocab_size).float()
            gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True)
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()

            # =====Train=====
            d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
            g_loss, _ = get_losses(d_out_real, d_out_fake, cfg.loss_type)  # origin: RSGAN

            # Cross Entropy
            # g_target = torch.ones(cfg.batch_size).long().cuda()
            # g_loss = self.cross_entro_cri(d_out_fake, g_target)

            # Reverse Cross Entropy
            # g_target = torch.zeros(cfg.batch_size).long().cuda()
            # g_loss = -0.5 * self.cross_entro_cri(d_out_fake, g_target)
            # g_loss = -self.cross_entro_cri(d_out_fake, g_target)

            # g_loss = -torch.sum(F.softmax(d_out_fake, dim=-1)[:, 1])

            self.optimize(self.gen_adv_opt, g_loss, self.gen)
            total_loss += g_loss.item()

        return total_loss / g_step if g_step != 0 else 0

    def adv_train_discriminator(self, d_step):
        total_loss = 0
        for step in range(d_step):
            real_samples = F.one_hot(self.oracle_data.random_batch()['target'], cfg.vocab_size).float()
            gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True)
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()

            # =====Train=====
            d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
            _, d_loss = get_losses(d_out_real, d_out_fake, cfg.loss_type)  # origin: RSGAN

            # Cross Entropy
            # real_target = torch.ones(cfg.batch_size).long().cuda()
            # fake_target = torch.zeros(cfg.batch_size).long().cuda()
            # d_real_loss = self.cross_entro_cri(d_out_real, real_target)
            # d_fake_loss = self.cross_entro_cri(d_out_fake, fake_target)
            # d_loss = d_real_loss + d_fake_loss

            self.optimize(self.dis_opt, d_loss, self.dis)
            total_loss += d_loss.item()

        return total_loss / d_step if d_step != 0 else 0

    def update_temperature(self, i, N):
        self.gen.temperature = get_fixed_temperature(cfg.temperature, i, N, cfg.temp_adpt)

    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False):
        """Add clip_grad_norm_"""
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()
