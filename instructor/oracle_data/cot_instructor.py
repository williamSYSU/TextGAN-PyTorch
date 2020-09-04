# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : cot_instructor.py
# @Time         : Created at 2020/4/20
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import config as cfg
from instructor.oracle_data.instructor import BasicInstructor
from models.CoT_D import Cot_D
from models.CoT_G import CoT_G
from utils.data_loader import GenDataIter


class CoTInstructor(BasicInstructor):
    def __init__(self, opt):
        super(CoTInstructor, self).__init__(opt)

        # generator, discriminator
        self.gen = CoT_G(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                         cfg.padding_idx, gpu=cfg.CUDA)
        self.dis = Cot_D(cfg.gen_embed_dim * 2, cfg.gen_hidden_dim * 2, cfg.vocab_size, cfg.max_seq_len,
                         cfg.padding_idx, gpu=cfg.CUDA)  # embed_dim and hidden_dim is larger
        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.gen_lr)

    def _run(self):
        # ===PRE-TRAINING===
        # TRAIN GENERATOR
        if not cfg.gen_pretrain:
            self.log.info('Starting Generator MLE Training...')
            self.pretrain_generator(cfg.MLE_train_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
                print('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path))

        # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')

        progress = tqdm(range(cfg.ADV_train_epoch))
        for epoch in progress:
            g_loss = self.adv_train_generator(cfg.ADV_g_step)  # Generator
            d_loss = self.train_mediator(epoch, cfg.ADV_d_step)  # Discriminator

            progress.set_description('g_loss: %.4f, d_loss: %.4f' % (g_loss, d_loss))

            if epoch % cfg.adv_log_step == 0 or epoch == cfg.ADV_train_epoch - 1:
                self.log.info('[ADV]: epoch = %d, %s' % (epoch, self.cal_metrics(fmt_str=True)))
                if cfg.if_save and not cfg.if_test:
                    self._save('ADV', epoch)
                    torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)

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
                pre_loss = self.train_gen_epoch(self.gen, self.oracle_data.loader, self.mle_criterion, self.gen_opt)

                # ===Test===
                if epoch % cfg.pre_log_step == 0 or epoch == epochs - 1:
                    self.log.info(
                        '[MLE-GEN] epoch %d : pre_loss = %.4f, %s' % (epoch, pre_loss, self.cal_metrics(fmt_str=True)))
                    if cfg.if_save and not cfg.if_test:
                        self._save('MLE', epoch)
            else:
                self.log.info('>>> Stop by pre signal, skip to adversarial training...')
                break

    def adv_train_generator(self, g_step):
        """
        Train the generator with mediator rewards
        """
        g_loss = []
        for step in range(g_step):
            inp, target = GenDataIter.prepare(self.gen.sample(cfg.batch_size, cfg.batch_size), gpu=cfg.CUDA)

            # ===Train===
            rewards = self.dis(inp, self.dis.init_hidden(cfg.batch_size))
            loss = self.gen.get_loss(inp, rewards)
            self.optimize(self.gen_adv_opt, loss)
            g_loss.append(loss.item())

        return np.mean(g_loss)

    def train_mediator(self, cur_epoch, d_step):
        """
        Training the mediator on real_data_samples (positive) and generated samples from gen (negative).
        """
        d_loss = []
        for step in range(d_step):
            # prepare loader for training
            real = list(self.oracle_data.loader)[cur_epoch % len(self.oracle_data.loader)]  # traverse all real data
            real_inp, real_tar = real['input'], real['target']
            fake_inp, fake_tar = GenDataIter.prepare(self.gen.sample(cfg.batch_size, cfg.batch_size), gpu=cfg.CUDA)
            if cfg.CUDA:
                real_inp, real_tar = real_inp.cuda(), real_tar.cuda()

            real_pred = self.dis.get_pred(real_inp, real_tar)
            fake_pred = self.dis.get_pred(fake_inp, fake_tar)
            loss = -torch.mean(real_pred + fake_pred) / 2.0

            self.optimize(self.dis_opt, loss)
            d_loss.append(loss.item())

        return np.mean(d_loss)
