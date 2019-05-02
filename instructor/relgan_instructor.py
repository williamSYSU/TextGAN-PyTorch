# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : relgan_instructor.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

from math import ceil
import sys
import time

import torch
import torch.optim as optim
import torch.nn as nn

import helpers
import rollout
import config as cfg
from instructor.instructor import BasicInstructor
from models.RelGAN_G import RelGAN_G
from models.RelGAN_D import RelGAN_D


class RelGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(RelGANInstructor, self).__init__(opt)

        # generator, discriminator
        self.gen = RelGAN_G(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                                    cfg.padding_idx, gpu=cfg.CUDA)
        self.dis = RelGAN_D(cfg.dis_embed_dim, cfg.vocab_size, cfg.dis_filter_sizes, cfg.dis_num_filters,
                                    cfg.k_label, cfg.padding_idx, gpu=cfg.CUDA)

        self.init_model()

        # optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)

    def _run(self):
        oracle_samples = self.oracle.sample(cfg.samples_num, cfg.batch_size)
        # oracle_samples = self.oracle.sample(cfg.samples_num, cfg.samples_num)

        # PRE-TRAIN GENERATOR
        self._print('\nStarting Generator MLE Training...\n')

        self._print('Generator MLE training...\n')
        self.pretrain_generator(oracle_samples, cfg.MLE_train_epoch)

    def pretrain_generator(self, real_samples, epochs):
        """
        Max Likelihood Pretraining for the gen

        - gen_opt: [mana_opt, work_opt]
        """
        for epoch in range(epochs):
            self.sig.update()
            if self.sig.pre_sig:
                self._print('epoch %d : ' % (epoch + 1))
                pre_loss = 0

                t0 = time.time()
                for i in range(0, cfg.samples_num, cfg.batch_size):
                    # =====Train=====
                    self.gen.train()
                    self.dis.train()

                    inp, target = helpers.prepare_generator_batch(real_samples[i:i + cfg.batch_size],
                                                                  start_letter=cfg.start_letter, gpu=cfg.CUDA)

                    loss = self.gen.batchNLLLoss(inp, target)

                    # update parameters
                    self.optimize(self.gen_opt, loss)

                    pre_loss += loss.data.item() / cfg.max_seq_len

                    if (i / cfg.batch_size) % ceil(
                            ceil(cfg.samples_num / float(cfg.batch_size)) / 10.) == 0:  # roughly every 10% of an epoch
                        self._print('.')

                # each loss in a batch is loss per sample
                pre_loss = pre_loss / ceil(cfg.samples_num / float(cfg.batch_size))

                # =====Test=====
                oracle_nll, gen_nll = self.eval_gen()

                t1 = time.time()

                self._print(' pre_loss = %.4f, oracle_NLL = %.4f, gen_NLL = %.4f, time = %.4f\n' % (
                    pre_loss, oracle_nll, gen_nll, t1 - t0))

            else:
                self._print('\n>>> Stop by pre signal, skip to adversarial training...')
                break