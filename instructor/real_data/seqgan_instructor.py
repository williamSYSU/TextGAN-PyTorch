# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : seqgan_instructor.py
# @Time         : Created at 2019-06-05
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn
import torch.optim as optim

import config as cfg
from instructor.real_data.instructor import BasicInstructor
from metrics.bleu import BLEU
from models.SeqGAN_D import SeqGAN_D
from models.SeqGAN_G import SeqGAN_G
from utils import rollout
from utils.data_loader import GenDataIter, DisDataIter
from utils.text_process import tensor_to_tokens


class SeqGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(SeqGANInstructor, self).__init__(opt)

        # generator, discriminator
        self.gen = SeqGAN_G(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                            cfg.padding_idx, cfg.temperature, gpu=cfg.CUDA)
        self.dis = SeqGAN_D(cfg.dis_embed_dim, cfg.vocab_size, cfg.padding_idx, gpu=cfg.CUDA)
        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)

        # Criterion
        self.mle_criterion = nn.NLLLoss()
        self.dis_criterion = nn.CrossEntropyLoss()

        # DataLoader
        self.gen_data = GenDataIter(self.gen.sample(cfg.batch_size, cfg.batch_size))
        self.dis_data = DisDataIter(self.gen_data.random_batch()['target'], self.oracle_data.random_batch()['target'])

        # Metrics
        self.bleu3 = BLEU(test_text=tensor_to_tokens(self.gen_data.target, self.index_word_dict),
                          real_text=tensor_to_tokens(self.test_data.target, self.index_word_dict),
                          gram=3)

    def _run(self):
        # =====PRE-TRAINING=====
        # TRAIN GENERATOR
        if not cfg.gen_pretrain:
            self.log.info('Starting Generator MLE Training...')
            self.pretrain_generator(cfg.MLE_train_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
                print('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path))

        # =====TRAIN DISCRIMINATOR======
        if not cfg.dis_pretrain:
            self.log.info('Starting Discriminator Training...')
            self.train_discriminator(cfg.d_step, cfg.d_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)
                print('Save pretrain_generator discriminator: {}'.format(cfg.pretrained_dis_path))

        # =====ADVERSARIAL TRAINING=====
        self.log.info('Starting Adversarial Training...')
        self.log.info('Initial generator: %s' % (self.cal_metrics(fmt_str=True)))

        for adv_epoch in range(cfg.ADV_train_epoch):
            self.log.info('-----\nADV EPOCH %d\n-----' % adv_epoch)
            self.sig.update()
            if self.sig.adv_sig:
                self.adv_train_generator(cfg.ADV_g_step)  # Generator
                self.train_discriminator(cfg.ADV_d_step, cfg.ADV_d_epoch, 'ADV')  # Discriminator

                if adv_epoch % cfg.adv_log_step == 0:
                    if cfg.if_save and not cfg.if_test:
                        self._save('ADV', adv_epoch)
            else:
                self.log.info('>>> Stop by adv_signal! Finishing adversarial training...')
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
        """
        The gen is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """
        rollout_func = rollout.ROLLOUT(self.gen, cfg.CUDA)
        total_g_loss = 0
        for step in range(g_step):
            inp, target = self.gen_data.prepare(self.gen.sample(cfg.batch_size, cfg.batch_size), gpu=cfg.CUDA)

            # =====Train=====
            rewards = rollout_func.get_reward(target, cfg.rollout_num, self.dis)
            adv_loss = self.gen.batchPGLoss(inp, target, rewards)
            self.optimize(self.gen_opt, adv_loss)
            total_g_loss += adv_loss.item()

        # =====Test=====
        self.log.info('[ADV-GEN]: g_loss = %.4f, %s' % (total_g_loss, self.cal_metrics(fmt_str=True)))

    def train_discriminator(self, d_step, d_epoch, phrase='MLE'):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
        Samples are drawn d_step times, and the discriminator is trained for d_epoch d_epoch.
        """
        # prepare loader for validate
        for step in range(d_step):
            # prepare loader for training
            pos_samples = self.oracle_data.target
            neg_samples = self.gen.sample(cfg.samples_num, 4 * cfg.batch_size)
            self.dis_data.reset(pos_samples, neg_samples)

            for epoch in range(d_epoch):
                # =====Train=====
                d_loss, train_acc = self.train_dis_epoch(self.dis, self.dis_data.loader, self.dis_criterion,
                                                         self.dis_opt)

            # =====Test=====
            self.log.info('[%s-DIS] d_step %d: d_loss = %.4f, train_acc = %.4f,' % (
                phrase, step, d_loss, train_acc))
