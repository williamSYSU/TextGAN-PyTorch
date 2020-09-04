# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : maligan_instructor.py
# @Time         : Created at 2019/10/17
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.


import torch
import torch.nn.functional as F
import torch.optim as optim

import config as cfg
from instructor.oracle_data.instructor import BasicInstructor
from models.MaliGAN_D import MaliGAN_D
from models.MaliGAN_G import MaliGAN_G
from utils.data_loader import GenDataIter, DisDataIter


class MaliGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(MaliGANInstructor, self).__init__(opt)

        # generator, discriminator
        self.gen = MaliGAN_G(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                             cfg.padding_idx, gpu=cfg.CUDA)
        self.dis = MaliGAN_D(cfg.dis_embed_dim, cfg.vocab_size, cfg.padding_idx, gpu=cfg.CUDA)
        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)

    def _run(self):
        # ===PRE-TRAINING===
        # TRAIN GENERATOR
        if not cfg.gen_pretrain:
            self.log.info('Starting Generator MLE Training...')
            self.pretrain_generator(cfg.MLE_train_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
                print('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path))

        # ===TRAIN DISCRIMINATOR====
        if not cfg.dis_pretrain:
            self.log.info('Starting Discriminator Training...')
            self.train_discriminator(cfg.d_step, cfg.d_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)
                print('Save pre-trained discriminator: {}'.format(cfg.pretrained_dis_path))

        # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')
        self.log.info('Initial generator: %s' % (self.cal_metrics(fmt_str=True)))

        for adv_epoch in range(cfg.ADV_train_epoch):
            self.log.info('-----\nADV EPOCH %d\n-----' % adv_epoch)
            self.sig.update()
            if self.sig.adv_sig:
                self.adv_train_generator(cfg.ADV_g_step)  # Generator
                self.train_discriminator(cfg.ADV_d_step, cfg.ADV_d_epoch, 'ADV')  # Discriminator

                if adv_epoch % cfg.adv_log_step == 0 or adv_epoch == cfg.ADV_train_epoch - 1:
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
        The gen is trained by MLE-like objective.
        """
        total_g_loss = 0
        for step in range(g_step):
            inp, target = GenDataIter.prepare(self.gen.sample(cfg.batch_size, cfg.batch_size), gpu=cfg.CUDA)

            # ===Train===
            rewards = self.get_mali_reward(target)
            adv_loss = self.gen.adv_loss(inp, target, rewards)
            self.optimize(self.gen_adv_opt, adv_loss)
            total_g_loss += adv_loss.item()

        # ===Test===
        self.log.info('[ADV-GEN]: g_loss = %.4f, %s' % (total_g_loss, self.cal_metrics(fmt_str=True)))

    def train_discriminator(self, d_step, d_epoch, phase='MLE'):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
        Samples are drawn d_step times, and the discriminator is trained for d_epoch d_epoch.
        """
        # prepare loader for validate
        global d_loss, train_acc
        pos_val = self.oracle.sample(8 * cfg.batch_size, 4 * cfg.batch_size)
        neg_val = self.gen.sample(8 * cfg.batch_size, 4 * cfg.batch_size)
        dis_eval_data = DisDataIter(pos_val, neg_val)

        for step in range(d_step):
            # prepare loader for training
            pos_samples = self.oracle_samples  # not re-sample the Oracle data
            neg_samples = self.gen.sample(cfg.samples_num, 4 * cfg.batch_size)
            dis_data = DisDataIter(pos_samples, neg_samples)

            for epoch in range(d_epoch):
                # ===Train===
                d_loss, train_acc = self.train_dis_epoch(self.dis, dis_data.loader, self.dis_criterion,
                                                         self.dis_opt)

            # ===Test===
            _, eval_acc = self.eval_dis(self.dis, dis_eval_data.loader, self.dis_criterion)
            self.log.info('[%s-DIS] d_step %d: d_loss = %.4f, train_acc = %.4f, eval_acc = %.4f,' % (
                phase, step, d_loss, train_acc, eval_acc))

            if cfg.if_save and not cfg.if_test:
                torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)

    def get_mali_reward(self, samples):
        rewards = []
        for _ in range(cfg.rollout_num):
            dis_out = F.softmax(self.dis(samples), dim=-1)[:, 1]
            rewards.append(dis_out)

        rewards = torch.mean(torch.stack(rewards, dim=0), dim=0)  # batch_size
        rewards = torch.div(rewards, 1 - rewards)
        rewards = torch.div(rewards, torch.sum(rewards))
        rewards -= torch.mean(rewards)
        rewards = rewards.unsqueeze(1).expand(samples.size())  # batch_size * seq_len

        return rewards
