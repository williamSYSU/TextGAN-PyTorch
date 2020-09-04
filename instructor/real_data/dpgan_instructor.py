# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : dpgan_instructor.py
# @Time         : Created at 2019/12/21
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.optim as optim

import config as cfg
from instructor.real_data.instructor import BasicInstructor
from models.DPGAN_D import DPGAN_D
from models.DPGAN_G import DPGAN_G


class DPGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(DPGANInstructor, self).__init__(opt)

        # generator, discriminator
        self.gen = DPGAN_G(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                           cfg.padding_idx, gpu=cfg.CUDA)
        self.dis = DPGAN_D(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                           cfg.padding_idx, gpu=cfg.CUDA)
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

        # # ===TRAIN DISCRIMINATOR====
        if not cfg.dis_pretrain:
            self.log.info('Starting Discriminator Training...')
            self.train_discriminator(cfg.d_step, cfg.d_epoch, 'MLE')
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
                pre_loss = self.train_gen_epoch(self.gen, self.train_data.loader, self.mle_criterion, self.gen_opt)

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
        The gen is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """
        discount_rate = 1
        total_g_loss = 0
        dis_count_list = [discount_rate ** i for i in range(cfg.max_seq_len)]
        dis_count_matrix = torch.Tensor(dis_count_list).unsqueeze(0).repeat(cfg.batch_size, 1)
        if cfg.CUDA:
            dis_count_matrix = dis_count_matrix.cuda()

        for step in range(g_step):
            inp = self.train_data.random_batch()['input']
            if cfg.CUDA:
                inp = inp.cuda()

            gen_sample, gen_sample_log_prob = self.gen.sample_teacher_forcing(inp)
            word_reward, sentence_reward = self.dis.getReward(gen_sample)
            sentence_reward = sentence_reward.repeat(1, cfg.max_seq_len)
            reward_matrix = sentence_reward * word_reward * dis_count_matrix
            for i in range(cfg.max_seq_len):
                reward_matrix[:, i] = reward_matrix[:, i:].sum(dim=-1)

            adv_loss = torch.sum(gen_sample_log_prob * reward_matrix)

            self.optimize(self.gen_adv_opt, adv_loss, self.gen)
            total_g_loss += adv_loss.item()

        # ===Test===
        self.log.info(
            '[ADV-GEN]: g_loss = %.4f, %s' % (total_g_loss / (g_step * cfg.batch_size), self.cal_metrics(fmt_str=True)))

    def train_discriminator(self, d_step, d_epoch, phase='MLE'):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
        Samples are drawn d_step times, and the discriminator is trained for d_epoch d_epoch.
        """
        # prepare loader for validate
        for step in range(d_step):
            # prepare loader for training
            pos_samples = self.train_data.target
            neg_samples = self.gen.sample(pos_samples.size(0), 4 * cfg.batch_size)

            pos_reward, neg_reward = 0, 0
            for epoch in range(d_epoch):
                # ===Train===
                pos_reward, neg_reward = self.train_dis_epoch(self.dis, pos_samples, neg_samples, self.dis_opt)

            # ===Test===
            self.log.info('[%s-DIS] d_step %d: pos_reward = %.4f, neg_reward = %.4f,' % (
                phase, step, pos_reward, neg_reward))

            if cfg.if_save and not cfg.if_test:
                torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)

    def eval_dis(self, model, pos_val, neg_val):
        _, pos_reward = model.getReward(pos_val)
        _, neg_reward = model.getReward(neg_val)
        return torch.mean(pos_reward), torch.mean(neg_reward)

    def train_dis_epoch(self, model, pos_samples, neg_samples, optimizer):
        pos_reward, neg_reward = 0, 0
        num_samples = pos_samples.size(0)
        num_batch = num_samples // cfg.batch_size
        for i in range(num_batch):
            pos_sample = pos_samples[i * cfg.batch_size: (i + 1) * cfg.batch_size]
            neg_sample = neg_samples[i * cfg.batch_size: (i + 1) * cfg.batch_size]

            _, pos_reward = model.getReward(pos_sample)
            _, neg_reward = model.getReward(neg_sample)

            loss = -torch.mean(pos_reward) + torch.mean(neg_reward)
            self.optimize(optimizer, loss, model)
        return pos_reward.mean().item(), neg_reward.mean().item()
