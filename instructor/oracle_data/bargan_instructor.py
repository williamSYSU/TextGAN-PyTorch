# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : bargan_instructor.py
# @Time         : Created at 2019-06-30
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import config as cfg
from instructor.oracle_data.instructor import BasicInstructor
from models.BarGAN_D import BarGAN_D
from models.BarGAN_G import BarGAN_G
from utils.data_loader import GenDataIter
from utils.helpers import get_fixed_temperature


class BarGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(BarGANInstructor, self).__init__(opt)

        # generator, discriminator
        self.gen = BarGAN_G(cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim, cfg.gen_hidden_dim,
                            cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        self.dis = BarGAN_D(cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.vocab_size, cfg.padding_idx,
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

        # Other params
        self.theta = 1.0

    def _run(self):
        # =====PRE-TRAINING (GENERATOR)=====
        # if not cfg.gen_pretrain:
        #     self.log.info('Starting Generator MLE Training...')
        #     self.pretrain_generator(cfg.MLE_train_epoch)
        #     if cfg.if_save and not cfg.if_test:
        #         torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
        #         self.log.info('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path))

        # # =====ADVERSARIAL TRAINING=====
        self.log.info('Starting Adversarial Training...')
        progress = tqdm(range(cfg.ADV_train_epoch))
        for adv_epoch in progress:
            g_loss = self.adv_train_generator(cfg.ADV_g_step)  # Generator
            d_loss = self.adv_train_discriminator(cfg.ADV_d_epoch, cfg.ADV_d_step)  # Discriminator
            # self.update_temperature(adv_epoch, cfg.ADV_train_epoch)  # update temperature

            progress.set_description(
                'g_loss: %.4f, d_loss: %.4f' % (g_loss, d_loss))

            # TEST
            if adv_epoch % cfg.adv_log_step == 0:
                self.log.info('[ADV] epoch %d: %s' % (
                    adv_epoch, self.cal_metrics(fmt_str=True)))

                if cfg.if_save and not cfg.if_test:
                    self._save('ADV', adv_epoch)

    def _test(self):
        print('>>> Begin test...')

        self._run()

        # test for whole process
        # self.pretrain_generator(cfg.MLE_train_epoch)
        # torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
        # self.log.info('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path))

        # progress = tqdm(range(cfg.ADV_train_epoch))
        # for adv_epoch in progress:
        #     g_loss = self.adv_train_generator(cfg.ADV_g_step)  # Generator
        #     d_loss = self.adv_train_discriminator(cfg.ADV_d_epoch, cfg.ADV_d_step)  # Discriminator
        #     progress.set_description('g_loss: %.4f, d_loss: %.4f' % (g_loss, d_loss))
        #
        #     # TEST
        #     if adv_epoch % cfg.adv_log_step == 0:
        #         self.log.info('[ADV] epoch %d: %s', adv_epoch, self.cal_metrics(fmt_str=True))

        # t0 = time.time()
        # self.gen.sample(cfg.samples_num, cfg.batch_size)
        # t1 = time.time()
        # print('time: ', t1 - t0)
        #
        # t0 = time.time()
        # self.gen.sample(cfg.samples_num, 8 * cfg.batch_size)
        # t1 = time.time()
        # print('time: ', t1 - t0)

        # out = torch.rand(64, 5000).cuda()
        # next_token = torch.LongTensor([0] * 64)
        # t0 = time.time()
        # for _ in range(150 * 20):
        #     # res = self.gen.add_gumbel_u(out)
        #     res = torch.zeros(out.size()).cuda().uniform_(0, 1)
        #     # res = self.gen.add_gumbel_v(out, next_token)
        # t1 = time.time()
        # print('time: ', t1 - t0)

        # t0 = time.time()
        # self.adv_train_discriminator(1, 1)
        # t1 = time.time()
        # print('time: ', t1 - t0)

        # self.adv_train_discriminator(1, 3)
        # self.adv_train_generator(1)
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
            real_samples_onehot = F.one_hot(self.oracle_data.random_batch()['target'], cfg.vocab_size).float()
            gen_samples, gen_samples_onehot, all_pred, all_gumbel_u, all_gumbel_v, all_hardlogQ = self.gen.sample(
                cfg.batch_size, cfg.batch_size, rebar=True)
            # gen_samples, gen_samples_onehot, all_pred, all_gumbel_u, all_gumbel_v, all_hardlogQ = self.gen.sample(
            #     cfg.n_samples * cfg.batch_size, cfg.n_samples * cfg.batch_size, rebar=True)

            if cfg.CUDA:
                real_samples_onehot, gen_samples_onehot = real_samples_onehot.cuda(), gen_samples_onehot.cuda()
                all_gumbel_u, all_gumbel_v, all_hardlogQ = all_gumbel_u.cuda(), all_gumbel_v.cuda(), all_hardlogQ.cuda()

            # =====Train=====
            # Gumbel vanilla
            # d_out_real = self.dis(real_samples_onehot)
            # d_out_fake = self.dis(all_gumbel_u)
            # g_loss, _ = get_losses(d_out_real, d_out_fake, cfg.loss_type)
            # self.optimize(self.gen_adv_opt, g_loss, self.gen)
            # total_loss += g_loss.item()

            # REBAR
            # d_out_real = self.dis(real_samples_onehot)
            d_out_real = None
            d_out_hard_fake = self.dis(gen_samples_onehot)
            d_out_gumu_fake = self.dis(all_gumbel_u)
            d_out_gumv_fake = self.dis(all_gumbel_v)
            # d_out_hard_fake = torch.sum(self.dis(gen_samples_onehot).view(cfg.n_samples, -1), dim=0)
            # d_out_gumu_fake = torch.sum(self.dis(all_gumbel_u).view(cfg.n_samples, -1), dim=0)
            # d_out_gumv_fake = torch.sum(self.dis(all_gumbel_v).view(cfg.n_samples, -1), dim=0)

            f_loss, h_loss, g_loss_hard = self.cal_rebar_loss(d_out_real, d_out_hard_fake, d_out_gumu_fake,
                                                              d_out_gumv_fake,
                                                              all_hardlogQ)
            self.optimize_rebar(self.gen_adv_opt, (f_loss, h_loss), self.gen)

            # total_loss += (f_loss + h_loss).item()
            total_loss += torch.sum(g_loss_hard).item()

            # vanilla
            # d_out_gumu_fake = self.dis(all_gumbel_u)
            # g_target = torch.ones(cfg.batch_size).long().cuda()
            # g_loss = self.cross_entro_cri(d_out_gumu_fake, g_target)
            # self.optimize(self.gen_adv_opt, g_loss, self.gen)
            # total_loss += g_loss.item()

        return total_loss / g_step if g_step != 0 else 0

    def adv_train_discriminator(self, d_epoch, d_step):
        total_loss = []
        # batch version
        # for step in range(d_step):
        #     real_samples_onehot = F.one_hot(self.oracle_data.random_batch()['target'], cfg.vocab_size).float()
        #     # gen_samples_onehot = F.one_hot(self.gen.sample(cfg.batch_size, cfg.batch_size), cfg.vocab_size).float()
        #
        #     # vanilla
        #     gen_samples, gen_samples_onehot, all_pred, all_gumbel_u, all_gumbel_v, all_hardlogQ = self.gen.sample(
        #         cfg.batch_size, cfg.batch_size, rebar=True)
        #     real_target = torch.ones(cfg.batch_size).long()
        #     fake_target = torch.zeros(cfg.batch_size).long()
        #
        #     if cfg.CUDA:
        #         real_samples_onehot, gen_samples_onehot = real_samples_onehot.cuda(), gen_samples_onehot.cuda()
        #         all_gumbel_u = all_gumbel_u.cuda()
        #         real_target, fake_target = real_target.cuda(), fake_target.cuda()
        #
        #     # =====Train=====
        #     # REBAR
        #     # d_out_real = self.dis(real_samples_onehot)
        #     # d_out_fake = self.dis(gen_samples_onehot)
        #     # _, d_loss = get_losses(d_out_real, d_out_fake, cfg.loss_type)
        #
        #     # vanilla
        #     d_out_real = self.dis(real_samples_onehot)
        #     d_out_fake = self.dis(all_gumbel_u)
        #     d_real_loss = self.cross_entro_cri(d_out_real, real_target)
        #     d_fake_loss = self.cross_entro_cri(d_out_fake, fake_target)
        #     d_loss = d_real_loss + d_fake_loss
        #
        #     self.optimize(self.dis_opt, d_loss, self.dis)
        #     total_loss.append(d_loss.item())

        # entire data version
        for epoch in range(d_epoch):
            # real_samples = F.one_hot(self.oracle_samples.clone(), cfg.vocab_size).float()
            # gen_samples = F.one_hot(self.gen.sample(cfg.samples_num, 8 * cfg.batch_size), cfg.vocab_size).float()
            real_samples = self.oracle_samples.clone()
            gen_samples = self.gen.sample(cfg.samples_num, 8 * cfg.batch_size)

            for step in range(d_step):
                # shuffle
                perm = torch.randperm(cfg.samples_num)
                real_samples = real_samples[perm]
                gen_samples = gen_samples[perm]
                for i in range(cfg.batch_size, cfg.samples_num, cfg.batch_size):
                    # real_data = real_samples[i - cfg.batch_size:i]
                    # fake_data = gen_samples[i - cfg.batch_size:i]
                    real_data = F.one_hot(real_samples[i - cfg.batch_size:i], cfg.vocab_size).float()
                    fake_data = F.one_hot(gen_samples[i - cfg.batch_size:i], cfg.vocab_size).float()
                    real_target = torch.ones(cfg.batch_size).long()
                    fake_target = torch.zeros(cfg.batch_size).long()
                    if cfg.CUDA:
                        real_data, fake_data = real_data.cuda(), fake_data.cuda()
                        real_target, fake_target = real_target.cuda(), fake_target.cuda()

                    # d_out_real = self.dis(real_data)
                    # d_out_fake = self.dis(fake_data)
                    # _, d_loss = get_losses(d_out_real, d_out_fake, cfg.loss_type)
                    d_out_real = self.dis(real_data)
                    d_out_fake = self.dis(fake_data)
                    d_real_loss = self.cross_entro_cri(d_out_real, real_target)
                    d_fake_loss = self.cross_entro_cri(d_out_fake, fake_target)
                    d_loss = d_real_loss + d_fake_loss

                    self.optimize(self.dis_opt, d_loss, self.dis)
                    total_loss.append(d_loss.item())

        if d_epoch == 0 or d_step == 0:
            return 0
        return np.mean(total_loss)

    def update_temperature(self, i, N):
        self.gen.temperature = get_fixed_temperature(cfg.temperature, i, N, cfg.temp_adpt)

    def cal_rebar_loss(self, d_out_real, d_out_hard_fake, d_out_gumu_fake, d_out_gumv_fake, hardlogQ):
        hardlogQ = torch.sum(hardlogQ, dim=1)

        # RSGAN
        # g_loss_hard, _ = get_losses(d_out_real, d_out_hard_fake, cfg.loss_type)
        # g_loss_gumu, _ = get_losses(d_out_real, d_out_gumu_fake, cfg.loss_type)
        # g_loss_gumv, _ = get_losses(d_out_real, d_out_gumv_fake, cfg.loss_type)

        # Cross Entropy
        g_target = torch.ones(d_out_hard_fake.size(0)).long().cuda()
        g_loss_hard = self.cross_entro_cri(d_out_hard_fake, g_target)
        g_loss_gumu = self.cross_entro_cri(d_out_gumu_fake, g_target)
        g_loss_gumv = self.cross_entro_cri(d_out_gumv_fake, g_target)

        # Reverse Cross Entropy
        # g_target = torch.zeros(d_out_hard_fake.size(0)).long().cuda()
        # g_loss_hard = -0.5 * self.cross_entro_cri(d_out_hard_fake, g_target)
        # g_loss_gumu = -0.5 * self.cross_entro_cri(d_out_gumu_fake, g_target)
        # g_loss_gumv = -0.5 * self.cross_entro_cri(d_out_gumv_fake, g_target)

        # use prob
        # g_loss_hard = -F.softmax(d_out_hard_fake, dim=-1)[:, 1]
        # g_loss_gumu = -F.softmax(d_out_hard_fake, dim=-1)[:, 1]
        # g_loss_gumv = -F.softmax(d_out_hard_fake, dim=-1)[:, 1]

        g_loss_hard = -g_loss_hard
        g_loss_gumu = -g_loss_gumu
        g_loss_gumv = -g_loss_gumv

        de_loss_hard = g_loss_hard.detach()
        de_loss_gumv = g_loss_gumv.detach()

        f_loss = - torch.mean(de_loss_hard * hardlogQ)
        h_loss = self.theta * torch.mean(de_loss_gumv * hardlogQ - g_loss_gumu + g_loss_gumv)

        return f_loss, h_loss, -g_loss_hard

    @staticmethod
    def optimize_rebar(opt, loss, model):
        """Add clip_grad_norm_"""
        f, h = loss
        opt.zero_grad()
        f_plus_h = []
        f_grad = torch.autograd.grad(f, list(model.parameters()), create_graph=True)
        h_grad = torch.autograd.grad(h, list(model.parameters()))

        for a, b in zip(f_grad, h_grad):
            flag_a = torch.sum(torch.isnan(a))
            flag_b = torch.sum(torch.isnan(b))
            if flag_a > 0:
                f_plus_h.append(b)
            elif flag_b > 0:
                f_plus_h.append(a)
            else:
                f_plus_h.append(a + b)

        for i, param in enumerate(model.parameters()):
            param.grad = f_plus_h[i]

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()

    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False):
        """Add clip_grad_norm_"""
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()
