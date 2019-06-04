# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : catgan_instructor.py
# @Time         : Created at 2019-05-28
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import config as cfg
from instructor.oracle_data.instructor import BasicInstructor
from models.CatGAN_D import CatGAN_D, CatGAN_C
from models.CatGAN_G import CatGAN_G
from models.Oracle import Oracle
from utils.cat_data_loader import CatGenDataIter, CatClasDataIter
from utils.data_loader import GenDataIter
from utils.data_utils import create_multi_oracle
from utils.helpers import get_fixed_temperature
from utils.text_process import write_tensor


class CatGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(CatGANInstructor, self).__init__(opt)

        # generator, discriminator
        self.oracle_list = [Oracle(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                                   cfg.padding_idx, gpu=cfg.CUDA) for _ in range(cfg.k_label)]

        self.gen = CatGAN_G(cfg.k_label, cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim,
                            cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        self.dis = CatGAN_D(cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.vocab_size, cfg.padding_idx,
                            gpu=cfg.CUDA)
        self.clas = CatGAN_C(cfg.k_label, cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.vocab_size,
                             cfg.padding_idx, gpu=cfg.CUDA)
        self.clas_eval = CatGAN_C(cfg.k_label, cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.vocab_size,
                                  cfg.padding_idx, gpu=cfg.CUDA)

        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_adv_lr)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)
        self.clas_opt = optim.Adam(self.clas.parameters(), lr=cfg.clas_lr)
        self.clas_eval_opt = optim.Adam(self.clas_eval.parameters(), lr=cfg.clas_lr)

        # Criterion
        self.mle_criterion = nn.NLLLoss()
        self.dis_criterion = nn.BCEWithLogitsLoss()
        self.clas_criterion = nn.CrossEntropyLoss()

        # DataLoader
        self.oracle_samples_list = [torch.load(cfg.multi_oracle_samples_path.format(i, cfg.samples_num))
                                    for i in range(cfg.k_label)]
        self.oracle_data_list = [GenDataIter(self.oracle_samples_list[i]) for i in range(cfg.k_label)]
        self.all_oracle_data = CatGenDataIter(self.oracle_samples_list)  # Shuffled all oracle data
        self.gen_data_list = [GenDataIter(self.gen.sample(cfg.batch_size, cfg.batch_size, label_i=i))
                              for i in range(cfg.k_label)]
        self.clas_data = CatClasDataIter(self.oracle_samples_list)  # fake init (reset during training)

    # >>>Version 1
    # def _run(self):
    #     # ===Pre-train===
    #     if not cfg.gen_pretrain:
    #         self.log.info('Starting Generator MLE Training...')
    #         self.pretrain_generator(cfg.MLE_train_epoch)
    #         if cfg.if_save and not cfg.if_test:
    #             torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
    #             self.log.info('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path))
    #
    #     self.log.info('Staring pre-train classifier...')
    #     # self.train_classifier('PRE', 50)
    #     self.train_classifier_eval(10)
    #
    #     # ===Adv-train===
    #     progress = tqdm(range(cfg.ADV_train_epoch))
    #     for adv_epoch in range(cfg.ADV_train_epoch):
    #         g_loss, gd_loss, gc_loss, gc_acc = self.adv_train_generator(cfg.ADV_g_step)
    #
    #         d_loss = self.adv_train_discriminator(cfg.ADV_d_step)
    #         c_loss, c_acc = self.train_classifier('ADV', cfg.ADV_d_step)
    #         # c_loss, c_acc = 0, 0
    #
    #         # =====Test=====
    #         # Eval classifier
    #         self.clas_data.reset([self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True, label_i=i)
    #                               for i in range(cfg.k_label)])
    #         eval_loss, eval_acc = self.eval_dis(self.clas_eval, self.clas_data.loader, self.clas_criterion)
    #         self.log.info(
    #             '[ADV] epoch %d: g_loss = %.4f, gd_loss = %.4f, gc_loss = %.4f, gc_acc = %.4f, d_loss = %.4f, c_loss = %.4f, c_acc = %.4f, eval_loss = %.4f, eval_acc = %.4f,' % (
    #                 adv_epoch, g_loss, gd_loss, gc_loss, gc_acc, d_loss, c_loss, c_acc, eval_loss, eval_acc))
    #         if adv_epoch % cfg.adv_log_step == 0:
    #             self.log.info(
    #                 '[ADV] epoch %d : %s' % (adv_epoch, self.comb_metrics(fmt_str=True)))

    # >>>Version 2
    def _run(self):
        # ===Pre-train===
        self.train_classifier(10)

        # ===Adv-train===
        progress = tqdm(range(cfg.ADV_train_epoch))
        for adv_epoch in range(cfg.ADV_train_epoch):
            g_loss, gd_loss, gc_loss, gc_acc = self.adv_train_generator(cfg.ADV_g_step)

            d_loss = self.adv_train_discriminator(cfg.ADV_d_step)
            c_loss, c_acc = self.train_classifier('ADV', cfg.ADV_d_step)
            # c_loss, c_acc = 0, 0

            # =====Test=====
            # Eval classifier
            self.clas_data.reset([self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True, label_i=i)
                                  for i in range(cfg.k_label)])
            eval_loss, eval_acc = self.eval_dis(self.clas_eval, self.clas_data.loader, self.clas_criterion)
            self.log.info(
                '[ADV] epoch %d: g_loss = %.4f, gd_loss = %.4f, gc_loss = %.4f, gc_acc = %.4f, d_loss = %.4f, c_loss = %.4f, c_acc = %.4f, eval_loss = %.4f, eval_acc = %.4f,' % (
                    adv_epoch, g_loss, gd_loss, gc_loss, gc_acc, d_loss, c_loss, c_acc, eval_loss, eval_acc))
            if adv_epoch % cfg.adv_log_step == 0:
                self.log.info(
                    '[ADV] epoch %d : %s' % (adv_epoch, self.comb_metrics(fmt_str=True)))

    def _test(self):
        self.log.debug('>>> Begin test...')

        # self._run()
        # self.train_classifier('PRE', 1)
        # self.train_classifier('PRE',50)
        # self.log.debug(self.adv_train_generator(1))
        # self.cal_metrics(0)

        # self.train_classifier_eval(10)

        # self.clas_data.reset([self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True, label_i=i)
        #                       for i in range(cfg.k_label)])
        # eval_loss, eval_acc = self.eval_dis(self.clas_eval, self.clas_data.loader, self.clas_criterion)

        # self.gen_data_list[0].reset(self.gen.sample(cfg.samples_num, 4 * cfg.batch_size, label_i=0))
        # oracle_nll = self.eval_gen(self.oracle_list[1],
        #                            self.gen_data_list[0].loader,
        #                            self.mle_criterion, 1)
        # self.log.debug(oracle_nll)
        pass

    def pretrain_generator(self, epochs):
        """
        Max Likelihood Pre-training for the generator
        """
        for epoch in range(epochs):
            # =====Train=====
            pre_loss = self.train_gen_epoch(self.gen, self.all_oracle_data.loader, self.mle_criterion, self.gen_opt)

            # =====Test=====
            if epoch % cfg.pre_log_step == 0 or epoch == epochs - 1:
                self.log.info(
                    '[MLE-GEN] epoch %d : pre_loss = %.4f, %s' % (
                        epoch, pre_loss, self.comb_metrics(fmt_str=True)))

                if not cfg.if_test and cfg.if_save:
                    for label_i in range(cfg.k_label):
                        self._save('MLE', epoch, label_i)

    # >>>Version 1
    # def train_classifier(self, phrase, c_step):
    #     """真假样本一起训练，为了让分类器不那么强"""
    #     total_loss = []
    #     total_acc = []
    #     for epoch in range(c_step):
    #         clas_samples_list = []
    #         for i in range(cfg.k_label):
    #             real_samples = F.one_hot(self.oracle_data_list[i].random_batch()['target'],
    #                                      cfg.vocab_size).float()
    #             gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True, label_i=i).cpu()
    #             clas_samples_list.append(torch.cat((real_samples, gen_samples), dim=0))
    #             # clas_samples_list.append(real_samples)
    #         self.clas_data.reset(clas_samples_list)
    #
    #         # =====Train=====
    #         c_loss, c_acc = self.train_dis_epoch(self.clas, self.clas_data.loader, self.clas_criterion, self.clas_opt)
    #
    #         total_loss.append(c_loss)
    #         total_acc.append(c_acc)
    #         if phrase == 'PRE':
    #             self.log.info('[%s-CLAS] epoch: %d, c_loss = %.4f, c_acc = %.4f' % (phrase, epoch, c_loss, c_acc))
    #     if c_step == 0:
    #         return 0, 0
    #     return np.mean(total_loss), np.mean(total_acc)

    # >>>Version 2

    # >>>Version 2
    def train_classifier(self, phrase, c_step):
        """假的为一类，真的分成k类"""
        total_loss = []
        total_acc = []
        for epoch in range(c_step):
            clas_samples_list = []
            for i in range(cfg.k_label):
                real_samples = F.one_hot(self.oracle_data_list[i].random_batch()['target'],
                                         cfg.vocab_size).float()
                gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True, label_i=i).cpu()
                clas_samples_list.append(torch.cat((real_samples, gen_samples), dim=0))
                # clas_samples_list.append(real_samples)
            self.clas_data.reset(clas_samples_list)

            # =====Train=====
            c_loss, c_acc = self.train_dis_epoch(self.clas, self.clas_data.loader, self.clas_criterion, self.clas_opt)

            total_loss.append(c_loss)
            total_acc.append(c_acc)
            if phrase == 'PRE':
                self.log.info('[%s-CLAS] epoch: %d, c_loss = %.4f, c_acc = %.4f' % (phrase, epoch, c_loss, c_acc))
        if c_step == 0:
            return 0, 0
        return np.mean(total_loss), np.mean(total_acc)

    # def train_classifier_eval(self, c_step):
    #     """只训练真实样本，用来验证生成器的样本是否分为两类。放全部真实样本进去训练。"""
    #     total_loss = []
    #     total_acc = []
    #     for epoch in range(c_step):
    #         clas_samples_list = []
    #         for i in range(cfg.k_label):
    #             real_samples = F.one_hot(self.oracle_data_list[i].input, cfg.vocab_size).float()
    #             clas_samples_list.append(real_samples)
    #         self.clas_data.reset(clas_samples_list)
    #
    #         # =====Train=====
    #         c_loss, c_acc = self.train_dis_epoch(self.clas_eval, self.clas_data.loader, self.clas_criterion,
    #                                              self.clas_eval_opt)
    #
    #         total_loss.append(c_loss)
    #         total_acc.append(c_acc)
    #         self.log.info('[CLAS-EVAL] epoch: %d, c_loss = %.4f, c_acc = %.4f' % (epoch, c_loss, c_acc))
    #     if c_step == 0:
    #         return 0, 0
    #     return np.mean(total_loss), np.mean(total_acc)

    # >>>Version 1
    # def adv_train_generator(self, g_step):
    #     total_g_loss = []
    #     total_gd_loss = []
    #     total_gc_loss = []
    #     total_gc_acc = []
    #     for step in range(g_step):
    #         real_samples_list = [F.one_hot(self.oracle_data_list[i].random_batch()['target'], cfg.vocab_size).float()
    #                              for i in range(cfg.k_label)]
    #         gen_samples_list = [self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True, label_i=i)
    #                             for i in range(cfg.k_label)]
    #
    #         # =====Train=====
    #         # Discriminator loss, input real and fake data
    #         dis_real_samples = torch.cat(real_samples_list, dim=0)
    #         dis_gen_samples = torch.cat(gen_samples_list, dim=0)
    #
    #         # shuffle
    #         dis_real_samples = dis_real_samples[torch.randperm(dis_real_samples.size(0))]
    #         dis_gen_samples = dis_gen_samples[torch.randperm(dis_gen_samples.size(0))]
    #         if cfg.CUDA:
    #             dis_real_samples, dis_gen_samples = dis_real_samples.cuda(), dis_gen_samples.cuda()
    #         d_out_real = self.dis(dis_real_samples)
    #         d_out_fake = self.dis(dis_gen_samples)
    #         gd_loss = self.dis_criterion(d_out_fake - d_out_real, torch.ones_like(d_out_fake))
    #
    #         # Classifier loss, only input fake data
    #         # clas_samples_list = [fake for fake in gen_samples_list]
    #         clas_samples_list = gen_samples_list
    #         self.clas_data.reset(clas_samples_list)
    #         gc_loss = 0
    #         gc_acc = 0
    #         gc_num = 0
    #         for i, data in enumerate(self.clas_data.loader):
    #             inp, target = data['input'], data['target']
    #             if cfg.CUDA:
    #                 inp, target = inp.cuda(), target.cuda()
    #             # pred = self.clas.forward(inp)
    #             pred = self.clas.forward(inp)
    #             loss = self.clas_criterion(pred, target)
    #             gc_loss += loss
    #             gc_acc += torch.sum((pred.argmax(dim=-1) == target)).item()
    #             gc_num += inp.size(0)
    #         gc_acc /= gc_num
    #
    #         # Total loss
    #         g_loss = gd_loss + gc_loss
    #
    #         self.optimize(self.gen_adv_opt, g_loss, self.gen)
    #         # self.optimize(self.gen_adv_opt, gc_loss, self.gen, retain_graph=True)
    #         # self.optimize(self.gen_adv_opt, gd_loss, self.gen, retain_graph=True)
    #         total_g_loss.append(g_loss.item())
    #         total_gd_loss.append(gd_loss.item())
    #         total_gc_loss.append(gc_loss.item())
    #         total_gc_acc.append(gc_acc)
    #
    #         # self.log.debug('In G: g_loss = %.4f' % g_loss.item())
    #
    #     if g_step == 0:
    #         return 0, 0, 0, 0
    #     return np.mean(total_g_loss), np.mean(total_gd_loss), np.mean(total_gc_loss), np.mean(total_gc_acc)

    # >>>Version 2
    def adv_train_generator(self, g_step):
        total_g_loss = []
        total_gd_loss = []
        total_gc_loss = []
        total_gc_acc = []
        for step in range(g_step):
            real_samples_list = [F.one_hot(self.oracle_data_list[i].random_batch()['target'], cfg.vocab_size).float()
                                 for i in range(cfg.k_label)]
            gen_samples_list = [self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True, label_i=i)
                                for i in range(cfg.k_label)]

            # =====Train=====
            # Discriminator loss, input real and fake data
            dis_real_samples = torch.cat(real_samples_list, dim=0)
            dis_gen_samples = torch.cat(gen_samples_list, dim=0)

            # shuffle
            dis_real_samples = dis_real_samples[torch.randperm(dis_real_samples.size(0))]
            dis_gen_samples = dis_gen_samples[torch.randperm(dis_gen_samples.size(0))]
            if cfg.CUDA:
                dis_real_samples, dis_gen_samples = dis_real_samples.cuda(), dis_gen_samples.cuda()
            d_out_real = self.dis(dis_real_samples)
            d_out_fake = self.dis(dis_gen_samples)
            gd_loss = self.dis_criterion(d_out_fake - d_out_real, torch.ones_like(d_out_fake))

            # Classifier loss, only input fake data
            # clas_samples_list = [fake for fake in gen_samples_list]
            clas_samples_list = gen_samples_list
            self.clas_data.reset(clas_samples_list)
            gc_loss = 0
            gc_acc = 0
            gc_num = 0
            for i, data in enumerate(self.clas_data.loader):
                inp, target = data['input'], data['target']
                if cfg.CUDA:
                    inp, target = inp.cuda(), target.cuda()
                # pred = self.clas.forward(inp)
                pred = self.clas.forward(inp)
                loss = self.clas_criterion(pred, target)
                gc_loss += loss
                gc_acc += torch.sum((pred.argmax(dim=-1) == target)).item()
                gc_num += inp.size(0)
            gc_acc /= gc_num

            # Total loss
            g_loss = gd_loss + gc_loss

            self.optimize(self.gen_adv_opt, g_loss, self.gen)
            # self.optimize(self.gen_adv_opt, gc_loss, self.gen, retain_graph=True)
            # self.optimize(self.gen_adv_opt, gd_loss, self.gen, retain_graph=True)
            total_g_loss.append(g_loss.item())
            total_gd_loss.append(gd_loss.item())
            total_gc_loss.append(gc_loss.item())
            total_gc_acc.append(gc_acc)

            # self.log.debug('In G: g_loss = %.4f' % g_loss.item())

        if g_step == 0:
            return 0, 0, 0, 0
        return np.mean(total_g_loss), np.mean(total_gd_loss), np.mean(total_gc_loss), np.mean(total_gc_acc)

    # def adv_train_discriminator(self, d_step):
    #     total_loss = []
    #     for step in range(d_step):
    #         # real_samples = torch.cat([F.one_hot(self.all_oracle_data.random_batch()['target'], cfg.vocab_size).float()
    #         #                           for _ in range(cfg.k_label)], dim=0)
    #         real_samples = torch.cat(
    #             [F.one_hot(self.oracle_data_list[i].random_batch()['target'], cfg.vocab_size).float()
    #              for i in range(cfg.k_label)], dim=0)
    #         gen_samples = torch.cat([self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True, label_i=i)
    #                                  for i in range(cfg.k_label)], dim=0)
    #         # shuffle
    #         real_samples = real_samples[torch.randperm(real_samples.size(0))]
    #         gen_samples = gen_samples[torch.randperm(gen_samples.size(0))]
    #         if cfg.CUDA:
    #             real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
    #
    #         # =====Train=====
    #         d_out_real = self.dis(real_samples)
    #         d_out_fake = self.dis(gen_samples)
    #         d_loss = self.dis_criterion(d_out_real - d_out_fake, torch.ones_like(d_out_real))
    #         g_loss = self.dis_criterion(d_out_fake - d_out_real, torch.ones_like(d_out_fake))
    #
    #         self.optimize(self.dis_opt, d_loss, self.dis)
    #         total_loss.append(d_loss.item())
    #
    #     self.log.debug('In D: g_loss = %.4f' % g_loss.item())
    #     if d_step == 0:
    #         return 0
    #     return np.mean(total_loss)

    def init_model(self):
        if cfg.oracle_pretrain:
            for i in range(cfg.k_label):
                oracle_path = cfg.multi_oracle_state_dict_path.format(i)
                if not os.path.exists(oracle_path):
                    create_multi_oracle(cfg.k_label)
                self.oracle_list[i].load_state_dict(torch.load(oracle_path))

        if cfg.gen_pretrain:
            self.log.info('Load MLE pretrained generator gen: {}'.format(cfg.pretrained_gen_path))
            self.gen.load_state_dict(torch.load(cfg.pretrained_gen_path))

        if cfg.CUDA:
            for i in range(cfg.k_label):
                self.oracle_list[i] = self.oracle_list[i].cuda()
            self.gen = self.gen.cuda()
            self.dis = self.dis.cuda()
            self.clas = self.clas.cuda()
            self.clas_eval = self.clas_eval.cuda()

    def update_temperature(self, i, N):
        self.gen.temperature = get_fixed_temperature(cfg.temperature, i, N, cfg.temp_adpt)

    def train_gen_epoch(self, model, data_loader, criterion, optimizer):
        total_loss = 0
        for i, data in enumerate(data_loader):
            inp, target, label = data['input'], data['target'], data['label']
            if cfg.CUDA:
                inp, target, label = inp.cuda(), target.cuda(), label.cuda()

            hidden = model.init_hidden(data_loader.batch_size)
            pred = model.forward(inp, hidden, label)
            loss = criterion(pred, target.view(-1))
            self.optimize(optimizer, loss, model)
            total_loss += loss.item()
        return total_loss / len(data_loader)

    @staticmethod
    def eval_gen(model, data_loader, criterion, label_i=None):
        assert type(label_i) == int, 'missing label'
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data['input'], data['target']
                label = torch.LongTensor([label_i] * data_loader.batch_size)
                if cfg.CUDA:
                    inp, target, label = inp.cuda(), target.cuda(), label.cuda()

                hidden = model.init_hidden(data_loader.batch_size)
                if model.name == 'oracle':
                    pred = model.forward(inp, hidden)
                else:
                    pred = model.forward(inp, hidden, label)
                loss = criterion(pred, target.view(-1))
                total_loss += loss.item()
        return total_loss / len(data_loader)

    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False):
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        # if model is not None:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()

    def cal_metrics(self, label_i=None):
        assert type(label_i) == int, 'missing label'
        self.gen_data_list[label_i].reset(self.gen.sample(cfg.samples_num, 4 * cfg.batch_size, label_i=label_i))
        oracle_nll = self.eval_gen(self.oracle_list[label_i],
                                   self.gen_data_list[label_i].loader,
                                   self.mle_criterion, label_i)
        gen_nll = self.eval_gen(self.gen,
                                self.oracle_data_list[label_i].loader,
                                self.mle_criterion, label_i)
        self_nll = self.eval_gen(self.gen,
                                 self.gen_data_list[label_i].loader,
                                 self.mle_criterion, label_i)

        return oracle_nll, gen_nll, self_nll

    def comb_metrics(self, fmt_str=False):
        oracle_nll, gen_nll, self_nll = [], [], []
        for gen_idx in range(cfg.k_label):
            o_nll, g_nll, s_nll = self.cal_metrics(gen_idx)
            oracle_nll.append(float('%.4f' % o_nll))
            gen_nll.append(float('%.4f' % g_nll))
            self_nll.append(float('%.4f' % s_nll))

        if fmt_str:
            return 'oracle_NLL = %s, gen_NLL = %s, self_NLL = %s,' % (oracle_nll, gen_nll, self_nll)
        return oracle_nll, gen_nll, self_nll

    def _save(self, phrase, epoch, label_i=None):
        assert type(label_i) == int
        torch.save(self.gen.state_dict(), cfg.save_model_root + 'gen_{}_{:05d}.pt'.format(phrase, epoch))
        save_sample_path = cfg.save_samples_root + 'samples_c{}_{}_{:05d}.txt'.format(label_i + 1, phrase, epoch)
        samples = self.gen.sample(cfg.batch_size, cfg.batch_size, label_i=label_i)
        write_tensor(save_sample_path, samples)
