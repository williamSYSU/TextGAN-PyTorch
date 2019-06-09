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
from models.CatGAN_D import CatGAN_C
from models.CatGAN_G import CatGAN_G
from models.Oracle import Oracle
from utils.cat_data_loader import CatGenDataIter, CatClasDataIter
from utils.data_loader import GenDataIter
from utils.data_utils import create_multi_oracle
from utils.helpers import get_fixed_temperature
from utils.text_process import write_tensor


class CatGANInstructor(BasicInstructor):
    """===Version 3==="""

    def __init__(self, opt):
        super(CatGANInstructor, self).__init__(opt)

        # self.log = create_logger(__name__, silent=False, to_disk=True, log_file=cfg.log_filename)

        # generator, discriminator
        self.oracle_list = [Oracle(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                                   cfg.padding_idx, gpu=cfg.CUDA) for _ in range(cfg.k_label)]

        self.gen = CatGAN_G(cfg.k_label, cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim,
                            cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        self.dis = CatGAN_C(cfg.k_label, cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.vocab_size,
                            cfg.padding_idx, gpu=cfg.CUDA)
        self.clas = self.dis

        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_adv_lr)

        dis_params, clas_params = self.dis.split_params()
        self.dis_opt = optim.Adam(dis_params, lr=cfg.dis_lr)
        self.clas_opt = optim.Adam(clas_params, lr=cfg.clas_lr)

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

    def _run(self):
        # ===Pre-train Generator===
        if not cfg.gen_pretrain:
            self.log.info('Starting Generator MLE Training...')
            self.pretrain_generator(cfg.MLE_train_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
                print('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path))

        # ===Pre-train Classifier===
        self.train_classifier(cfg.PRE_clas_epoch, 'PRE')

        self.log.info('Initial metrics: %s', self.comb_metrics(fmt_str=True))
        # ===Adv-train===
        progress = tqdm(range(cfg.ADV_train_epoch))
        for adv_epoch in progress:
            g_loss, gd_loss, gc_loss, gc_acc = self.adv_train_generator(cfg.ADV_g_step)
            # d_loss = self.adv_train_discriminator(cfg.ADV_d_step) # !!! no adv-train for discriminator
            c_loss, c_acc = self.train_classifier(cfg.ADV_d_step, 'ADV')

            # =====Test=====
            # self.log.info(
            #     '[ADV] epoch %d: g_loss = %.4f, c_loss = %.4f, c_acc = %.4f, d_loss = %.4f' % (
            #         adv_epoch, g_loss, c_loss, c_acc, d_loss))
            # self.log.info(
            #     '[ADV] epoch %d: g_loss = %.4f, gd_loss = %.4f, gc_loss = %.4f, gc_acc = %.4f, c_loss = %.4f, c_acc = %.4f,' % (
            #         adv_epoch, g_loss, gd_loss, gc_loss, gc_acc, c_loss, c_acc))
            progress.set_description('g_loss = %.4f, c_loss = %.4f' % (g_loss, c_loss))
            if adv_epoch % cfg.adv_log_step == 0:
                self.log.info(
                    '[ADV] epoch %d : %s' % (adv_epoch, self.comb_metrics(fmt_str=True)))
                if not cfg.if_test and cfg.if_save:
                    for label_i in range(cfg.k_label):
                        self._save('ADV', adv_epoch, label_i)

    def _test(self):
        self.log.debug('>>> Begin test...')

        self._run()
        # self.train_classifier(1, 'PRE')
        # self.train_classifier(1, 'ADV')
        # self.adv_train_discriminator(1)
        # self.adv_train_generator(1)
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

    def train_classifier(self, c_step, phase):
        """真假样本一起训练，为了让分类器不那么强"""
        self.clas.dis_or_clas = 'clas'  # !!!!!
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
            if phase == 'PRE':
                self.log.info('[%s-CLAS] epoch: %d, c_loss = %.4f, c_acc = %.4f' % (phase, epoch, c_loss, c_acc))
        self.clas.dis_or_clas = None
        if c_step == 0:
            return 0, 0
        return np.mean(total_loss), np.mean(total_acc)

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
            self.dis.dis_or_clas = 'dis'  # !!!!!
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
            self.dis.dis_or_clas = None

            # Classifier loss, only input fake data
            # clas_samples_list = [fake for fake in gen_samples_list]
            self.clas.dis_or_clas = 'clas'  # !!!!!
            inp = torch.cat(gen_samples_list, dim=0)
            target = torch.zeros(inp.size(0)).long()
            for idx in range(1, len(gen_samples_list)):
                start = sum([gen_samples_list[i].size(0) for i in range(idx)])
                target[start: start + gen_samples_list[idx].size(0)] = idx
            if cfg.CUDA:
                inp, target = inp.cuda(), target.cuda()
            pred = self.clas(inp)
            gc_loss = self.clas_criterion(pred, target)
            gc_acc = torch.sum((pred.argmax(dim=-1) == target)).item() / inp.size(0)
            self.clas.dis_or_clas = None

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

    def adv_train_discriminator(self, d_step):
        self.dis.dis_or_clas = 'dis'  # !!!!!
        total_loss = []
        for step in range(d_step):
            # real_samples = torch.cat([F.one_hot(self.all_oracle_data.random_batch()['target'], cfg.vocab_size).float()
            #                           for _ in range(cfg.k_label)], dim=0)
            real_samples = torch.cat(
                [F.one_hot(self.oracle_data_list[i].random_batch()['target'], cfg.vocab_size).float()
                 for i in range(cfg.k_label)], dim=0)
            gen_samples = torch.cat([self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True, label_i=i)
                                     for i in range(cfg.k_label)], dim=0)
            # shuffle
            real_samples = real_samples[torch.randperm(real_samples.size(0))].detach()
            gen_samples = gen_samples[torch.randperm(gen_samples.size(0))].detach()
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()

            # =====Train=====
            d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
            d_loss = self.dis_criterion(d_out_real - d_out_fake, torch.ones_like(d_out_real))
            g_loss = self.dis_criterion(d_out_fake - d_out_real, torch.ones_like(d_out_fake))

            self.optimize(self.dis_opt, d_loss, self.dis)
            total_loss.append(d_loss.item())
        self.log.debug('In D: g_loss = %.4f' % g_loss.item())
        self.dis.dis_or_clas = None
        if d_step == 0:
            return 0
        return np.mean(total_loss)

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
        for label_i in range(cfg.k_label):
            o_nll, g_nll, s_nll = self.cal_metrics(label_i)
            oracle_nll.append(float('%.4f' % o_nll))
            gen_nll.append(float('%.4f' % g_nll))
            self_nll.append(float('%.4f' % s_nll))

        if fmt_str:
            return 'oracle_NLL = %s, gen_NLL = %s, self_NLL = %s,' % (oracle_nll, gen_nll, self_nll)
        return oracle_nll, gen_nll, self_nll

    def _save(self, phase, epoch, label_i=None):
        assert type(label_i) == int
        torch.save(self.gen.state_dict(), cfg.save_model_root + 'gen_{}_{:05d}.pt'.format(phase, epoch))
        save_sample_path = cfg.save_samples_root + 'samples_c{}_{}_{:05d}.txt'.format(label_i, phase, epoch)
        samples = self.gen.sample(cfg.batch_size, cfg.batch_size, label_i=label_i)
        write_tensor(save_sample_path, samples)
