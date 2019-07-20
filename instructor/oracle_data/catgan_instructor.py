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
        self.desp_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)

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
        # if not cfg.clas_pretrain:
        #     self.train_classifier(cfg.PRE_clas_epoch, 'PRE')
        #     if cfg.if_save:
        #         torch.save(self.clas.state_dict(), cfg.pretrained_clas_path)
        #         print('Save pre-trained classifier: {}'.format(cfg.pretrained_clas_path))
        # self.adv_train_discriminator(5)
        # self.adv_train_descriptor(50)

        self.log.info('Initial metrics: %s', self.comb_metrics(fmt_str=True))
        self.freeze_dis = True
        self.freeze_clas = False
        # ===Adv-train===
        progress = tqdm(range(cfg.ADV_train_epoch))
        for adv_epoch in progress:
            g_loss, gd_loss, gc_loss, gc_acc = self.adv_train_generator(cfg.ADV_g_step)
            # d_loss = self.adv_train_discriminator(cfg.ADV_d_step, 'ADV')  # !!! no adv-train for discriminator
            # c_loss, c_acc = self.train_classifier(cfg.ADV_d_step, 'ADV')
            d_loss, dd_loss, dc_loss = self.adv_train_descriptor(cfg.ADV_d_step, 'ADV')

            self.update_temperature(adv_epoch, cfg.ADV_train_epoch)

            # =====Test=====
            # progress.set_description('g_loss = %.4f, c_loss = %.4f' % (g_loss, c_loss))
            progress.set_description('g_loss = %.4f, d_loss = %.4f, dd_loss = %.4f' % (g_loss, d_loss, dd_loss))
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
        # self.adv_train_descriptor(1)

        # >>>>>>>>>>>Test adversarial training Discriminator
        # self.freeze_dis = True
        # self.freeze_clas = False
        # # self.train_classifier(150, 'PRE')
        # # self.adv_train_discriminator(50,'PRE')
        # # self.adv_train_descriptor(150, 'PRE')
        #
        # progress = tqdm(range(cfg.ADV_train_epoch))
        # for adv_epoch in progress:
        #     g_loss, gd_loss, gc_loss, gc_acc = self.adv_train_generator(cfg.ADV_g_step)
        #     # d_loss = self.adv_train_discriminator(cfg.ADV_d_step, 'ADV')
        #     d_loss, dd_loss, dc_loss = self.adv_train_descriptor(cfg.ADV_d_step, 'ADV')
        #
        #     # =====Test=====
        #     # progress.set_description(
        #         # 'g_loss = %.4f, d_loss = %.4f' % (g_loss, d_loss))
        #     progress.set_description(
        #         'g_loss = %.4f, d_loss = %.4f, dd_loss = %.4f, dc_loss = %.4f' % (
        #             g_loss, d_loss, dd_loss, dc_loss))
        #     if adv_epoch % cfg.adv_log_step == 0:
        #         self.log.info('[ADV] epoch %d : %s' % (adv_epoch, self.comb_metrics(fmt_str=True)))

        # self.train_classifier(150, 'PRE')
        # count = 0
        #
        # while True:
        #     self.oracle = Oracle(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size,
        #                          cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        #     if cfg.CUDA:
        #         self.oracle = self.oracle.cuda()
        #     self.oracle_data.reset(self.oracle.sample(cfg.samples_num, 8 * cfg.batch_size))
        #     gt = self.eval_gen(self.oracle, self.oracle_data.loader, self.mle_criterion, 0)
        #     if 5.6 < gt < 5.7:
        #         print(gt)
        #         torch.save(self.oracle.state_dict(), 'pretrain/oracle_data/oracle{}_lstm.pt.tmp'.format(count))
        #         torch.save(self.oracle.sample(cfg.samples_num, 8 * cfg.batch_size),
        #                    'pretrain/oracle_data/oracle{}_lstm_samples_{}.pt.tmp'.format(count, cfg.samples_num))
        #         torch.save(self.oracle.sample(cfg.samples_num // 2, 8 * cfg.batch_size),
        #                    'pretrain/oracle_data/oracle{}_lstm_samples_{}.pt.tmp'.format(count, cfg.samples_num // 2))
        #         count += 1
        #         if count >= 2:
        #             break

        # gt0 = self.eval_gen(self.oracle_list[0], self.oracle_data_list[0].loader, self.mle_criterion, 0)
        # gt1 = self.eval_gen(self.oracle_list[1], self.oracle_data_list[1].loader, self.mle_criterion, 0)
        # print(gt0, '\n', gt1)

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

    def train_classifier(self, c_step, phase='PRE'):
        """真假样本一起训练，为了让分类器不那么强"""
        self.clas.dis_or_clas = 'clas'  # !!!!!
        total_loss = []
        total_acc = []
        for epoch in range(c_step):
            _, _, clas_inp, clas_target = self.prepare_dis_clas_data('D')

            pred = self.clas(clas_inp)
            c_loss = self.clas_criterion(pred, clas_target)
            c_acc = torch.sum((pred.argmax(dim=-1) == clas_target)).item() / clas_inp.size(0)
            self.optimize(self.clas, c_loss, self.clas)

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
            dis_real_samples, dis_gen_samples, clas_inp, clas_target = self.prepare_dis_clas_data('G')

            # =====Train=====
            # Discriminator loss, input real and fake data
            self.dis.dis_or_clas = 'dis'  # !!!!!
            d_out_real = self.dis(dis_real_samples)
            d_out_fake = self.dis(dis_gen_samples)
            gd_loss = self.dis_criterion(d_out_fake - d_out_real, torch.ones_like(d_out_fake))
            self.dis.dis_or_clas = None

            # Classifier loss, only input fake data
            self.clas.dis_or_clas = 'clas'  # !!!!!
            pred = self.clas(clas_inp)
            gc_loss = self.clas_criterion(pred, clas_target)
            gc_acc = torch.sum((pred.argmax(dim=-1) == clas_target)).item() / clas_inp.size(0)
            self.clas.dis_or_clas = None
            # gc_loss = torch.Tensor([0])
            # gc_acc = torch.Tensor([0])

            # Total loss
            g_loss = gd_loss + gc_loss
            # g_loss = gd_loss

            self.optimize(self.gen_adv_opt, g_loss, self.gen)
            total_g_loss.append(g_loss.item())
            total_gd_loss.append(gd_loss.item())
            total_gc_loss.append(gc_loss.item())
            total_gc_acc.append(gc_acc)

            # self.log.debug('In G: g_loss = %.4f' % g_loss.item())

        if g_step == 0:
            return 0, 0, 0, 0
        return np.mean(total_g_loss), np.mean(total_gd_loss), np.mean(total_gc_loss), np.mean(total_gc_acc)

    def adv_train_descriptor(self, d_step, phase='PRE'):
        total_d_loss = []
        total_dd_loss = []
        total_dc_loss = []
        for step in range(d_step):
            dis_real_samples, dis_gen_samples, clas_inp, clas_target = self.prepare_dis_clas_data('D')

            # Discriminator loss
            if not self.freeze_dis:
                self.dis.dis_or_clas = 'dis'
                d_out_real = self.dis(dis_real_samples)
                d_out_fake = self.dis(dis_gen_samples)
                dd_loss = self.dis_criterion(d_out_real - d_out_fake, torch.ones_like(d_out_real))
                self.dis.dis_or_clas = None
            else:
                dd_loss = torch.Tensor([0.]).cuda()

            # Classifier loss
            if not self.freeze_clas:
                self.clas.dis_or_clas = 'clas'
                pred = self.clas(clas_inp)
                dc_loss = self.clas_criterion(pred, clas_target)
                self.clas.dis_or_clas = None
            else:
                dc_loss = torch.Tensor([0.]).cuda()

            d_loss = dd_loss + dc_loss
            self.optimize(self.desp_opt, d_loss)

            total_d_loss.append(d_loss.item())
            total_dd_loss.append(dd_loss.item())
            total_dc_loss.append(dc_loss.item())

            # if dd_loss < 0.1:
            #     if not self.freeze_dis:
            #         self.log.debug('Freeze dis!')
            #     self.freeze_dis = True
            # else:
            #     if self.freeze_dis:
            #         self.log.debug('Unfreeze dis!')
            #     self.freeze_dis = False
            # if dc_loss < 0.5:
            #     if not self.freeze_clas:
            #         self.log.debug('Freeze clas!')
            #     self.freeze_clas = True
            # else:
            #     if self.freeze_clas:
            #         self.log.debug('Unfreeze clas!')
            #     self.freeze_clas = False

            if phase == 'PRE':
                self.log.debug('[PRE-epoch %d]In G: d_loss = %.4f, dd_loss = %.4f, dc_loss = %.4f', step, d_loss.item(),
                               dd_loss.item(), dc_loss.item())

        if d_step == 0:
            return 0, 0, 0
        return np.mean(total_d_loss), np.mean(total_dd_loss), np.mean(total_dc_loss)

    def adv_train_discriminator(self, d_step, phase='PRE'):
        self.dis.dis_or_clas = 'dis'  # !!!!!
        total_loss = []
        for step in range(d_step):
            dis_real_samples, dis_gen_samples, _, _ = self.prepare_dis_clas_data('D')
            if cfg.CUDA:
                dis_real_samples, dis_gen_samples = dis_real_samples.detach().cuda(), dis_gen_samples.detach().cuda()

            # =====Train=====
            d_out_real = self.dis(dis_real_samples)
            d_out_fake = self.dis(dis_gen_samples)
            d_loss = self.dis_criterion(d_out_real - d_out_fake, torch.ones_like(d_out_real))
            g_loss = self.dis_criterion(d_out_fake - d_out_real, torch.ones_like(d_out_fake))

            self.optimize(self.dis_opt, d_loss, self.dis)
            total_loss.append(d_loss.item())

            if phase == 'PRE':
                self.log.debug('[PRE-epoch %d]In D: d_loss = %.4f, g_loss = %.4f', step, d_loss.item(), g_loss.item())
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

        if cfg.clas_pretrain:
            self.log.info('Load pretrained classifier: {}'.format(cfg.pretrained_clas_path))
            self.clas.load_state_dict(torch.load(cfg.pretrained_clas_path))

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
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
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

    def prepare_dis_clas_data(self, which):
        assert which == 'D' or which == 'G', 'only support for D and G!!'
        real_samples_list = [
            F.one_hot(self.oracle_data_list[i].random_batch()['target'][:cfg.batch_size // cfg.k_label],
                      cfg.vocab_size).float().cuda()
            for i in range(cfg.k_label)]
        if which == 'D':
            with torch.no_grad():
                gen_samples_list = [
                    self.gen.sample(cfg.batch_size // cfg.k_label, cfg.batch_size // cfg.k_label, one_hot=True,
                                    label_i=i)
                    for i in range(cfg.k_label)]
        else:  # 'G'
            gen_samples_list = [
                self.gen.sample(cfg.batch_size // cfg.k_label, cfg.batch_size // cfg.k_label, one_hot=True, label_i=i)
                for i in range(cfg.k_label)]

        # prepare dis data
        dis_real_samples = torch.cat(real_samples_list, dim=0)
        dis_gen_samples = torch.cat(gen_samples_list, dim=0)

        # prepare clas data
        clas_samples_list = [torch.cat((real, fake), dim=0) for (real, fake) in
                             zip(real_samples_list, gen_samples_list)]
        clas_inp, clas_target = CatClasDataIter.prepare(clas_samples_list, detach=True if which == 'D' else False,
                                                        gpu=cfg.CUDA)
        clas_inp = clas_inp[:cfg.batch_size]
        clas_target = clas_target[:cfg.batch_size]

        return dis_real_samples, dis_gen_samples, clas_inp, clas_target
