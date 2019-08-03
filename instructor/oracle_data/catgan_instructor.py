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
from utils.gan_loss import GANLoss
from utils.text_process import write_tensor


class CatGANInstructor(BasicInstructor):

    def __init__(self, opt):
        super(CatGANInstructor, self).__init__(opt)

        # generator, discriminator
        self.oracle_list = [Oracle(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                                   cfg.padding_idx, gpu=cfg.CUDA) for _ in range(cfg.k_label)]

        self.gen = CatGAN_G(cfg.k_label, cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim,
                            cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        # self.gen = SlotCatGAN_G(cfg.k_label, cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim,
        #                     cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        self.dis = CatGAN_D(cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.vocab_size,
                            cfg.padding_idx, gpu=cfg.CUDA)
        self.clas = CatGAN_C(cfg.k_label, cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.vocab_size,
                             cfg.padding_idx, gpu=cfg.CUDA)

        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_adv_lr)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)
        self.clas_opt = optim.Adam(self.clas.parameters(), lr=cfg.clas_lr)

        # Criterion
        self.mle_criterion = nn.NLLLoss()
        self.dis_criterion = nn.BCEWithLogitsLoss()
        self.clas_criterion = nn.CrossEntropyLoss()
        self.G_criterion = GANLoss(cfg.loss_type, 'G', cfg.d_type, CUDA=cfg.CUDA)
        self.D_criterion = GANLoss(cfg.loss_type, 'D', cfg.d_type, CUDA=cfg.CUDA)

        # DataLoader
        if not cfg.oracle_pretrain:
            create_multi_oracle(cfg.k_label)
            for i in range(cfg.k_label):
                oracle_path = cfg.multi_oracle_state_dict_path.format(i)
                self.oracle_list[i].load_state_dict(torch.load(oracle_path, map_location='cuda:%d' % cfg.device))

        self.oracle_samples_list = [torch.load(cfg.multi_oracle_samples_path.format(i, cfg.samples_num))
                                    for i in range(cfg.k_label)]
        self.oracle_data_list = [GenDataIter(self.oracle_samples_list[i]) for i in range(cfg.k_label)]
        self.all_oracle_data = CatGenDataIter(self.oracle_samples_list)  # Shuffled all oracle data
        self.gen_data_list = [GenDataIter(self.gen.sample(cfg.batch_size, cfg.batch_size, label_i=i))
                              for i in range(cfg.k_label)]
        self.clas_data = CatClasDataIter(self.oracle_samples_list)  # init classifier train data
        # self.freeze_id = None

    def init_model(self):
        if cfg.oracle_pretrain:
            for i in range(cfg.k_label):
                oracle_path = cfg.multi_oracle_state_dict_path.format(i)
                if not os.path.exists(oracle_path):
                    create_multi_oracle(cfg.k_label)
                self.oracle_list[i].load_state_dict(torch.load(oracle_path, map_location='cuda:%d' % cfg.device))

        if cfg.gen_pretrain:
            self.log.info('Load MLE pretrained generator gen: {}'.format(cfg.pretrained_gen_path))
            self.gen.load_state_dict(torch.load(cfg.pretrained_gen_path, map_location='cuda:%d' % cfg.device))

        if cfg.CUDA:
            for i in range(cfg.k_label):
                self.oracle_list[i] = self.oracle_list[i].cuda()
            self.gen = self.gen.cuda()
            self.dis = self.dis.cuda()
            self.clas = self.clas.cuda()

    def _run(self):
        # ===Pre-train Classifier with real data===
        self.log.info('Start training Classifier...')
        self.train_classifier(cfg.PRE_clas_epoch)

        # ===Pre-train Generator===
        if not cfg.gen_pretrain:
            self.log.info('Starting Generator MLE Training...')
            self.pretrain_generator(cfg.MLE_train_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
                print('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path))

        # ===Adv-train===
        progress = tqdm(range(cfg.ADV_train_epoch))
        for adv_epoch in progress:
            g_loss = self.adv_train_generator(cfg.ADV_g_step)
            d_loss = self.adv_train_discriminator(cfg.ADV_d_step, 'ADV')  # !!! no adv-train for discriminator

            self.update_temperature(adv_epoch, cfg.ADV_train_epoch)

            # =====Test=====
            progress.set_description(
                'g_loss = %.4f, d_loss = %.4f, temp = %.4f' % (g_loss, d_loss, self.gen.temperature))
            if adv_epoch % cfg.adv_log_step == 0:
                self.log.info(
                    '[ADV] epoch %d : %s' % (adv_epoch, self.comb_metrics(fmt_str=True)))

                if not cfg.if_test and cfg.if_save:
                    for label_i in range(cfg.k_label):
                        self._save('ADV', adv_epoch, label_i)

    def _test(self):
        self.log.debug('>>> Begin test...')

        self._run()
        # self.adv_train_discriminator(1)
        # self.adv_train_generator(1)
        # self.adv_train_descriptor(1)
        # self.update_temperature(1000,2000)
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

    def train_classifier(self, epochs):
        self.clas_data.reset(self.oracle_samples_list)  # TODO: bug: have to reset
        for epoch in range(epochs):
            c_loss, c_acc = self.train_dis_epoch(self.clas, self.clas_data.loader, self.clas_criterion, self.clas_opt)
            self.log.info('[PRE-CLAS] epoch %d: c_loss = %.4f, c_acc = %.4f', epoch, c_loss, c_acc)

        if not cfg.if_test and cfg.if_save:
            torch.save(self.clas.state_dict(), cfg.pretrained_clas_path)

    def adv_train_generator(self, g_step):
        total_loss = []
        for step in range(g_step):
            dis_real_samples, dis_gen_samples = self.prepare_train_data('G')

            # =====Train=====
            g_loss = 0
            all_d_out_real = []
            all_d_out_fake = []
            for i, (real_samples, fake_samples) in enumerate(zip(dis_real_samples, dis_gen_samples)):
                # if self.freeze_id is not None and i == self.freeze_id:
                #     continue
                d_out_real = self.dis(real_samples)
                d_out_fake = self.dis(fake_samples)
                g_loss += self.G_criterion(d_out_real, d_out_fake)
                all_d_out_real.append(d_out_real.view(cfg.batch_size, -1))
                all_d_out_fake.append(d_out_fake.view(cfg.batch_size, -1))

            if cfg.use_all_real_fake:
                all_d_out_real = torch.cat(all_d_out_real, dim=0)
                all_d_out_fake = torch.cat(all_d_out_fake, dim=0)
                all_d_out_real = all_d_out_real[torch.randperm(all_d_out_real.size(0))]
                all_d_out_fake = all_d_out_fake[torch.randperm(all_d_out_fake.size(0))]
                g_loss += self.G_criterion(all_d_out_real, all_d_out_fake)

            self.optimize(self.gen_adv_opt, g_loss, self.gen)
            total_loss.append(g_loss.item())

        if g_step == 0:
            return 0
        return np.mean(total_loss)

    def adv_train_discriminator(self, d_step, phase='PRE'):
        total_loss = []
        for step in range(d_step):
            dis_real_samples, dis_gen_samples = self.prepare_train_data('D')

            # =====Train=====
            d_loss = 0
            all_d_out_real = []
            all_d_out_fake = []
            for (real_samples, fake_samples) in zip(dis_real_samples, dis_gen_samples):  # for each label samples
                d_out_real = self.dis(real_samples)
                d_out_fake = self.dis(fake_samples)

                # vanilla
                d_loss += self.D_criterion(d_out_real, d_out_fake)

                # real --> real
                # d_out_real_reshape = d_out_real.view(cfg.batch_size, -1)
                # d_out_fake_reshape = d_out_fake.view(cfg.batch_size, -1)
                # cut_size = cfg.batch_size // 2
                # target_size = cfg.batch_size * cfg.num_rep // 2
                # d_loss += self.dis_criterion(
                #     d_out_real_reshape[:cut_size].view(-1) - d_out_real_reshape[cut_size:].view(-1),
                #     torch.zeros(target_size).cuda())
                # d_loss += self.dis_criterion(
                #     d_out_fake_reshape[:cut_size].view(-1) - d_out_fake_reshape[cut_size:].view(-1),
                #     torch.zeros(target_size).cuda())

                all_d_out_real.append(d_out_real.view(cfg.batch_size, -1))
                all_d_out_fake.append(d_out_fake.view(cfg.batch_size, -1))

            if cfg.use_all_real_fake:
                all_d_out_real = torch.cat(all_d_out_real, dim=0)
                all_d_out_fake = torch.cat(all_d_out_fake, dim=0)
                all_d_out_real = all_d_out_real[torch.randperm(all_d_out_real.size(0))]
                all_d_out_fake = all_d_out_fake[torch.randperm(all_d_out_fake.size(0))]
                d_loss += self.D_criterion(all_d_out_real, all_d_out_fake)

            self.optimize(self.dis_opt, d_loss, self.dis)
            total_loss.append(d_loss.item())

            if phase == 'PRE':
                self.log.debug('[PRE-epoch %d]In D: d_loss = %.4f', step, d_loss.item())
        if d_step == 0:
            return 0
        return np.mean(total_loss)

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

    def _save(self, phase, epoch, label_i=None):
        assert type(label_i) == int
        torch.save(self.gen.state_dict(), cfg.save_model_root + 'gen_{}_{:05d}.pt'.format(phase, epoch))
        save_sample_path = cfg.save_samples_root + 'samples_c{}_{}_{:05d}.txt'.format(label_i, phase, epoch)
        samples = self.gen.sample(cfg.batch_size, cfg.batch_size, label_i=label_i)
        write_tensor(save_sample_path, samples)

    def prepare_train_data(self, which):
        assert which == 'D' or which == 'G', 'only support for D and G!!'
        real_samples_list = [
            F.one_hot(self.oracle_data_list[i].random_batch()['target'][:cfg.batch_size], cfg.vocab_size).float().cuda()
            for i in range(cfg.k_label)]
        if which == 'D':
            with torch.no_grad():
                gen_samples_list = [self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True, label_i=i)
                                    for i in range(cfg.k_label)]
        else:  # 'G'
            gen_samples_list = [
                self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True, label_i=i)
                for i in range(cfg.k_label)]

        return real_samples_list, gen_samples_list
