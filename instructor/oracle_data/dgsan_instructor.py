# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : dgsan_instructor.py
# @Time         : Created at 2020/4/12
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.
import copy
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import config as cfg
from instructor.oracle_data.instructor import BasicInstructor
from models.DGSAN_G import DGSAN_G
from utils.data_loader import GenDataIter
from utils.helpers import create_oracle


class DGSANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(DGSANInstructor, self).__init__(opt)

        # generator
        self.gen = DGSAN_G(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                           cfg.padding_idx, gpu=cfg.CUDA)
        self.old_gen = DGSAN_G(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                               cfg.padding_idx, gpu=cfg.CUDA)
        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)

    def init_model(self):
        if cfg.oracle_pretrain:
            if not os.path.exists(cfg.oracle_state_dict_path):
                create_oracle()
            self.oracle.load_state_dict(
                torch.load(cfg.oracle_state_dict_path, map_location='cuda:{}'.format(cfg.device)))

        if cfg.gen_pretrain:
            self.log.info('Load MLE pretrained generator gen: {}'.format(cfg.pretrained_gen_path))
            self.gen.load_state_dict(torch.load(cfg.pretrained_gen_path, map_location='cuda:{}'.format(cfg.device)))

        if cfg.CUDA:
            self.oracle = self.oracle.cuda()
            self.gen = self.gen.cuda()
            self.old_gen = self.old_gen.cuda()

    def _run(self):
        # ===PRE-TRAINING===
        if not cfg.gen_pretrain:
            self.log.info('Starting Generator MLE Training...')
            self.pretrain_generator(cfg.MLE_train_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
                print('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path))

        # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')
        self.old_gen.load_state_dict(copy.deepcopy(self.gen.state_dict()))

        progress = tqdm(range(cfg.ADV_train_epoch))
        for adv_epoch in progress:
            g_loss = self.adv_train_generator()
            self.old_gen.load_state_dict(copy.deepcopy(self.gen.state_dict()))

            progress.set_description('g_loss: %.4f' % g_loss)

            if adv_epoch % cfg.adv_log_step == 0 or adv_epoch == cfg.ADV_train_epoch - 1:
                self.log.info(
                    '[ADV]: epoch: %d, g_loss = %.4f, %s' % (adv_epoch, g_loss, self.cal_metrics(fmt_str=True)))
                if cfg.if_save and not cfg.if_test:
                    self._save('ADV', adv_epoch)

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

    def adv_train_generator(self):
        g_loss = []
        gen_data = GenDataIter(self.old_gen.sample(cfg.samples_num, cfg.batch_size))
        for (real, fake) in zip(self.oracle_data.loader, gen_data.loader):
            real_inp, real_tar = real['input'], real['target']
            fake_inp, fake_tar = fake['input'], fake['target']
            if cfg.CUDA:
                real_inp, real_tar, fake_inp, fake_tar = real_inp.cuda(), real_tar.cuda(), fake_inp.cuda(), fake_tar.cuda()

            # ===Train===
            real_new_pred = self.cal_pred(self.gen, real_inp, real_tar)
            real_old_pred = self.cal_pred(self.old_gen, real_inp, real_tar)
            fake_new_pred = self.cal_pred(self.gen, fake_inp, fake_tar)
            fake_old_pred = self.cal_pred(self.old_gen, fake_inp, fake_tar)

            eps = 0
            real_loss = -torch.sum(torch.log(1 / (1 + real_old_pred / (real_new_pred + eps) + eps) + eps))
            fake_loss = -torch.sum(torch.log(1 / (1 + fake_new_pred / (fake_old_pred + eps) + eps) + eps))
            adv_loss = real_loss + fake_loss

            self.optimize(self.gen_adv_opt, adv_loss)
            g_loss.append(adv_loss.item())

        return np.mean(g_loss)

    def cal_pred(self, model, input, target):
        pred = torch.exp(model(input, model.init_hidden(cfg.batch_size)))
        target_onehot = F.one_hot(target.view(-1), cfg.vocab_size).float()
        pred = torch.sum(pred * target_onehot, dim=-1)
        return pred.view(cfg.batch_size, -1)
