# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : JSDGAN_instructor.py
# @Time         : Created at 2019/11/16
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.
import os
import torch
import torch.optim as optim

import config as cfg
from instructor.oracle_data.instructor import BasicInstructor
from models.JSDGAN_G import JSDGAN_G
from utils.helpers import create_oracle


class JSDGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(JSDGANInstructor, self).__init__(opt)

        # generator
        self.gen = JSDGAN_G(cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim, cfg.gen_hidden_dim,
                            cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)

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

    def _run(self):
        # ===PRE-TRAINING===
        self.log.info('Starting Generator MLE Training...')
        self.pretrain_generator(cfg.MLE_train_epoch)

        # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')

        for adv_epoch in range(cfg.ADV_train_epoch):
            g_loss = self.adv_train_generator(cfg.ADV_g_step)  # Generator

            if adv_epoch % cfg.adv_log_step == 0 or adv_epoch == cfg.ADV_train_epoch - 1:
                self.log.info('[ADV] epoch %d: g_loss = %.4f, %s' % (adv_epoch, g_loss, self.cal_metrics(fmt_str=True)))

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

    def adv_train_generator(self, g_step):
        """
        The gen is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """
        global inp, target
        total_loss = 0
        for step in range(g_step):
            for i, data in enumerate(self.oracle_data.loader):
                inp, target = data['input'], data['target']
                if cfg.CUDA:
                    inp, target = inp.cuda(), target.cuda()

                # ===Train===
                adv_loss = self.gen.JSD_loss(inp, target)
                self.optimize(self.gen_opt, adv_loss, self.gen)
                total_loss += adv_loss.item()

        return total_loss
