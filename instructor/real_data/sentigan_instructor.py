# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : sentigan_instructor.py
# @Time         : Created at 2019-07-09
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn
import torch.optim as optim

import config as cfg
from instructor.real_data.instructor import BasicInstructor
from metrics.bleu import BLEU
from models.SentiGAN_D import SentiGAN_D, SentiGAN_C
from models.SentiGAN_G import SentiGAN_G
from utils import rollout
from utils.cat_data_loader import CatClasDataIter
from utils.data_loader import GenDataIter
from utils.text_process import tensor_to_tokens, write_tokens


class SentiGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(SentiGANInstructor, self).__init__(opt)

        # generator, discriminator
        self.gen_list = [SentiGAN_G(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                                    cfg.padding_idx, cfg.temperature, gpu=cfg.CUDA) for _ in range(cfg.k_label)]
        self.dis = SentiGAN_D(cfg.k_label, cfg.dis_embed_dim, cfg.vocab_size, cfg.padding_idx, gpu=cfg.CUDA)
        self.clas = SentiGAN_C(cfg.k_label, cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.extend_vocab_size,
                               cfg.padding_idx, gpu=cfg.CUDA)
        self.init_model()

        # Optimizer
        self.gen_opt_list = [optim.Adam(gen.parameters(), lr=cfg.gen_lr) for gen in self.gen_list]
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)
        self.clas_opt = optim.Adam(self.clas.parameters(), lr=cfg.clas_lr)

        # Criterion
        self.mle_criterion = nn.NLLLoss()
        self.dis_criterion = nn.CrossEntropyLoss()
        self.clas_criterion = nn.CrossEntropyLoss()

        # DataLoader
        self.train_data_list = [GenDataIter(cfg.cat_train_data.format(i)) for i in range(cfg.k_label)]
        self.test_data_list = [GenDataIter(cfg.cat_test_data.format(i), if_test_data=True) for i in range(cfg.k_label)]
        self.clas_data_list = [GenDataIter(cfg.cat_test_data.format(str(i) + '_clas'), if_test_data=True) for i in
                               range(cfg.k_label)]
        self.gen_data_list = [GenDataIter(self.gen_list[i].sample(cfg.batch_size, cfg.batch_size))
                              for i in range(cfg.k_label)]

        self.train_samples_list = [self.train_data_list[i].target for i in range(cfg.k_label)]
        self.clas_samples_list = [self.clas_data_list[i].target for i in range(cfg.k_label)]

        self.dis_data = CatClasDataIter(self.train_samples_list)  # fake init (reset during training)
        self.clas_data = CatClasDataIter(self.clas_samples_list, shuffle=True)  # init classifier train data
        self.eval_clas_data = CatClasDataIter(self.train_samples_list)

        # Others
        self.bleu = [BLEU(test_text=tensor_to_tokens(self.gen_data_list[i].target, self.idx2word_dict),
                          real_text=tensor_to_tokens(self.test_data_list[i].target,
                                                     self.test_data_list[i].idx2word_dict), gram=[2, 3, 4, 5])
                     for i in range(cfg.k_label)]
        self.self_bleu = [BLEU(test_text=tensor_to_tokens(self.gen_data_list[i].target, self.idx2word_dict),
                               real_text=tensor_to_tokens(self.gen_data_list[i].target, self.idx2word_dict),
                               gram=3) for i in range(cfg.k_label)]

    def init_model(self):
        if cfg.dis_pretrain:
            self.log.info(
                'Load pretrained discriminator: {}'.format(cfg.pretrained_dis_path))
            self.dis.load_state_dict(torch.load(cfg.pretrained_dis_path))
        if cfg.gen_pretrain:
            for i in range(cfg.k_label):
                self.log.info('Load MLE pretrained generator gen: {}'.format(cfg.pretrained_gen_path + '%d' % i))
                self.gen_list[i].load_state_dict(torch.load(cfg.pretrained_gen_path + '%d' % i))
        if cfg.clas_pretrain:
            self.log.info('Load  pretrained classifier: {}'.format(cfg.pretrained_clas_path))
            self.clas.load_state_dict(torch.load(cfg.pretrained_clas_path, map_location='cuda:%d' % cfg.device))

        if cfg.CUDA:
            for i in range(cfg.k_label):
                self.gen_list[i] = self.gen_list[i].cuda()
            self.dis = self.dis.cuda()
            self.clas = self.clas.cuda()

    def _run(self):
        # ===Pre-train Classifier with real data===
        self.log.info('Start training Classifier...')
        self.train_classifier(cfg.PRE_clas_epoch)

        # ===PRE-TRAIN GENERATOR===
        if not cfg.gen_pretrain:
            self.log.info('Starting Generator MLE Training...')
            self.pretrain_generator(cfg.MLE_train_epoch)
            if cfg.if_save and not cfg.if_test:
                for i in range(cfg.k_label):
                    torch.save(self.gen_list[i].state_dict(), cfg.pretrained_gen_path + '%d' % i)
                    print('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path + '%d' % i))

        # =====TRAIN DISCRIMINATOR======
        if not cfg.dis_pretrain:
            self.log.info('Starting Discriminator Training...')
            self.train_discriminator(cfg.d_step, cfg.d_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)
                print('Save pre-trained discriminator: {}'.format(cfg.pretrained_dis_path))

        # =====ADVERSARIAL TRAINING=====
        self.log.info('Starting Adversarial Training...')
        self.log.info('Initial generator: %s', self.comb_metrics(fmt_str=True))

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
                for i in range(cfg.k_label):
                    pre_loss = self.train_gen_epoch(self.gen_list[i], self.train_data_list[i].loader,
                                                    self.mle_criterion, self.gen_opt_list[i])

                    # =====Test=====
                    if epoch % cfg.pre_log_step == 0 or epoch == epochs - 1:
                        if i == cfg.k_label - 1:
                            self.log.info('[MLE-GEN] epoch %d : pre_loss = %.4f, %s' % (
                                epoch, pre_loss, self.comb_metrics(fmt_str=True)))
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
        for i in range(cfg.k_label):
            rollout_func = rollout.ROLLOUT(self.gen_list[i], cfg.CUDA)
            total_g_loss = 0
            for step in range(g_step):
                inp, target = self.gen_data_list[i].prepare(self.gen_list[i].sample(cfg.batch_size, cfg.batch_size),
                                                            gpu=cfg.CUDA)

                # =====Train=====
                rewards = rollout_func.get_reward(target, cfg.rollout_num, self.dis)
                adv_loss = self.gen_list[i].batchPGLoss(inp, target, rewards)
                self.optimize(self.gen_opt_list[i], adv_loss)
                total_g_loss += adv_loss.item()

        # =====Test=====
        self.log.info('[ADV-GEN]: %s', self.comb_metrics(fmt_str=True))

    def train_discriminator(self, d_step, d_epoch, phrase='MLE'):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
        Samples are drawn d_step times, and the discriminator is trained for d_epoch d_epoch.
        """
        # prepare loader for validate
        global d_loss, train_acc

        for step in range(d_step):
            # prepare loader for training
            real_samples = []
            fake_samples = []
            for i in range(cfg.k_label):
                real_samples.append(self.train_samples_list[i])
                fake_samples.append(self.gen_list[i].sample(cfg.samples_num // cfg.k_label, 8 * cfg.batch_size))

            dis_samples_list = [torch.cat(fake_samples, dim=0)] + real_samples
            self.dis_data.reset(dis_samples_list)

            for epoch in range(d_epoch):
                # =====Train=====
                d_loss, train_acc = self.train_dis_epoch(self.dis, self.dis_data.loader, self.dis_criterion,
                                                         self.dis_opt)

            # =====Test=====
            self.log.info('[%s-DIS] d_step %d: d_loss = %.4f, train_acc = %.4f' % (
                phrase, step, d_loss, train_acc))

            if cfg.if_save and not cfg.if_test and phrase == 'MLE':
                torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)

    def cal_metrics_with_label(self, label_i=None):
        assert type(label_i) == int, 'missing label'
        eval_samples = self.gen_list[label_i].sample(cfg.samples_num, 8 * cfg.batch_size)
        self.gen_data_list[label_i].reset(eval_samples)
        new_gen_tokens = tensor_to_tokens(eval_samples, self.idx2word_dict)
        self.bleu[label_i].test_text = new_gen_tokens
        self.self_bleu[label_i].real_text = new_gen_tokens
        self.self_bleu[label_i].test_text = tensor_to_tokens(self.gen_list[label_i].sample(200, 200),
                                                             self.idx2word_dict)

        # BLEU-[2,3,4,5]
        bleu_score = self.bleu[label_i].get_score(ignore=False)

        # NLL_gen
        gen_nll = self.eval_gen(self.gen_list[label_i],
                                self.train_data_list[label_i].loader,
                                self.mle_criterion)

        # NLL_div
        div_nll = self.eval_gen(self.gen_list[label_i],
                                self.gen_data_list[label_i].loader,
                                self.mle_criterion)

        # Self-BLEU
        self_bleu_score = self.self_bleu[label_i].get_score(ignore=True)

        # Evaluation Classifier accuracy
        self.clas_data.reset([eval_samples], label_i)
        _, c_acc = self.eval_dis(self.clas, self.clas_data.loader, self.clas_criterion)

        return bleu_score, gen_nll, div_nll, self_bleu_score, c_acc

    def _save(self, phrase, epoch):
        """Save model state dict and generator's samples"""
        for i in range(cfg.k_label):
            if phrase != 'ADV':
                torch.save(self.gen_list[i].state_dict(),
                           cfg.save_model_root + 'gen{}_{}_{:05d}.pt'.format(i, phrase, epoch))
            save_sample_path = cfg.save_samples_root + 'samples_d{}_{}_{:05d}.txt'.format(i, phrase, epoch)
            samples = self.gen_list[i].sample(cfg.batch_size, cfg.batch_size)
            write_tokens(save_sample_path, tensor_to_tokens(samples, self.idx2word_dict))
