# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : leakgan_instructor.py
# @Time         : Created at 2019-06-05
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.optim as optim

import config as cfg
from instructor.real_data.instructor import BasicInstructor
from models.LeakGAN_D import LeakGAN_D
from models.LeakGAN_G import LeakGAN_G
from utils import rollout
from utils.data_loader import GenDataIter, DisDataIter
from utils.text_process import tensor_to_tokens, write_tokens


class LeakGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(LeakGANInstructor, self).__init__(opt)

        # generator, discriminator
        self.gen = LeakGAN_G(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                             cfg.padding_idx, cfg.goal_size, cfg.step_size, cfg.CUDA)
        self.dis = LeakGAN_D(cfg.dis_embed_dim, cfg.vocab_size, cfg.padding_idx, gpu=cfg.CUDA)
        self.init_model()

        # optimizer
        mana_params, work_params = self.gen.split_params()
        mana_opt = optim.Adam(mana_params, lr=cfg.gen_lr)
        work_opt = optim.Adam(work_params, lr=cfg.gen_lr)

        self.gen_opt = [mana_opt, work_opt]
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)

    def _run(self):
        for inter_num in range(cfg.inter_epoch):
            self.log.info('>>> Interleaved Round %d...' % inter_num)
            self.sig.update()  # update signal
            if self.sig.pre_sig:
                # ===DISCRIMINATOR PRE-TRAINING===
                if not cfg.dis_pretrain:
                    self.log.info('Starting Discriminator Training...')
                    self.train_discriminator(cfg.d_step, cfg.d_epoch)
                    if cfg.if_save and not cfg.if_test:
                        torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)
                        print('Save pre-trained discriminator: {}'.format(cfg.pretrained_dis_path))

                # ===GENERATOR MLE TRAINING===
                if not cfg.gen_pretrain:
                    self.log.info('Starting Generator MLE Training...')
                    self.pretrain_generator(cfg.MLE_train_epoch)
                    if cfg.if_save and not cfg.if_test:
                        torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
                        print('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path))
            else:
                self.log.info('>>> Stop by pre_signal! Skip to adversarial training...')
                break

        # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')
        self.log.info('Initial generator: %s' % (str(self.cal_metrics(fmt_str=True))))

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
        Max Likelihood Pretraining for the gen

        - gen_opt: [mana_opt, work_opt]
        """
        for epoch in range(epochs):
            self.sig.update()
            if self.sig.pre_sig:
                pre_mana_loss = 0
                pre_work_loss = 0

                # ===Train===
                for i, data in enumerate(self.train_data.loader):
                    inp, target = data['input'], data['target']
                    if cfg.CUDA:
                        inp, target = inp.cuda(), target.cuda()

                    mana_loss, work_loss = self.gen.pretrain_loss(target, self.dis)
                    self.optimize_multi(self.gen_opt, [mana_loss, work_loss])
                    pre_mana_loss += mana_loss.data.item()
                    pre_work_loss += work_loss.data.item()
                pre_mana_loss = pre_mana_loss / len(self.train_data.loader)
                pre_work_loss = pre_work_loss / len(self.train_data.loader)

                # ===Test===
                if epoch % cfg.pre_log_step == 0 or epoch == epochs - 1:
                    self.log.info('[MLE-GEN] epoch %d : pre_mana_loss = %.4f, pre_work_loss = %.4f, %s' % (
                        epoch, pre_mana_loss, pre_work_loss, self.cal_metrics(fmt_str=True)))

                    if cfg.if_save and not cfg.if_test:
                        self._save('MLE', epoch)
            else:
                self.log.info('>>> Stop by pre signal, skip to adversarial training...')
                break

    def adv_train_generator(self, g_step, current_k=0):
        """
        The gen is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """

        rollout_func = rollout.ROLLOUT(self.gen, cfg.CUDA)
        adv_mana_loss = 0
        adv_work_loss = 0
        for step in range(g_step):
            with torch.no_grad():
                gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, self.dis,
                                              train=True)  # !!! train=True, the only place
                inp, target = GenDataIter.prepare(gen_samples, gpu=cfg.CUDA)

            # ===Train===
            rewards = rollout_func.get_reward_leakgan(target, cfg.rollout_num, self.dis,
                                                      current_k).cpu()  # reward with MC search
            mana_loss, work_loss = self.gen.adversarial_loss(target, rewards, self.dis)

            # update parameters
            self.optimize_multi(self.gen_opt, [mana_loss, work_loss])
            adv_mana_loss += mana_loss.data.item()
            adv_work_loss += work_loss.data.item()
        # ===Test===
        self.log.info('[ADV-GEN] adv_mana_loss = %.4f, adv_work_loss = %.4f, %s' % (
            adv_mana_loss / g_step, adv_work_loss / g_step, self.cal_metrics(fmt_str=True)))

    def train_discriminator(self, d_step, d_epoch, phase='MLE'):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
        Samples are drawn d_step times, and the discriminator is trained for d_epoch d_epoch.
        """
        d_loss, train_acc = 0, 0
        for step in range(d_step):
            # prepare loader for training
            pos_samples = self.train_data.target
            neg_samples = self.gen.sample(cfg.samples_num, cfg.batch_size, self.dis)
            dis_data = DisDataIter(pos_samples, neg_samples)

            for epoch in range(d_epoch):
                # ===Train===
                d_loss, train_acc = self.train_dis_epoch(self.dis, dis_data.loader, self.dis_criterion,
                                                         self.dis_opt)

            # ===Test===
            self.log.info('[%s-DIS] d_step %d: d_loss = %.4f, train_acc = %.4f,' % (
                phase, step, d_loss, train_acc))

    def cal_metrics(self, fmt_str=False):
        with torch.no_grad():
            # Prepare data for evaluation
            eval_samples = self.gen.sample(cfg.samples_num, cfg.batch_size, self.dis)
            gen_data = GenDataIter(eval_samples)
            gen_tokens = tensor_to_tokens(eval_samples, self.idx2word_dict)
            gen_tokens_s = tensor_to_tokens(self.gen.sample(200, cfg.batch_size, self.dis), self.idx2word_dict)

            # Reset metrics
            self.bleu.reset(test_text=gen_tokens, real_text=self.test_data.tokens)
            self.nll_gen.reset(self.gen, self.train_data.loader, leak_dis=self.dis)
            self.nll_div.reset(self.gen, gen_data.loader, leak_dis=self.dis)
            self.self_bleu.reset(test_text=gen_tokens_s, real_text=gen_tokens)
            self.ppl.reset(gen_tokens)

        if fmt_str:
            return ', '.join(['%s = %s' % (metric.get_name(), metric.get_score()) for metric in self.all_metrics])
        else:
            return [metric.get_score() for metric in self.all_metrics]

    def _save(self, phase, epoch):
        torch.save(self.gen.state_dict(), cfg.save_model_root + 'gen_{}_{:05d}.pt'.format(phase, epoch))
        save_sample_path = cfg.save_samples_root + 'samples_{}_{:05d}.txt'.format(phase, epoch)
        samples = self.gen.sample(cfg.batch_size, cfg.batch_size, self.dis)
        write_tokens(save_sample_path, tensor_to_tokens(samples, self.idx2word_dict))
