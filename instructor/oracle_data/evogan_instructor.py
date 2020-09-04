# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : evogan_instructor.py
# @Time         : Created at 2019-07-09
# @Blog         : http://zhiweil.ml/
# @Description  : CatGAN for general text generation
# Copyrights (C) 2018. All Rights Reserved.
import copy
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import config as cfg
from instructor.oracle_data.instructor import BasicInstructor
from metrics.nll import NLL
from models.EvoGAN_D import EvoGAN_D
from models.EvoGAN_G import EvoGAN_G
from utils.data_loader import GenDataIter
from utils.gan_loss import GANLoss
from utils.helpers import get_fixed_temperature, get_losses, create_oracle


class EvoGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(EvoGANInstructor, self).__init__(opt)

        # generator, discriminator
        self.gen = EvoGAN_G(cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim, cfg.gen_hidden_dim,
                            cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        self.parents = [EvoGAN_G(cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim, cfg.gen_hidden_dim,
                                 cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA).state_dict()
                        for _ in range(cfg.n_parent)]  # list of Generator state_dict
        self.dis = EvoGAN_D(cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.vocab_size,
                            cfg.padding_idx, gpu=cfg.CUDA)

        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_adv_lr)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)
        self.parent_mle_opts = [copy.deepcopy(self.gen_opt.state_dict())
                                for _ in range(cfg.n_parent)]
        self.parent_adv_opts = [copy.deepcopy(self.gen_adv_opt.state_dict())
                                for _ in range(cfg.n_parent)]  # list of optimizer state dict

        # Criterion
        self.G_criterion = [GANLoss(loss_mode, 'G', cfg.d_type, CUDA=cfg.CUDA) for loss_mode in cfg.mu_type.split()]
        self.D_criterion = GANLoss(cfg.loss_type, 'D', cfg.d_type, CUDA=cfg.CUDA)

    def init_model(self):
        if cfg.oracle_pretrain:
            if not os.path.exists(cfg.oracle_state_dict_path):
                create_oracle()
            self.oracle.load_state_dict(torch.load(cfg.oracle_state_dict_path, map_location='cuda:%d' % cfg.device))

        if cfg.dis_pretrain:
            self.log.info(
                'Load pretrained discriminator: {}'.format(cfg.pretrained_dis_path))
            self.dis.load_state_dict(torch.load(cfg.pretrained_dis_path, map_location='cuda:{}'.format(cfg.device)))

        if cfg.gen_pretrain:
            for i in range(cfg.n_parent):
                self.log.info('Load MLE pretrained generator gen: {}'.format(cfg.pretrained_gen_path + '%d' % i))
                self.parents[i] = torch.load(cfg.pretrained_gen_path + '%d' % 0, map_location='cpu')

        if cfg.CUDA:
            self.oracle = self.oracle.cuda()
            self.gen = self.gen.cuda()

            if cfg.multi_gpu:
                self.dis = torch.nn.parallel.DataParallel(self.dis, device_ids=cfg.devices)
            self.dis = self.dis.cuda()

    def load_gen(self, parent, parent_opt, mle=False):
        self.gen.load_state_dict(copy.deepcopy(parent))
        if mle:
            self.gen_opt.load_state_dict(copy.deepcopy(parent_opt))
            self.gen_opt.zero_grad()
        else:
            self.gen_adv_opt.load_state_dict(copy.deepcopy(parent_opt))
            self.gen_adv_opt.zero_grad()

    def _run(self):
        # ===PRE-TRAINING (GENERATOR)===
        if not cfg.gen_pretrain:
            for i, (parent, parent_opt) in enumerate(zip(self.parents, self.parent_mle_opts)):
                self.log.info('Starting Generator-{} MLE Training...'.format(i))
                self.load_gen(parent, parent_opt, mle=True)  # load state dict
                self.pretrain_generator(cfg.MLE_train_epoch)
                self.parents[i] = copy.deepcopy(self.gen.state_dict())  # save state dict
                if cfg.if_save and not cfg.if_test:
                    torch.save(self.gen.state_dict(), cfg.pretrained_gen_path + '%d' % i)
                    self.log.info('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path + '%d' % i))

        # # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')
        progress = tqdm(range(cfg.ADV_train_epoch))
        for adv_epoch in progress:
            if cfg.temperature == 1:
                score, fit_score, select_mu = self.evolve_generator(cfg.ADV_g_step)
            else:  # evolve with temperature
                score, fit_score, select_mu = self.evolve_generator_with_temp(adv_epoch, cfg.ADV_g_step)
            d_loss = self.evolve_discriminator(cfg.ADV_d_step)

            best_id = int(np.argmax(score))
            progress.set_description('mu: %s, d_loss = %.4f, temp = %.4f' % (
                ' '.join(select_mu), d_loss, self.parents[best_id]['temperature'].item()))

            # TEST
            if adv_epoch % cfg.adv_log_step == 0 or adv_epoch == cfg.ADV_train_epoch - 1:
                best_id = int(np.argmax(score))
                self.load_gen(self.parents[best_id], self.parent_adv_opts[best_id])

                # self.log.info('[ADV] epoch %d: temp = %.4f' % (adv_epoch, self.gen.temperature.item()))
                # self.log.info(fit_score[best_id])
                self.log.info('[ADV] epoch %d: temp = %.4f, d_loss = %.4f, %s' % (
                    adv_epoch, self.gen.temperature.item(), d_loss, self.cal_metrics(fmt_str=True)))

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
                # ===Train===
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

    def evolve_generator(self, evo_g_step):
        # evaluation real data
        self.prepare_eval_real_data()

        best_score = np.zeros(cfg.n_parent)
        best_fit = []
        best_child = []
        best_child_opt = []
        best_fake_samples = []
        selected_mutation = []
        count = 0

        # all children share the same real data output from Discriminator
        with torch.no_grad():
            real_samples = F.one_hot(self.oracle_data.random_batch()['target'], cfg.vocab_size).float()
            if cfg.CUDA:
                real_samples = real_samples.cuda()
            self.d_out_real = self.dis(real_samples)

        for i, (parent, parent_opt) in enumerate(zip(self.parents, self.parent_adv_opts)):
            for j, criterionG in enumerate(self.G_criterion):
                # Variation
                self.load_gen(parent, parent_opt)  # load state dict to self.gen
                # single loss
                self.variation(evo_g_step, criterionG)

                # mixture variation: double loss with random weight
                # choice = random.sample(range(0, len(self.G_criterion)), 2)
                # cri_list = [self.G_criterion[choice[0]], self.G_criterion[choice[1]]]
                # self.variation(evo_g_step, cri_list)

                # all loss with random weight
                # self.variation(evo_g_step, self.G_criterion)

                # Evaluation
                self.prepare_eval_fake_data()  # evaluation fake data
                Fq, Fd, score = self.evaluation(cfg.eval_type)

                # Selection
                if count < cfg.n_parent:
                    best_score[count] = score
                    best_fit.append([Fq, Fd, score])
                    best_child.append(copy.deepcopy(self.gen.state_dict()))
                    best_child_opt.append(copy.deepcopy(self.gen_adv_opt.state_dict()))
                    best_fake_samples.append(self.eval_fake_samples)
                    selected_mutation.append(criterionG.loss_mode)
                else:  # larger than previous child, replace it
                    fit_com = score - best_score
                    if max(fit_com) > 0:
                        id_replace = np.where(fit_com == max(fit_com))[0][0]
                        best_score[id_replace] = score
                        best_fit[id_replace] = [Fq, Fd, score]
                        best_child[id_replace] = copy.deepcopy(self.gen.state_dict())
                        best_child_opt[id_replace] = copy.deepcopy(self.gen_adv_opt.state_dict())
                        best_fake_samples[id_replace] = self.eval_fake_samples
                        selected_mutation[id_replace] = criterionG.loss_mode
                count += 1

        self.parents = copy.deepcopy(best_child)
        self.parent_adv_opts = copy.deepcopy(best_child_opt)
        self.best_fake_samples = torch.cat(best_fake_samples, dim=0)
        return best_score, np.array(best_fit), selected_mutation

    def evolve_generator_with_temp(self, cur_adv_step, evo_g_step):
        # evaluation real data
        self.prepare_eval_real_data()

        best_score = np.zeros(cfg.n_parent)
        best_fit = []
        best_child = []
        best_child_opt = []
        best_fake_samples = []
        selected_mutation = []
        count = 0

        # all children share the same real data output from Discriminator
        with torch.no_grad():
            real_samples = F.one_hot(self.oracle_data.random_batch()['target'], cfg.vocab_size).float()
            if cfg.CUDA:
                real_samples = real_samples.cuda()
            self.d_out_real = self.dis(real_samples)

        for i, (parent, parent_opt) in enumerate(zip(self.parents, self.parent_adv_opts)):
            for j, criterionG in enumerate(self.G_criterion):
                all_temp = self.get_evo_temp(cur_adv_step)  # get evo temp

                temp_score = float('-inf')
                temp_fit = None
                temp_child = None
                temp_child_opt = None
                temp_fake_samples = None

                # Selection based on temperature, use eval_type=nll
                for temp in all_temp:
                    # Variation
                    self.load_gen(parent, parent_opt)  # load state dict to self.gen
                    self.gen.temperature.data = temp  # update Generator temperature

                    self.variation(evo_g_step, criterionG)

                    # Evaluation
                    self.prepare_eval_fake_data()  # evaluation fake data
                    _, _, t_score = self.evaluation('Ra')  # for temp evolutionary
                    loss_Fq, loss_Fd, loss_score = self.evaluation(cfg.eval_type)  # for loss evolutionary

                    if t_score > temp_score:
                        temp_score = loss_score
                        temp_fit = [loss_Fq, loss_Fd, loss_score]
                        temp_child = copy.deepcopy(self.gen.state_dict())
                        temp_child_opt = copy.deepcopy(self.gen_adv_opt.state_dict())
                        temp_fake_samples = copy.deepcopy(self.eval_fake_samples)

                # Selection based on mu_type, use eval_type=cfg.eval_type
                if count < cfg.n_parent:
                    best_score[count] = temp_score
                    best_fit.append(temp_fit)
                    best_child.append(temp_child)
                    best_child_opt.append(temp_child_opt)
                    best_fake_samples.append(temp_fake_samples)
                    selected_mutation.append(criterionG.loss_mode)
                else:  # larger than previous child, replace it
                    fit_com = temp_score - best_score
                    if max(fit_com) > 0:
                        id_replace = np.where(fit_com == max(fit_com))[0][0]
                        best_score[id_replace] = temp_score
                        best_fit[id_replace] = temp_fit
                        best_child[id_replace] = temp_child
                        best_child_opt[id_replace] = temp_child_opt
                        best_fake_samples[id_replace] = temp_fake_samples
                        selected_mutation[id_replace] = criterionG.loss_mode
                count += 1

        self.parents = copy.deepcopy(best_child)
        self.parent_adv_opts = copy.deepcopy(best_child_opt)
        self.best_fake_samples = torch.cat(best_fake_samples, dim=0)
        return best_score, np.array(best_fit), selected_mutation

    def evolve_generator_population(self, evo_g_step):
        """
        1. randomly choose a parent from population;
        2. variation;
        3. evaluate all parents and child, choose the best
        """
        # evaluation real data
        self.prepare_eval_real_data()

        best_score = np.zeros(cfg.n_parent)
        best_fit = []
        best_child = []
        best_child_opt = []
        best_fake_samples = []
        selected_mutation = []

        # all children share the same real data output from Discriminator
        with torch.no_grad():
            real_samples = F.one_hot(self.oracle_data.random_batch()['target'], cfg.vocab_size).float()
            if cfg.CUDA:
                real_samples = real_samples.cuda()
            self.d_out_real = self.dis(real_samples)

        # evaluate all parents
        for i, (parent, parent_opt) in enumerate(zip(self.parents, self.parent_adv_opts)):
            self.load_gen(parent, parent_opt)
            self.prepare_eval_fake_data()
            Fq, Fd, score = self.evaluation(cfg.eval_type)

            best_score[i] = score
            best_fit.append([Fq, Fd, score])
            best_child.append(copy.deepcopy(self.gen.state_dict()))
            best_child_opt.append(copy.deepcopy(self.gen_adv_opt.state_dict()))
            best_fake_samples.append(self.eval_fake_samples)

        # randomly choose a parent, variation
        target_idx = random.randint(0, len(self.parents) - 1)
        for j, criterionG in enumerate(self.G_criterion):
            self.load_gen(self.parents[target_idx], self.parent_adv_opts[target_idx])  # load generator

            # Variation
            self.variation(evo_g_step, criterionG)

            # Evaluation
            self.prepare_eval_fake_data()  # evaluation fake data
            Fq, Fd, score = self.evaluation(cfg.eval_type)

            # Selection
            fit_com = score - best_score
            if max(fit_com) > 0:
                id_replace = np.where(fit_com == max(fit_com))[0][0]
                best_score[id_replace] = score
                best_fit[id_replace] = [Fq, Fd, score]
                best_child[id_replace] = copy.deepcopy(self.gen.state_dict())
                best_child_opt[id_replace] = copy.deepcopy(self.gen_adv_opt.state_dict())
                best_fake_samples[id_replace] = self.eval_fake_samples
                selected_mutation.append(criterionG.loss_mode)

        self.parents = copy.deepcopy(best_child)
        self.parent_adv_opts = copy.deepcopy(best_child_opt)
        self.best_fake_samples = torch.cat(best_fake_samples, dim=0)
        return best_score, np.array(best_fit), selected_mutation

    def evolve_discriminator(self, evo_d_step):
        total_loss = 0
        for step in range(evo_d_step):
            real_samples = F.one_hot(self.oracle_data.random_batch()['target'], cfg.vocab_size).float()
            gen_samples = self.best_fake_samples[step * cfg.batch_size:(step + 1) * cfg.batch_size]
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()

            # ===Train===
            d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
            d_loss = self.D_criterion(d_out_real, d_out_fake)

            self.optimize(self.dis_opt, d_loss, self.dis)
            total_loss += d_loss.item()

        return total_loss / evo_d_step if evo_d_step != 0 else 0

    def variation(self, g_step, criterionG):
        """Must call self.load_gen() before variation"""
        total_loss = 0
        for step in range(g_step):
            # real_samples = F.one_hot(self.oracle_data.random_batch()['target'], cfg.vocab_size).float()
            gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True)
            if cfg.CUDA:
                # real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
                gen_samples = gen_samples.cuda()

            # ===Train===
            # d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
            if type(criterionG) == list:  # multiple loss
                # mixture variation: double loss
                rand_w = torch.rand(1).cuda()
                cri_1, cri_2 = criterionG
                g_loss = rand_w * cri_1(self.d_out_real, d_out_fake) + (1 - rand_w) * cri_2(self.d_out_real, d_out_fake)

                # all loss
                # rand_w = F.softmax(torch.rand(len(criterionG)).cuda(), dim=0)
                # all_loss = []
                # for crit in criterionG:
                #     all_loss.append(crit(d_out_real, d_out_fake))
                # g_loss = torch.dot(rand_w, torch.stack(all_loss, dim=0))
            else:
                # g_loss = criterionG(d_out_real, d_out_fake)
                g_loss = criterionG(self.d_out_real, d_out_fake)
            self.optimize(self.gen_adv_opt, g_loss, self.gen)
            total_loss += g_loss.item()

        return total_loss / g_step if g_step != 0 else 0

    def evaluation(self, eval_type):
        """Evaluation all children, update child score. Note that the eval data should be the same"""

        eval_samples = self.gen.sample(cfg.eval_b_num * cfg.batch_size, cfg.max_bn * cfg.batch_size)
        gen_data = GenDataIter(eval_samples)

        # Fd
        if cfg.lambda_fd != 0:
            Fd = NLL.cal_nll(self.gen, gen_data.loader, self.mle_criterion)  # NLL_div
        else:
            Fd = 0

        if eval_type == 'standard':
            Fq = self.eval_d_out_fake.mean().cpu().item()
        elif eval_type == 'rsgan':
            g_loss, d_loss = get_losses(self.eval_d_out_real, self.eval_d_out_fake, 'rsgan')
            Fq = d_loss.item()
        elif eval_type == 'nll':
            if cfg.lambda_fq != 0:
                Fq = -NLL.cal_nll(self.oracle, gen_data.loader, self.mle_criterion)  # NLL_Oracle
            else:
                Fq = 0
        elif eval_type == 'Ra':
            g_loss = torch.sigmoid(self.eval_d_out_fake - torch.mean(self.eval_d_out_real)).sum()
            Fq = g_loss.item()
        else:
            raise NotImplementedError("Evaluation '%s' is not implemented" % eval_type)

        score = cfg.lambda_fq * Fq + cfg.lambda_fd * Fd
        return Fq, Fd, score

    def prepare_eval_real_data(self):
        with torch.no_grad():
            self.eval_real_samples = torch.cat(
                [F.one_hot(self.oracle_data.random_batch()['target'], cfg.vocab_size).float()
                 for _ in range(cfg.eval_b_num)], dim=0)
            if cfg.CUDA:
                self.eval_real_samples = self.eval_real_samples.cuda()

            if cfg.eval_type == 'rsgan' or cfg.eval_type == 'Ra':
                self.eval_d_out_real = self.dis(self.eval_real_samples)

    def prepare_eval_fake_data(self):
        with torch.no_grad():
            self.eval_fake_samples = self.gen.sample(cfg.eval_b_num * cfg.batch_size,
                                                     cfg.eval_b_num * cfg.batch_size, one_hot=True)
            if cfg.CUDA:
                self.eval_fake_samples = self.eval_fake_samples.cuda()

            if cfg.eval_type == 'rsgan' or cfg.eval_type == 'Ra':
                self.eval_d_out_fake = self.dis(self.eval_fake_samples)

    @staticmethod
    def get_evo_temp(cur_step):
        """randomly get different temperature according to current adversarial step"""
        mu_temp_type = cfg.mu_temp.split()
        all_temp = list()

        # all_temp.append(get_fixed_temperature(1.0, 0, 0, 'no'))  # temp=1.0
        all_temp.append(get_fixed_temperature(cfg.temperature, cur_step, cfg.ADV_train_epoch,
                                              random.choice(mu_temp_type)))  # current step
        all_temp.append(
            get_fixed_temperature(cfg.temperature, cur_step + cfg.evo_temp_step, cfg.ADV_train_epoch,
                                  random.choice(mu_temp_type)))
        if cur_step > cfg.evo_temp_step:
            all_temp.append(
                get_fixed_temperature(cfg.temperature, cur_step - cfg.evo_temp_step, cfg.ADV_train_epoch,
                                      random.choice(mu_temp_type)))

        return torch.Tensor(all_temp)  # three temp
