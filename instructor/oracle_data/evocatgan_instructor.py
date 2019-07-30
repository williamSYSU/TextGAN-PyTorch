# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : evocatgan_instructor.py
# @Time         : Created at 2019-07-18
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.
import copy
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import config as cfg
from instructor.oracle_data.instructor import BasicInstructor
from models.EvocatGAN_D import EvoCatGAN_D, EvoCatGAN_C
from models.EvocatGAN_G import EvoCatGAN_G
from models.Oracle import Oracle
from utils.cat_data_loader import CatGenDataIter, CatClasDataIter
from utils.data_loader import GenDataIter
from utils.data_utils import create_multi_oracle
from utils.gan_loss import GANLoss
from utils.helpers import get_fixed_temperature
from utils.text_process import write_tensor


class EvoCatGANInstructor(BasicInstructor):

    def __init__(self, opt):
        super(EvoCatGANInstructor, self).__init__(opt)

        # self.log = create_logger(__name__, silent=False, to_disk=True, log_file=cfg.log_filename)

        # generator, discriminator
        self.oracle_list = [Oracle(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                                   cfg.padding_idx, gpu=cfg.CUDA) for _ in range(cfg.k_label)]

        self.gen = EvoCatGAN_G(cfg.k_label, cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim,
                               cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        self.parents = [EvoCatGAN_G(cfg.k_label, cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim,
                                    cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx,
                                    gpu=cfg.CUDA).state_dict()
                        for _ in range(cfg.n_parent)]  # list of Generator state_dict
        self.dis = EvoCatGAN_D(cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.vocab_size,
                               cfg.padding_idx, gpu=cfg.CUDA)
        self.clas = EvoCatGAN_C(cfg.k_label, cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.vocab_size,
                                cfg.padding_idx, gpu=cfg.CUDA)

        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_adv_lr)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)
        self.clas_opt = optim.Adam(self.clas.parameters(), lr=cfg.clas_lr)
        self.parent_mle_opts = [copy.deepcopy(self.gen_opt.state_dict())
                                for _ in range(cfg.n_parent)]
        self.parent_adv_opts = [copy.deepcopy(self.gen_adv_opt.state_dict())
                                for _ in range(cfg.n_parent)]  # list of optimizer state dict

        # Criterion
        self.mle_criterion = nn.NLLLoss()
        self.clas_criterion = nn.CrossEntropyLoss()
        self.G_criterion = [GANLoss(loss_mode, 'G', cfg.d_type, CUDA=cfg.CUDA) for loss_mode in cfg.mu_type.split()]
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

    def init_model(self):
        if cfg.oracle_pretrain:
            for i in range(cfg.k_label):
                oracle_path = cfg.multi_oracle_state_dict_path.format(i)
                if not os.path.exists(oracle_path):
                    create_multi_oracle(cfg.k_label)
                self.oracle_list[i].load_state_dict(torch.load(oracle_path, map_location='cuda:%d' % cfg.device))

        if cfg.gen_pretrain:
            for i in range(cfg.n_parent):
                if not cfg.use_population:
                    self.log.info('Load MLE pretrained generator gen: {}'.format(cfg.pretrained_gen_path + '%d' % i))
                    self.parents[i] = torch.load(cfg.pretrained_gen_path + '%d' % i, map_location='cpu')
                else:
                    self.log.info('Use population, all parents are pretrained with same weights.')
                    self.log.info('Load MLE pretrained generator gen: {}'.format(cfg.pretrained_gen_path + '%d' % 0))
                    self.parents[i] = torch.load(cfg.pretrained_gen_path + '%d' % 0, map_location='cpu')

        if cfg.clas_pretrain:
            self.log.info('Load  pretrained classifier: {}'.format(cfg.pretrained_clas_path))
            self.clas.load_state_dict(torch.load(cfg.pretrained_clas_path, map_location='cuda:%d' % cfg.device))

        if cfg.CUDA:
            for i in range(cfg.k_label):
                self.oracle_list[i] = self.oracle_list[i].cuda()
            self.gen = self.gen.cuda()
            self.dis = self.dis.cuda()
            self.clas = self.clas.cuda()

    def load_gen(self, parent, parent_opt, mle=False):
        self.gen.load_state_dict(copy.deepcopy(parent))
        if mle:
            self.gen_opt.load_state_dict(copy.deepcopy(parent_opt))
            self.gen_opt.zero_grad()
        else:
            self.gen_adv_opt.load_state_dict(copy.deepcopy(parent_opt))
            self.gen_adv_opt.zero_grad()

    def _run(self):
        # ===Pre-train Classifier with real data===
        self.log.info('Start training Classifier...')
        self.train_classifier(cfg.PRE_clas_epoch)

        # ===Pre-train Generator===
        if not cfg.gen_pretrain:
            for i, (parent, parent_opt) in enumerate(zip(self.parents, self.parent_mle_opts)):
                self.log.info('Starting Generator-{} MLE Training...'.format(i))
                self.load_gen(parent, parent_opt, mle=True)  # load state dict
                self.pretrain_generator(cfg.MLE_train_epoch)
                self.parents[i] = copy.deepcopy(self.gen.state_dict())  # save state dict
                if cfg.if_save and not cfg.if_test:
                    torch.save(self.gen.state_dict(), cfg.pretrained_gen_path + '%d' % i)
                    self.log.info('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path + '%d' % i))

        # ===Adv-train===
        progress = tqdm(range(cfg.ADV_train_epoch))
        for adv_epoch in progress:
            if not cfg.use_population:
                score, fit_score, select_mu = self.evolve_generator(cfg.ADV_g_step)
            else:
                score, fit_score, select_mu = self.evolve_generator_population(cfg.ADV_g_step)
            d_loss = self.evolve_discriminator(cfg.ADV_d_step)

            # self.update_temperature(adv_epoch, cfg.ADV_train_epoch)   # TODO: update parents temperature

            progress.set_description('mu: %s, d_loss = %.4f' % (' '.join(select_mu), d_loss))

            # =====Test=====
            if adv_epoch % cfg.adv_log_step == 0:
                best_id = int(np.argmax(score))
                self.load_gen(self.parents[best_id], self.parent_adv_opts[best_id])

                self.log.info('[ADV] epoch %d: score: %.4f, d_loss: %.4f, %s' % (
                    adv_epoch, score[best_id], d_loss, self.comb_metrics(fmt_str=True)))

                if cfg.if_save and not cfg.if_test:
                    for label_i in range(cfg.k_label):
                        self._save('ADV', adv_epoch, label_i)

    def _test(self):
        self.log.debug('>>> Begin test...')

        self._run()
        # self.variation(1, self.G_criterion[0])
        # self.evolve_generator(1)
        # self.evolve_discriminator(1)

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

        # all child share the same real data output from Discriminator
        with torch.no_grad():
            real_samples = [F.one_hot(self.oracle_data_list[i].random_batch()['target'], cfg.vocab_size).float()
                            for i in range(cfg.k_label)]
            if cfg.CUDA:
                real_samples = [real_samples[i].cuda() for i in range(cfg.k_label)]
            self.d_out_real = [self.dis(real_samples[i]) for i in range(cfg.k_label)]  # d_out_real for each label

        for i, (parent, parent_opt) in enumerate(zip(self.parents, self.parent_adv_opts)):
            for j, criterionG in enumerate(self.G_criterion):
                # ===Variation===
                self.load_gen(parent, parent_opt)  # load state dict to self.gen
                self.variation(evo_g_step, criterionG)

                # ===Evaluation===
                self.prepare_eval_fake_data()  # evaluation fake data
                Fq, Fd, score = self.evaluation(cfg.eval_type)

                # ===Selection===
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
        self.best_fake_samples = best_fake_samples
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

        # all child share the same real data output from Discriminator
        with torch.no_grad():
            real_samples = [F.one_hot(self.oracle_data_list[i].random_batch()['target'], cfg.vocab_size).float()
                            for i in range(cfg.k_label)]
            if cfg.CUDA:
                real_samples = [real_samples[i].cuda() for i in range(cfg.k_label)]
            self.d_out_real = [self.dis(real_samples[i]) for i in range(cfg.k_label)]  # d_out_real for each label

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
        self.best_fake_samples = best_fake_samples
        return best_score, np.array(best_fit), selected_mutation

    def evolve_discriminator(self, evo_d_step):
        global dc_loss, dd_loss, d_loss
        total_loss = []

        all_gen_samples_list = list(map(self.merge, *self.best_fake_samples))  # merge each label of data
        self.all_gen_samples_list = self.shuffle_eval_samples(all_gen_samples_list)  # shuffle data
        for step in range(evo_d_step):
            dis_real_samples, dis_gen_samples = self.prepare_train_data('D', step)

            d_loss = 0
            all_d_out_real = []
            all_d_out_fake = []
            for (real_samples, fake_samples) in zip(dis_real_samples, dis_gen_samples):  # for each label samples
                d_out_real = self.dis(real_samples)
                d_out_fake = self.dis(fake_samples)
                d_loss += self.D_criterion(d_out_real, d_out_fake)
                all_d_out_real.append(d_out_real.view(cfg.batch_size, -1))
                all_d_out_fake.append(d_out_fake.view(cfg.batch_size, -1))

            if cfg.use_all_real_fake:
                all_d_out_real = torch.cat(all_d_out_real, dim=0)
                all_d_out_fake = torch.cat(all_d_out_fake, dim=0)
                all_d_out_real = all_d_out_real[torch.randperm(all_d_out_real.size(0))]
                all_d_out_fake = all_d_out_fake[torch.randperm(all_d_out_fake.size(0))]
                d_loss += self.D_criterion(all_d_out_real, all_d_out_fake)

            self.optimize(self.dis_opt, d_loss, self.gen)
            total_loss.append(d_loss.item())

        if evo_d_step == 0:
            return 0
        return np.mean(total_loss)

    def variation(self, g_step, criterionG):
        """Optimize one child (Generator)"""
        total_loss = []
        for step in range(g_step):
            dis_real_samples, dis_gen_samples = self.prepare_train_data('G')

            # =====Train=====
            g_loss = 0
            all_d_out_real = []
            all_d_out_fake = []
            # for i, (real_samples, fake_samples) in enumerate(zip(dis_real_samples, dis_gen_samples)):
            for i, (d_out_real, fake_samples) in enumerate(zip(self.d_out_real, dis_gen_samples)):  # share real
                # d_out_real = self.dis(real_samples)
                d_out_fake = self.dis(fake_samples)
                g_loss += criterionG(d_out_real, d_out_fake)
                all_d_out_real.append(d_out_real.view(cfg.batch_size, -1))
                all_d_out_fake.append(d_out_fake.view(cfg.batch_size, -1))

            if cfg.use_all_real_fake:
                all_d_out_real = torch.cat(all_d_out_real, dim=0)
                all_d_out_fake = torch.cat(all_d_out_fake, dim=0)
                all_d_out_real = all_d_out_real[torch.randperm(all_d_out_real.size(0))]
                all_d_out_fake = all_d_out_fake[torch.randperm(all_d_out_fake.size(0))]
                g_loss += criterionG(all_d_out_real, all_d_out_fake)

            self.optimize(self.gen_adv_opt, g_loss, self.gen)
            total_loss.append(g_loss.item())

        if g_step == 0:
            return 0
        return np.mean(total_loss)

    def evaluation(self, eval_type):
        """Evaluation all child, update child score. Note that the eval data should be the same"""
        if 'nll' in eval_type:
            nll_oracle = []
            nll_self = []
            for label_i in range(cfg.k_label):
                self.gen_data_list[label_i].reset(
                    self.gen.sample(cfg.eval_b_num * cfg.batch_size, cfg.max_bn * cfg.batch_size, label_i=label_i))

                if cfg.lambda_fq != 0:
                    nll_oracle.append(-self.eval_gen(self.oracle_list[label_i],
                                                     self.gen_data_list[label_i].loader,
                                                     self.mle_criterion, label_i))  # NLL_Oracle
                if cfg.lambda_fd != 0:
                    nll_self.append(self.eval_gen(self.gen,
                                                  self.gen_data_list[label_i].loader,
                                                  self.mle_criterion, label_i))  # NLL_self

            if 'f1' in eval_type:
                if cfg.k_label == 1:
                    Fq = nll_oracle[0] if len(nll_oracle) > 0 else 0
                    Fd = nll_self[0] if len(nll_self) > 0 else 0
                elif cfg.k_label == 2:
                    Fq = nll_oracle[0] * nll_oracle[1] / (nll_oracle[0] + nll_oracle[1]) if len(nll_oracle) > 0 else 0
                    Fd = nll_self[0] * nll_self[1] / (nll_self[0] + nll_self[1]) if len(nll_self) > 0 else 0
                else:
                    raise NotImplementedError("k_label = %d is not supported" % cfg.k_label)
            else:  # sum
                Fq = sum(nll_oracle)
                Fd = sum(nll_self)
        elif eval_type == 'Ra':
            g_loss = 0
            for i in range(cfg.k_label):
                g_loss += torch.sum(self.eval_d_out_fake[i] - torch.mean(self.eval_d_out_real[i])).pow(2)

            Fq = g_loss.item()
            Fd = 0
        else:
            raise NotImplementedError("Evaluation '%s' is not implemented" % eval_type)

        score = cfg.lambda_fq * Fq + cfg.lambda_fd * Fd
        return Fq, Fd, score

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

        eval_samples = self.gen.sample(cfg.samples_num, 4 * cfg.batch_size, label_i=label_i)
        self.gen_data_list[label_i].reset(eval_samples)
        oracle_nll = self.eval_gen(self.oracle_list[label_i],
                                   self.gen_data_list[label_i].loader,
                                   self.mle_criterion, label_i)
        gen_nll = self.eval_gen(self.gen,
                                self.oracle_data_list[label_i].loader,
                                self.mle_criterion, label_i)
        self_nll = self.eval_gen(self.gen,
                                 self.gen_data_list[label_i].loader,
                                 self.mle_criterion, label_i)

        # Evaluation Classifier accuracy
        self.clas_data.reset([eval_samples], label_i)
        _, c_acc = self.eval_dis(self.clas, self.clas_data.loader, self.clas_criterion)

        return oracle_nll, gen_nll, self_nll, c_acc

    def comb_metrics(self, fmt_str=False):
        oracle_nll, gen_nll, self_nll, clas_acc = [], [], [], []
        for label_i in range(cfg.k_label):
            o_nll, g_nll, s_nll, acc = self.cal_metrics(label_i)
            oracle_nll.append(round(o_nll, 4))
            gen_nll.append(round(g_nll, 4))
            self_nll.append(round(s_nll, 4))
            clas_acc.append(round(acc, 4))

        if fmt_str:
            return 'oracle_NLL = %s, gen_NLL = %s, self_NLL = %s, clas_acc = %s' % (
                oracle_nll, gen_nll, self_nll, clas_acc)
        return oracle_nll, gen_nll, self_nll, clas_acc

    def _save(self, phase, epoch, label_i=None):
        assert type(label_i) == int
        torch.save(self.gen.state_dict(), cfg.save_model_root + 'gen_{}_{:05d}.pt'.format(phase, epoch))
        save_sample_path = cfg.save_samples_root + 'samples_c{}_{}_{:05d}.txt'.format(label_i, phase, epoch)
        samples = self.gen.sample(cfg.batch_size, cfg.batch_size, label_i=label_i)
        write_tensor(save_sample_path, samples)

    @staticmethod
    def merge(*args):
        return torch.cat(args, dim=0)

    def shuffle_eval_samples(self, all_eval_samples):
        temp = []
        for i in range(cfg.k_label):
            temp.append(all_eval_samples[i][torch.randperm(all_eval_samples[i].size(0))])
        return temp

    def prepare_train_data(self, which, step=None):
        """Prepare train data for both Generator and Discriminator, each samples_list contains k_label batches of data"""
        assert which == 'D' or which == 'G', 'only support for D and G!!'
        real_samples_list = [
            F.one_hot(self.oracle_data_list[i].random_batch()['target'][:cfg.batch_size],
                      cfg.vocab_size).float().cuda()
            for i in range(cfg.k_label)]
        if which == 'D':
            assert step is not None, 'missing step'
            gen_samples_list = [self.all_gen_samples_list[i][step * cfg.batch_size:(step + 1) * cfg.batch_size]
                                for i in range(cfg.k_label)]  # get a batch from each label
        else:  # 'G'
            gen_samples_list = [
                self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True, label_i=i)
                for i in range(cfg.k_label)]

        return real_samples_list, gen_samples_list

    def prepare_eval_real_data(self):
        """Prepare evaluation real data, contains k_label batches of data"""
        with torch.no_grad():
            self.eval_real_samples = [torch.cat(
                [F.one_hot(self.oracle_data_list[i].random_batch()['target'], cfg.vocab_size).float()
                 for _ in range(cfg.eval_b_num)], dim=0) for i in range(cfg.k_label)]
            if cfg.CUDA:
                self.eval_real_samples = [self.eval_real_samples[i].cuda() for i in range(cfg.k_label)]

            if cfg.eval_type == 'rsgan' or cfg.eval_type == 'Ra':
                self.eval_d_out_real = [self.dis(self.eval_real_samples[i]) for i in range(cfg.k_label)]

    def prepare_eval_fake_data(self):
        """Prepare evaluation fake data, contains k_label batches of data"""
        with torch.no_grad():
            self.eval_fake_samples = [self.gen.sample(cfg.eval_b_num * cfg.batch_size,
                                                      cfg.eval_b_num * cfg.batch_size, one_hot=True, label_i=i)
                                      for i in range(cfg.k_label)]
            if cfg.CUDA:
                self.eval_fake_samples = [self.eval_fake_samples[i].cuda() for i in range(cfg.k_label)]

            if cfg.eval_type == 'rsgan' or cfg.eval_type == 'Ra':
                self.eval_d_out_fake = [self.dis(self.eval_fake_samples[i]) for i in range(cfg.k_label)]
