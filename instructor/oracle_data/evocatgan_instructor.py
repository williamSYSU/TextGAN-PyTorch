# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : evocatgan_instructor.py
# @Time         : Created at 2019-07-18
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.
import random

import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import config as cfg
from instructor.oracle_data.instructor import BasicInstructor
from models.CatGAN_D import CatGAN_D
from models.CatGAN_G import CatGAN_G
from models.EvocatGAN_D import EvoCatGAN_C
from models.EvocatGAN_G import EvoCatGAN_G
from models.Oracle import Oracle
from utils.cat_data_loader import CatGenDataIter, CatClasDataIter
from utils.data_loader import GenDataIter
from utils.data_utils import create_multi_oracle
from utils.gan_loss import GANLoss
from utils.helpers import get_fixed_temperature, get_losses
from utils.text_process import write_tensor


class EvoCatGANInstructor(BasicInstructor):
    """===Version 3==="""

    def __init__(self, opt):
        super(EvoCatGANInstructor, self).__init__(opt)

        # self.log = create_logger(__name__, silent=False, to_disk=True, log_file=cfg.log_filename)

        # generator, discriminator
        self.oracle_list = [Oracle(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                                   cfg.padding_idx, gpu=cfg.CUDA) for _ in range(cfg.k_label)]

        self.gen = CatGAN_G(cfg.k_label, cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim,
                            cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        # self.gen = SlotCatGAN_G(cfg.k_label, cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim,
        #                     cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        self.dis = CatGAN_D(cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.vocab_size,
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
        self.mle_criterion = nn.NLLLoss()
        self.cross_entro_cri = nn.CrossEntropyLoss()
        self.G_critertion = [GANLoss(loss_mode, 'G', cfg.d_type, CUDA=cfg.CUDA) for loss_mode in cfg.mu_type.split()]
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

        if cfg.CUDA:
            for i in range(cfg.k_label):
                self.oracle_list[i] = self.oracle_list[i].cuda()
            self.gen = self.gen.cuda()
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

                self.log.info('[ADV] epoch %d: g_fit: %s, d_loss: %.4f, %s' % (
                    adv_epoch, str(fit_score[best_id]), d_loss, self.comb_metrics(fmt_str=True)))

                if cfg.if_save and not cfg.if_test:
                    for label_i in range(cfg.k_label):
                        self._save('ADV', adv_epoch, label_i)

    def _test(self):
        self.log.debug('>>> Begin test...')

        self._run()
        # self.variation(1, self.G_critertion[0])
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

    def evolve_generator(self, evo_g_step):
        # evaluation real data
        self.prepare_eval_real_data()

        best_score = np.zeros(cfg.n_parent)
        best_fit = []
        best_child = []
        best_child_opt = []
        best_fake_samples_pred = []
        selected_mutation = []
        count = 0

        # TODO: self.d_out_real

        for i, (parent, parent_opt) in enumerate(zip(self.parents, self.parent_adv_opts)):
            for j, criterionG in enumerate(self.G_critertion):
                # ===Variation===
                self.load_gen(parent, parent_opt)  # load state dict to self.gen
                # single loss
                self.variation(evo_g_step, criterionG)

                # ===Evaluation===
                self.prepare_eval_fake_data()  # evaluation fake data
                Fq, Fd, score, eval_fake_samples_pred = self.evaluation(cfg.eval_type)

                # ===Selection===
                if count < cfg.n_parent:
                    best_score[count] = score
                    best_fit.append([Fq, Fd, score])
                    best_child.append(copy.deepcopy(self.gen.state_dict()))
                    best_child_opt.append(copy.deepcopy(self.gen_adv_opt.state_dict()))
                    best_fake_samples_pred.append(eval_fake_samples_pred)
                    selected_mutation.append(criterionG.loss_mode)
                else:  # larger than previous child, replace it
                    fit_com = score - best_score
                    if max(fit_com) > 0:
                        id_replace = np.where(fit_com == max(fit_com))[0][0]
                        best_score[id_replace] = score
                        best_fit[id_replace] = [Fq, Fd, score]
                        best_child[id_replace] = copy.deepcopy(self.gen.state_dict())
                        best_child_opt[id_replace] = copy.deepcopy(self.gen_adv_opt.state_dict())
                        best_fake_samples_pred[id_replace] = eval_fake_samples_pred
                        selected_mutation[id_replace] = criterionG.loss_mode
                count += 1

        self.parents = copy.deepcopy(best_child)
        self.parent_adv_opts = copy.deepcopy(best_child_opt)
        self.best_fake_samples_pred = best_fake_samples_pred
        return best_score, np.array(best_fit), selected_mutation

    # TODO
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

        with torch.no_grad():
            real_samples = F.one_hot(self.oracle_data.random_batch()['target'], cfg.vocab_size).float()
            if cfg.CUDA:
                real_samples = real_samples.cuda()
            self.d_out_real = self.dis(real_samples)

        # evaluate all parents
        for i, (parent, parent_opt) in enumerate(zip(self.parents, self.parent_adv_opts)):
            self.load_gen(parent, parent_opt)
            self.prepare_eval_fake_data()
            Fq, Fd, score, _ = self.evaluation(cfg.eval_type)

            best_score[i] = score
            best_fit.append([Fq, Fd, score])
            best_child.append(copy.deepcopy(self.gen.state_dict()))
            best_child_opt.append(copy.deepcopy(self.gen_adv_opt.state_dict()))
            best_fake_samples.append(self.eval_fake_samples)

        # randomly choose a parent, variation
        target_idx = random.randint(0, len(self.parents) - 1)
        for j, criterionG in enumerate(self.G_critertion):
            self.load_gen(self.parents[target_idx], self.parent_adv_opts[target_idx])  # load generator

            # Variation
            self.variation(evo_g_step, criterionG)

            # Evaluation
            self.prepare_eval_fake_data()  # evaluation fake data
            Fq, Fd, score, _ = self.evaluation(cfg.eval_type)

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
        global dc_loss, dd_loss, d_loss
        total_loss = []

        all_gen_samples_list = list(map(self.merge, *self.best_fake_samples_pred))
        all_gen_samples_list = self.shuffle_eval_samples(all_gen_samples_list)
        for step in range(evo_d_step):
            dis_real_samples, dis_gen_samples = self.prepare_train_data('D')

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

            self.optimize(self.dis_opt, d_loss, self.gen)
            total_loss.append(d_loss.item())

        if evo_d_step == 0:
            return 0
        return np.mean(total_loss)

    # TODO
    def variation(self, g_step, criterionG):
        """Optimize one child (Generator)"""
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
            gd_loss = criterionG(d_out_real, d_out_fake)
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

    # TODO
    def evaluation(self, eval_type):
        """Evaluation all child, update child score. Note that the eval data should be the same"""
        with torch.no_grad():
            eval_fake_samples = []
            eval_fake_samples_pred = []
            for label_i in range(cfg.k_label):
                fake_samples, fake_samples_pred = self.gen.sample(cfg.eval_b_num * cfg.batch_size,
                                                                  cfg.eval_b_num * cfg.batch_size,
                                                                  one_hot=True, label_i=label_i)
                eval_fake_samples.append(fake_samples)
                eval_fake_samples_pred.append(fake_samples_pred)

            if eval_type == 'nll':
                nll_oracle = []
                nll_gen = []
                nll_self = []
                for label_i in range(cfg.k_label):
                    self.gen_data_list[label_i].reset(eval_fake_samples[label_i])

                    if cfg.lambda_fq != 0:
                        nll_oracle.append(-self.eval_gen(self.oracle_list[label_i],
                                                         self.gen_data_list[label_i].loader,
                                                         self.mle_criterion, label_i))  # NLL_Oracle
                    if cfg.lambda_fd != 0:
                        # nll_gen.append(-self.eval_gen(self.gen,
                        #                               self.oracle_data_list[label_i].loader,
                        #                               self.mle_criterion, label_i))  # NLL_gen
                        nll_self.append(self.eval_gen(self.gen,
                                                      self.gen_data_list[label_i].loader,
                                                      self.mle_criterion, label_i))  # NLL_self
                if cfg.k_label == 1:
                    Fq = nll_oracle[0] if len(nll_oracle) > 0 else 0
                    Fd = nll_self[0] if len(nll_self) > 0 else 0
                elif cfg.k_label == 2:
                    Fq = nll_oracle[0] * nll_oracle[1] / (nll_oracle[0] + nll_oracle[1]) if len(nll_oracle) > 0 else 0
                    Fd = nll_self[0] * nll_self[1] / (nll_self[0] + nll_self[1]) if len(nll_self) > 0 else 0
                else:
                    raise NotImplementedError("k_label = %d is not supported" % cfg.k_label)

            else:
                raise NotImplementedError("Evaluation '%s' is not implemented" % eval_type)

            score = cfg.lambda_fq * Fq + cfg.lambda_fd * Fd
            Fq = round(Fq, 3)
            Fd = round(Fd, 3)
            score = round(score, 3)
            return Fq, Fd, score, eval_fake_samples_pred

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

    @staticmethod
    def merge(*args):
        return torch.cat(args, dim=0)

    def prepare_dis_clas_data(self, which, all_gen_samples_list=None, step=None):
        real_samples_list = [
            F.one_hot(self.oracle_data_list[i].random_batch()['target'][:cfg.batch_size // cfg.k_label],
                      cfg.vocab_size).float().cuda()
            for i in range(cfg.k_label)]
        if which == 'D':
            assert all_gen_samples_list is not None and step is not None, 'samples and step have to be given!'
            gen_samples_list = [
                all_gen_samples_list[i][
                step * (cfg.batch_size // cfg.k_label):(step + 1) * (cfg.batch_size // cfg.k_label)]
                for i in range(cfg.k_label)]
        elif which == 'G':
            gen_samples_list = [
                self.gen.sample(cfg.batch_size // cfg.k_label, cfg.batch_size // cfg.k_label, one_hot=True, label_i=i)
                for i in range(cfg.k_label)]
        else:
            raise NotImplementedError('Only support for D and G!')

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

    def shuffle_eval_samples(self, all_eval_samples):
        temp = []
        for i in range(cfg.k_label):
            temp.append(all_eval_samples[i][torch.randperm(all_eval_samples[i].size(0))])
        return temp

    def prepare_train_data(self, which):
        assert which == 'D' or which == 'G', 'only support for D and G!!'
        real_samples_list = [
            F.one_hot(self.oracle_data_list[i].random_batch()['target'][:cfg.batch_size],
                      cfg.vocab_size).float().cuda()
            for i in range(cfg.k_label)]
        if which == 'D':
            with torch.no_grad():
                gen_samples_list = [
                    self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True,
                                    label_i=i)
                    for i in range(cfg.k_label)]
        else:  # 'G'
            gen_samples_list = [
                self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True, label_i=i)
                for i in range(cfg.k_label)]

        return real_samples_list, gen_samples_list

    # TODO
    def prepare_eval_real_data(self):
        with torch.no_grad():
            self.eval_real_samples = torch.cat(
                [F.one_hot(self.oracle_data.random_batch()['target'], cfg.vocab_size).float()
                 for _ in range(cfg.eval_b_num)], dim=0)
            if cfg.CUDA:
                self.eval_real_samples = self.eval_real_samples.cuda()

            if cfg.eval_type == 'rsgan':
                self.eval_d_out_real = self.dis(self.eval_real_samples)

    # TODO
    def prepare_eval_fake_data(self):
        with torch.no_grad():
            self.eval_fake_samples = self.gen.sample(cfg.eval_b_num * cfg.batch_size,
                                                     cfg.max_bn * cfg.batch_size, one_hot=True)
            if cfg.CUDA:
                self.eval_fake_samples = self.eval_fake_samples.cuda()

            if cfg.eval_type == 'rsgan':
                self.eval_d_out_fake = self.dis(self.eval_fake_samples)
