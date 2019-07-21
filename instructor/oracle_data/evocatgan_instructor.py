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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import config as cfg
from instructor.oracle_data.instructor import BasicInstructor
from models.CatGAN_D import CatGAN_C
from models.CatGAN_G import CatGAN_G
from models.EvocatGAN_D import EvoCatGAN_C, EvoCatGAN_D
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
        self.G_critertion = [GANLoss(loss_mode, 'G', cfg.d_type, CUDA=cfg.CUDA) for loss_mode in cfg.mu_type.split()]
        self.D_critertion = GANLoss(cfg.loss_type, 'D', cfg.d_type, CUDA=cfg.CUDA)

        # DataLoader
        self.oracle_samples_list = [torch.load(cfg.multi_oracle_samples_path.format(i, cfg.samples_num))
                                    for i in range(cfg.k_label)]
        self.oracle_data_list = [GenDataIter(self.oracle_samples_list[i]) for i in range(cfg.k_label)]
        self.all_oracle_data = CatGenDataIter(self.oracle_samples_list)  # Shuffled all oracle data
        self.gen_data_list = [GenDataIter(self.gen.sample(cfg.batch_size, cfg.batch_size, label_i=i))
                              for i in range(cfg.k_label)]

        self.freeze_dis = cfg.freeze_dis
        self.freeze_clas = cfg.freeze_clas

    def init_model(self):
        if cfg.oracle_pretrain:
            for i in range(cfg.k_label):
                oracle_path = cfg.multi_oracle_state_dict_path.format(i)
                if not os.path.exists(oracle_path):
                    create_multi_oracle(cfg.k_label)
                self.oracle_list[i].load_state_dict(torch.load(oracle_path, map_location='cuda:%d' % cfg.device))

        if cfg.gen_pretrain:
            for i in range(cfg.n_parent):
                self.log.info('Load MLE pretrained generator gen: {}'.format(cfg.pretrained_gen_path + '%d' % i))
                self.parents[i] = torch.load(cfg.pretrained_gen_path + '%d' % i, map_location='cuda:%d' % cfg.device)

        if cfg.CUDA:
            for i in range(cfg.k_label):
                self.oracle_list[i] = self.oracle_list[i].cuda()
            self.gen = self.gen.cuda()
            self.dis = self.dis.cuda()

    def load_gen(self, parent, parent_opt, mle=False):
        self.gen.load_state_dict(parent)
        if mle:
            self.gen_opt.load_state_dict(parent_opt)
            self.gen_opt.zero_grad()
        else:
            self.gen_adv_opt.load_state_dict(parent_opt)
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
            score, fit_score, select_mu = self.evolve_generator(cfg.ADV_g_step)
            d_loss = self.evolve_discriminator(cfg.ADV_d_step, 'ADV')
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
        best_score = np.zeros(cfg.n_parent)
        best_fit = []
        best_child = []
        best_child_opt = []
        best_fake_samples_pred = []
        selected_mutation = []
        count = 0

        for i, (parent, parent_opt) in enumerate(zip(self.parents, self.parent_adv_opts)):
            for j, criterionG in enumerate(self.G_critertion):
                # ===Variation===
                self.load_gen(parent, parent_opt)  # load state dict to self.gen
                # single loss
                self.variation(evo_g_step, criterionG)

                # ===Evaluation===
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

    def evolve_discriminator(self, evo_d_step, phase='PRE'):
        total_loss = []
        all_gen_samples_list = list(
            map(self.merge, *self.best_fake_samples_pred))  # merge all child samples of each category, len=k_label
        all_gen_samples_list = self.shuffle_eval_samples(all_gen_samples_list)
        for step in range(evo_d_step):
            dis_real_samples, dis_gen_samples = self.prepare_dis_data('D', all_gen_samples_list, step)

            # =====Train=====
            d_loss = 0
            all_d_out_real = []
            all_d_out_fake = []
            for (real_samples, fake_samples) in zip(dis_real_samples, dis_gen_samples):
                d_out_real = self.dis(real_samples)
                d_out_fake = self.dis(fake_samples)
                d_loss += self.D_critertion(d_out_real, d_out_fake)
                all_d_out_real.append(d_out_real.view(cfg.batch_size, -1))
                all_d_out_fake.append(d_out_fake.view(cfg.batch_size, -1))

            if cfg.use_all_real_fake:
                all_d_out_real = torch.cat(all_d_out_real, dim=0)
                all_d_out_fake = torch.cat(all_d_out_fake, dim=0)
                all_d_out_real = all_d_out_real[torch.randperm(all_d_out_real.size(0))]
                all_d_out_fake = all_d_out_fake[torch.randperm(all_d_out_fake.size(0))]
                d_loss += self.D_critertion(all_d_out_real, all_d_out_fake)

            self.optimize(self.gen_adv_opt, d_loss, self.gen)
            total_loss.append(d_loss.item())

            if phase == 'PRE':
                self.log.debug('[PRE-epoch %d]In D: d_loss = %.4f', step, d_loss.item())
        if evo_d_step == 0:
            return 0
        return np.mean(total_loss)

    def variation(self, g_step, criterionG):
        """Optimize one child (Generator)"""
        total_loss = []
        for step in range(g_step):
            dis_real_samples, dis_gen_samples = self.prepare_dis_data('G')

            # =====Train=====
            g_loss = 0
            all_d_out_real = []
            all_d_out_fake = []
            for (real_samples, fake_samples) in zip(dis_real_samples, dis_gen_samples):
                d_out_real = self.dis(real_samples)
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

            # self.log.debug('In G: g_loss = %.4f' % g_loss.item())

        if g_step == 0:
            return 0
        return np.mean(total_loss)

    def evaluation(self, eval_type):
        """Evaluation all child, update child score. Note that the eval data should be the same"""
        with torch.no_grad():
            # prepare eval samples
            eval_fake_samples = []
            eval_fake_samples_pred = []
            for label_i in range(cfg.k_label):
                fake_samples, fake_samples_pred = self.gen.sample(cfg.eval_b_num * cfg.batch_size,
                                                                  cfg.eval_b_num * cfg.batch_size,
                                                                  one_hot=True, need_samples=True, label_i=label_i)
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

            elif eval_type == 'rsgan':
                # TODO: rsgan loss at each category
                fake_samples = torch.cat(eval_fake_samples_pred, dim=0)[
                    torch.randperm(eval_fake_samples_pred[0].size(0) * 2)]
                Fq = self.dis(fake_samples[:cfg.batch_size]).mean().cpu().item()
                Fd = 0
                pass
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

    def prepare_dis_data(self, which, all_gen_samples_list=None, step=None):
        assert which == 'D' or which == 'G', 'only support for D and G!!'
        real_samples_list = [
            F.one_hot(self.oracle_data_list[i].random_batch()['target'][:cfg.batch_size],
                      cfg.vocab_size).float().cuda()
            for i in range(cfg.k_label)]
        if which == 'D':
            assert all_gen_samples_list is not None and step is not None, 'samples and step have to be given!'
            gen_samples_list = [
                all_gen_samples_list[i][step * cfg.batch_size:(step + 1) * cfg.batch_size]
                for i in range(cfg.k_label)]
        else:  # 'G'
            gen_samples_list = [
                self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True, label_i=i)
                for i in range(cfg.k_label)]

        return real_samples_list, gen_samples_list

    def shuffle_eval_samples(self, all_eval_samples):
        temp = []
        for i in range(cfg.k_label):
            temp.append(all_eval_samples[i][torch.randperm(all_eval_samples[i].size(0))])
        return temp
