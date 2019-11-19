# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : instructor.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import os
import torch
import torch.nn as nn

import config as cfg
from models.Oracle import Oracle
from utils.data_loader import GenDataIter
from utils.helpers import Signal, create_logger, create_oracle, get_fixed_temperature
from utils.text_process import write_tensor


class BasicInstructor:
    def __init__(self, opt):
        self.log = create_logger(__name__, silent=False, to_disk=True,
                                 log_file=cfg.log_filename if cfg.if_test
                                 else [cfg.log_filename, cfg.save_root + 'log.txt'])
        self.sig = Signal(cfg.signal_file)
        self.opt = opt

        # oracle, generator, discriminator
        self.oracle = Oracle(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                             cfg.padding_idx, gpu=cfg.CUDA)
        self.oracle_list = None
        self.dis = None
        self.clas = None

        self.show_config()

        # DataLoader
        if not os.path.exists(cfg.oracle_samples_path.format(cfg.samples_num)) or not cfg.oracle_pretrain:
            create_oracle()
            self.oracle.load_state_dict(torch.load(cfg.oracle_state_dict_path))
        self.oracle_samples = torch.load(cfg.oracle_samples_path.format(cfg.samples_num))
        self.oracle_data = GenDataIter(self.oracle_samples)

        self.gen_data = None
        self.gen_data_list = None
        self.dis_data = None
        self.clas_data = None
        self.oracle_data_list = None

        # Criterion
        self.mle_criterion = nn.NLLLoss()
        self.dis_criterion = None
        self.clas_criterion = None

    def _run(self):
        print('Nothing to run in Basic Instructor!')
        pass

    def _test(self):
        pass

    def init_model(self):
        if cfg.oracle_pretrain:
            if not os.path.exists(cfg.oracle_state_dict_path):
                create_oracle()
            self.oracle.load_state_dict(torch.load(cfg.oracle_state_dict_path))

        if cfg.dis_pretrain:
            self.log.info(
                'Load pretrained discriminator: {}'.format(cfg.pretrained_dis_path))
            self.dis.load_state_dict(torch.load(cfg.pretrained_dis_path))
        if cfg.gen_pretrain:
            self.log.info('Load MLE pretrained generator gen: {}'.format(cfg.pretrained_gen_path))
            self.gen.load_state_dict(torch.load(cfg.pretrained_gen_path, map_location='cuda:{}'.format(cfg.device)))

        if cfg.CUDA:
            self.oracle = self.oracle.cuda()
            self.gen = self.gen.cuda()
            self.dis = self.dis.cuda()

    def train_gen_epoch(self, model, data_loader, criterion, optimizer):
        total_loss = 0
        for i, data in enumerate(data_loader):
            inp, target = data['input'], data['target']
            if cfg.CUDA:
                inp, target = inp.cuda(), target.cuda()

            hidden = model.init_hidden(data_loader.batch_size)
            pred = model.forward(inp, hidden)
            loss = criterion(pred, target.view(-1))
            self.optimize(optimizer, loss, model)
            total_loss += loss.item()
        return total_loss / len(data_loader)

    def train_dis_epoch(self, model, data_loader, criterion, optimizer):
        total_loss = 0
        total_acc = 0
        total_num = 0
        for i, data in enumerate(data_loader):
            inp, target = data['input'], data['target']
            if cfg.CUDA:
                inp, target = inp.cuda(), target.cuda()

            pred = model.forward(inp)
            loss = criterion(pred, target)
            self.optimize(optimizer, loss, model)

            total_loss += loss.item()
            total_acc += torch.sum((pred.argmax(dim=-1) == target)).item()
            total_num += inp.size(0)

        total_loss /= len(data_loader)
        total_acc /= total_num
        return total_loss, total_acc

    @staticmethod
    def eval_gen(model, data_loader, criterion):
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data['input'], data['target']
                if cfg.CUDA:
                    inp, target = inp.cuda(), target.cuda()

                hidden = model.init_hidden(data_loader.batch_size)
                pred = model.forward(inp, hidden)
                loss = criterion(pred, target.view(-1))
                total_loss += loss.item()
        return total_loss / len(data_loader)

    @staticmethod
    def eval_dis(model, data_loader, criterion):
        total_loss = 0
        total_acc = 0
        total_num = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data['input'], data['target']
                if cfg.CUDA:
                    inp, target = inp.cuda(), target.cuda()

                pred = model.forward(inp)
                loss = criterion(pred, target)
                total_loss += loss.item()
                total_acc += torch.sum((pred.argmax(dim=-1) == target)).item()
                total_num += inp.size(0)
            total_loss /= len(data_loader)
            total_acc /= total_num
        return total_loss, total_acc

    @staticmethod
    def optimize_multi(opts, losses):
        for i, (opt, loss) in enumerate(zip(opts, losses)):
            opt.zero_grad()
            loss.backward(retain_graph=True if i < len(opts) - 1 else False)
            opt.step()

    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False):
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()

    def show_config(self):
        """Show parser parameters settings"""
        self.log.info(100 * '=')
        self.log.info('> training arguments:')
        for arg in vars(self.opt):
            self.log.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
        self.log.info(100 * '=')

    def cal_metrics(self, fmt_str=False):
        """
        Calculate metrics
        :param fmt_str: if return format string for logging
        """
        self.gen_data.reset(self.gen.sample(cfg.samples_num, 4 * cfg.batch_size))
        oracle_nll = self.eval_gen(self.oracle,
                                   self.gen_data.loader,
                                   self.mle_criterion)
        gen_nll = self.eval_gen(self.gen,
                                self.oracle_data.loader,
                                self.mle_criterion)
        div_nll = self.eval_gen(self.gen,
                                self.gen_data.loader,
                                self.mle_criterion)

        if fmt_str:
            return 'oracle_NLL = %.4f, gen_NLL = %.4f, div_NLL = %.4f' % (oracle_nll, gen_nll, div_nll)
        return oracle_nll, gen_nll, div_nll

    def cal_metrics_with_label(self, label_i=None):
        assert type(label_i) == int, 'missing label'
        eval_samples = self.gen.sample(cfg.samples_num, 8 * cfg.batch_size, label_i=label_i)
        self.gen_data_list[label_i].reset(eval_samples)
        oracle_nll = self.eval_gen(self.oracle_list[label_i],
                                   self.gen_data_list[label_i].loader,
                                   self.mle_criterion, label_i)
        div_nll = self.eval_gen(self.gen,
                                self.gen_data_list[label_i].loader,
                                self.mle_criterion, label_i)

        # Evaluation Classifier accuracy
        self.clas_data.reset([eval_samples], label_i)
        _, c_acc = self.eval_dis(self.clas, self.clas_data.loader, self.clas_criterion)

        return oracle_nll, div_nll, c_acc

    def comb_metrics(self, fmt_str=False):
        oracle_nll, div_nll, clas_acc = [], [], []
        for label_i in range(cfg.k_label):
            o_nll, s_nll, acc = self.cal_metrics_with_label(label_i)
            oracle_nll.append(round(o_nll, 4))
            div_nll.append(round(s_nll, 4))
            clas_acc.append(round(acc, 4))

        if fmt_str:
            return 'oracle_NLL = %s, div_NLL = %s, clas_acc = %s' % (
                oracle_nll, div_nll, clas_acc)
        return oracle_nll, div_nll, clas_acc

    def cal_metrics_with_label(self, label_i=None):
        assert type(label_i) == int, 'missing label'
        eval_samples = self.gen.sample(cfg.samples_num, 8 * cfg.batch_size, label_i=label_i)
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
        # oracle_nll, gen_nll, self_nll = 0, 0, 0

        # Evaluation Classifier accuracy
        self.clas_data.reset([eval_samples], label_i)
        _, c_acc = self.eval_dis(self.clas, self.clas_data.loader, self.clas_criterion)

        return oracle_nll, gen_nll, self_nll, c_acc

    def comb_metrics(self, fmt_str=False):
        oracle_nll, gen_nll, self_nll, clas_acc = [], [], [], []
        for label_i in range(cfg.k_label):
            o_nll, g_nll, s_nll, acc = self.cal_metrics_with_label(label_i)
            oracle_nll.append(round(o_nll, 4))
            gen_nll.append(round(g_nll, 4))
            self_nll.append(round(s_nll, 4))
            clas_acc.append(round(acc, 4))

        if fmt_str:
            return 'oracle_NLL = %s, gen_NLL = %s, self_NLL = %s, clas_acc = %s' % (
                oracle_nll, gen_nll, self_nll, clas_acc)
        return oracle_nll, gen_nll, self_nll, clas_acc

    def _save(self, phrase, epoch):
        """Save model state dict and generator's samples"""
        if phrase != 'ADV':
            torch.save(self.gen.state_dict(), cfg.save_model_root + 'gen_{}_{:05d}.pt'.format(phrase, epoch))
        save_sample_path = cfg.save_samples_root + 'samples_{}_{:05d}.txt'.format(phrase, epoch)
        samples = self.gen.sample(cfg.batch_size, cfg.batch_size)
        write_tensor(save_sample_path, samples)

    def update_temperature(self, i, N):
        self.gen.temperature.data = torch.Tensor([get_fixed_temperature(cfg.temperature, i, N, cfg.temp_adpt)])
        if cfg.CUDA:
            self.gen.temperature.data = self.gen.temperature.data.cuda()
