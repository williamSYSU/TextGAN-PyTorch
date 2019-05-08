# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : instructor.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import sys
import torch
import torch.optim as optim
from data_utils import GenDataIter, DisDataIter

import helpers
from helpers import Signal
import config as cfg

from models.discriminator import CNNDiscriminator
from models.generator import LSTMGenerator
from models.Oracle import Oracle


class BasicInstructor:
    def __init__(self, opt):
        self.log = open(cfg.log_filename + '.txt', 'w') if not cfg.if_test else None
        self.sig = Signal(cfg.signal_file)
        self.opt = opt

        # oracle, generator, discriminator
        self.oracle = Oracle(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                             cfg.padding_idx, gpu=cfg.CUDA)

        self.show_config()

        # DataLoader
        self.oracle_samples = torch.load(cfg.oracle_samples_path.format(cfg.samples_num))
        self.oracle_data = GenDataIter(self.oracle_samples)

    def _run(self):
        print('Nothing to run in Basic Instructor!')
        pass

    def _test(self):
        pass

    def init_model(self):
        if cfg.oracle_pretrain:
            self.oracle.load_state_dict(torch.load(cfg.oracle_state_dict_path))

        if cfg.dis_pretrain:
            self._print(
                'Load pretrain_generator discriminator: {}\n'.format(cfg.pretrained_dis_path))
            self.dis.load_state_dict(torch.load(cfg.pretrained_dis_path))
        if cfg.gen_pretrain:
            self._print('Load MLE pretrain_generator gen: {}\n'.format(cfg.pretrained_gen_path))
            self.gen.load_state_dict(torch.load(cfg.pretrained_gen_path))

        if cfg.CUDA:
            self.oracle = self.oracle.cuda()
            self.gen = self.gen.cuda()
            self.dis = self.dis.cuda()

    def train_gen_epoch(self, model, data_loader, criterion, optimizer):
        # model.train()
        total_loss = 0
        for i, data in enumerate(data_loader):
            inp, target = data['input'], data['target']
            if cfg.CUDA:
                inp, target = inp.cuda(), target.cuda()

            hidden = model.init_hidden(data_loader.batch_size)
            pred = model.forward(inp, hidden)
            loss = criterion(pred, target.view(-1))
            self.optimize(optimizer, loss)
            total_loss += loss.item()
        return total_loss / len(data_loader)

    def train_dis_epoch(self, model, data_loader, criterion, optimizer):
        # model.train()
        total_loss = 0
        total_acc = 0
        for i, data in enumerate(data_loader):
            inp, target = data['input'], data['target']
            if cfg.CUDA:
                inp, target = inp.cuda(), target.cuda()

            pred = model.forward(inp)
            loss = criterion(pred, target)
            self.optimize(optimizer, loss)

            total_loss += loss.item()
            total_acc += torch.sum((pred.argmax(dim=-1) == target)).item()

        total_loss /= len(data_loader)
        total_acc /= len(data_loader)
        return total_loss, total_acc

    @staticmethod
    def eval_gen(model, data_loader, criterion):
        # model.eval()
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
        # model.eval()
        total_loss = 0
        total_acc = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data['input'], data['target']
                if cfg.CUDA:
                    inp, target = inp.cuda(), target.cuda()

                pred = model.forward(inp)
                loss = criterion(pred, target)

                total_loss += loss.item()
                total_acc += torch.sum((pred.argmax(dim=-1) == target)).item()

            total_loss /= len(data_loader)
            total_acc /= len(data_loader)
        return total_loss, total_acc

    @staticmethod
    def optimize_multi(opts, losses):
        for i, (opt, loss) in enumerate(zip(opts, losses)):
            opt.zero_grad()
            loss.backward(retain_graph=True if i < len(opts) - 1 else False)
            opt.step()

    @staticmethod
    def optimize(opt, loss, retain_graph=False):
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        opt.step()

    # TODO: remove
    @staticmethod
    def get_nll(data_loader, evaluator):
        with torch.no_grad():
            nll_loss = 0
            for data in data_loader:
                inp, target = data['input'], data['target']
                if cfg.CUDA:
                    inp, target = inp.cuda(), target.cuda()

                nll_loss += evaluator.batchNLLLoss(inp, target).item()

        return nll_loss / len(data_loader)

    def _print(self, content):
        print(content, end='')
        sys.stdout.flush()
        if not cfg.if_test:
            self.log.write(content)

    def show_config(self):
        self._print(100 * '=' + '\n')
        self._print('> training arguments:\n')
        for arg in vars(self.opt):
            self._print('>>> {0}: {1}\n'.format(arg, getattr(self.opt, arg)))
        self._print(100 * '=' + '\n')
