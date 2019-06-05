# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : instructor.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch

import config as cfg
from utils.data_loader import GenDataIter
from utils.helpers import Signal, create_logger
from utils.text_process import load_dict, write_tokens, tensor_to_tokens


class BasicInstructor:
    def __init__(self, opt):
        self.log = create_logger(__name__, silent=False, to_disk=False if cfg.if_test else True,
                                 log_file=None if cfg.if_test
                                 else [cfg.log_filename, cfg.save_root + 'log.txt'])
        self.sig = Signal(cfg.signal_file)
        self.opt = opt
        self.show_config()

        # load dictionary
        self.word_index_dict, self.index_word_dict = load_dict(cfg.dataset)

        # Dataloader
        self.oracle_data = GenDataIter(cfg.train_data)
        self.test_data = GenDataIter(cfg.test_data)

    def _run(self):
        print('Nothing to run in Basic Instructor!')
        pass

    def _test(self):
        pass

    def init_model(self):
        if cfg.dis_pretrain:
            self.log.info(
                'Load pretrain_generator discriminator: {}'.format(cfg.pretrained_dis_path))
            self.dis.load_state_dict(torch.load(cfg.pretrained_dis_path))
        if cfg.gen_pretrain:
            self.log.info('Load MLE pretrain_generator gen: {}'.format(cfg.pretrained_gen_path))
            self.gen.load_state_dict(torch.load(cfg.pretrained_gen_path))

        if cfg.CUDA:
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
        opt.step()

    def show_config(self):
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
        self.bleu3.test_text = tensor_to_tokens(self.gen_data.target, self.index_word_dict)
        bleu3_score = self.bleu3.get_score(ignore=False)

        gen_nll = self.eval_gen(self.gen,
                                self.oracle_data.loader,
                                self.mle_criterion)

        if fmt_str:
            return 'BLEU-3 = %.4f, gen_NLL = %.4f,' % (bleu3_score, gen_nll)
        return bleu3_score, gen_nll

    def _save(self, phrase, epoch):
        """Save model state dict and generator's samples"""
        torch.save(self.gen.state_dict(), cfg.save_model_root + 'gen_{}_{:05d}.pt'.format(phrase, epoch))
        save_sample_path = cfg.save_samples_root + 'samples_{}_{:05d}.txt'.format(phrase, epoch)
        samples = self.gen.sample(cfg.batch_size, cfg.batch_size)
        write_tokens(save_sample_path, tensor_to_tokens(samples, self.index_word_dict))
