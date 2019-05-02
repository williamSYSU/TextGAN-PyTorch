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

import helpers
from helpers import Signal
import config as cfg

from models.discriminator import CNNDiscriminator
from models.generator import LSTMGenerator
from models.Oracle import Oracle


class BasicInstructor:
    def __init__(self, opt):
        self.log = open(cfg.log_filename + '.txt', 'w')
        self.sig = Signal(cfg.signal_file)
        self.opt = opt

        # oracle, generator, discriminator
        self.oracle = Oracle(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                             cfg.padding_idx, gpu=cfg.CUDA)
        self.gen = LSTMGenerator(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                                 cfg.padding_idx, gpu=cfg.CUDA)
        self.dis = CNNDiscriminator(cfg.dis_embed_dim, cfg.vocab_size, cfg.dis_filter_sizes, cfg.dis_num_filters,
                                    cfg.k_label, cfg.padding_idx, gpu=cfg.CUDA)
        self.init_model()

        self.show_config()

    def _run(self):
        print('Nothing to run in Basic Instructor!')
        pass

    def init_model(self):
        if cfg.oracle_pretrain:
            self.oracle.load_state_dict(torch.load(cfg.oracle_state_dict_path))

        if cfg.dis_pretrain:
            self._print(
                'Load pretrain_generator discriminator: {}\n'.format(cfg.pretrained_dis_path.format(cfg.k_label)))
            self.dis.load_state_dict(torch.load(cfg.pretrained_dis_path.format(cfg.k_label)))
        if cfg.gen_pretrain:
            self._print('Load MLE pretrain_generator gen: {}\n'.format(cfg.pretrained_gen_path))
            self.gen.load_state_dict(torch.load(cfg.pretrained_gen_path))

        if cfg.CUDA:
            self.oracle = self.oracle.cuda()
            self.gen = self.gen.cuda()
            self.dis = self.dis.cuda()

    def eval_gen(self):
        self.gen.eval()
        self.dis.eval()
        with torch.no_grad():
            # sample from gen and compute oracle NLL
            oracle_nll = self.get_nll(self.gen.sample(cfg.samples_num, cfg.batch_size), self.oracle)

            gen_nll = self.get_nll(self.oracle.sample(cfg.samples_num, cfg.batch_size), self.gen)
            # gen_nll = 0

        return oracle_nll, gen_nll

    def eval_dis(self, val_inp, val_target):
        self.dis.eval()
        with torch.no_grad():
            val_size = 2 * cfg.samples_num
            val_acc = 0
            for i in range(0, val_size, 8 * cfg.batch_size):  # 8 * batch_size for faster
                inp, target = val_inp[i:i + 8 * cfg.batch_size], val_target[i:i + 8 * cfg.batch_size]

                if cfg.CUDA:
                    inp = inp.cuda()
                    target = target.cuda()

                val_pred = self.dis.batchClassify(inp)
                val_acc += torch.sum((val_pred.argmax(dim=-1) == target)).item() / val_size

        return val_acc

    def optimize_multi(self, opts, losses):
        for i, (opt, loss) in enumerate(zip(opts, losses)):
            opt.zero_grad()
            loss.backward(retain_graph=True if i < len(opts) - 1 else False)
            opt.step()

    def optimize(self, opt, loss):
        opt.zero_grad()
        loss.backward()
        opt.step()

    def get_nll(self, samples, gen):
        nll_loss = 0
        for i in range(0, len(samples), cfg.batch_size):
            inp, target = helpers.prepare_generator_batch(samples[i:i + cfg.batch_size], cfg.start_letter,
                                                          cfg.CUDA)
            oracle_loss = gen.batchNLLLoss(inp, target) / cfg.max_seq_len
            nll_loss += oracle_loss.data.item()

        return nll_loss / (len(samples) // cfg.batch_size)

    def _print(self, content):
        print(content, end='')
        sys.stdout.flush()
        self.log.write(content)

    def show_config(self):
        self._print(100 * '=' + '\n')
        self._print('> training arguments:\n')
        for arg in vars(self.opt):
            self._print('>>> {0}: {1}\n'.format(arg, getattr(self.opt, arg)))
        self._print(100 * '=' + '\n')
