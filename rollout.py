# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : SentiGAN-william
# @FileName     : rollout.py
# @Time         : Created at 2019-03-15
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np


class ROLLOUT():
    def __init__(self, generator, gpu=True):
        self.generator = generator
        self.max_seq_len = generator.max_seq_len
        self.gpu = gpu

    def rollout_mc_search(self, sentences, given_num):
        """
        fill up remain tokens with MC search
        :param sentences: size of batch_size * max_seq_len
        :param given_num:
        :return:
        """
        batch_size = sentences.size()[0]

        # get current state
        hidden = self.generator.init_hidden(batch_size)
        for i in range(given_num):
            inp = sentences[:, i].view(1, -1)
            out, hidden = self.generator(inp, hidden)

        samples = torch.zeros(batch_size, self.max_seq_len).long()
        samples[:, :given_num] = sentences[:, :given_num]

        if self.gpu:
            samples = samples.cuda()

        # MC search
        for i in range(given_num, self.max_seq_len):
            out = torch.multinomial(torch.exp(out), 1)
            samples[:, i] = out.view(-1).data
            inp = out.view(-1)

            out, hidden = self.generator(inp, hidden)

        return samples

    def get_reward(self, sentences, rollout_num, dis, current_k):
        """
        get reward via Monte Carlo search
        :param sentences: size of batch_size * max_seq_len
        :param rollout_num:
        :param dis:
        :param current_k: current training generator
        :return:
        """
        batch_size = sentences.size()[0]
        rewards = torch.zeros([rollout_num * self.max_seq_len, batch_size]).float()
        idx = 0
        for i in range(rollout_num):
            for given_num in range(1, self.max_seq_len + 1):
                samples = self.rollout_mc_search(sentences, given_num)
                # reward = discriminator.batchClassify(samples)
                out = dis.batchClassify(samples)
                out = F.softmax(out, dim=-1)
                # print('out:', out)
                reward = out[:, current_k + 1]
                rewards[idx] = reward
                idx += 1

                # torch.cuda.empty_cache()
                # print('rollout time:',idx)

            # the last token reward
            # reward = discriminator.batchClassify(sentences)
            # out = discriminator.batchClasssifySenti(sentences)
            # out = F.softmax(out, dim=-1)
            # reward = out[:, current_k + 1]
            # rewards[idx] = reward
            # idx += 1

        rewards = torch.Tensor(rewards).cuda()
        rewards = torch.sum(rewards, dim=0) / (rollout_num * self.max_seq_len)
        return rewards

    def get_token_reward(self, sentences, rollout_num, dis, current_k, given_num):
        """
        get reward of each token in sequence via Monte Carlo search
        """
        batch_size = sentences.size()[0]
        rewards = torch.zeros([rollout_num, batch_size]).float()
        idx = 0
        for i in range(rollout_num):
            samples = self.rollout_mc_search(sentences, given_num)
            out = dis.batchClassify(samples)
            out = F.softmax(out, dim=-1)
            reward = out[:, current_k + 1]
            rewards[idx] = reward
            idx += 1

        rewards = torch.Tensor(rewards).cuda()
        rewards = torch.sum(rewards, dim=0) / rollout_num
        return rewards
