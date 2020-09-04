# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : rollout.py
# @Time         : Created at 2019-03-15
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import copy
import torch
import torch.nn.functional as F


class ROLLOUT:
    def __init__(self, gen, gpu=True):
        self.gen = gen
        self.old_model = copy.deepcopy(gen)
        self.max_seq_len = gen.max_seq_len
        self.vocab_size = gen.vocab_size
        self.step_size = gen.step_size if gen.name == 'leakgan' else 0
        self.goal_out_size = gen.goal_out_size if gen.name == 'leakgan' else 0
        self.gpu = gpu

    def rollout_mc_search(self, sentences, given_num):
        """
        fill up remain tokens with MC search
        :param sentences: size of batch_size * max_seq_len
        :param given_num:
        :return:
        """
        batch_size = sentences.size(0)

        # get current state
        hidden = self.gen.init_hidden(batch_size)
        # for i in range(given_num):
        inp = sentences[:, :given_num]
        out, hidden = self.gen.forward(inp, hidden, need_hidden=True)
        out = out.view(batch_size, -1, self.vocab_size)[:, -1]

        samples = torch.zeros(batch_size, self.max_seq_len).long()
        samples[:, :given_num] = sentences[:, :given_num]

        if self.gpu:
            samples = samples.cuda()

        # MC search
        for i in range(given_num, self.max_seq_len):
            out = torch.multinomial(torch.exp(out), 1)
            samples[:, i] = out.view(-1).data
            inp = out.view(-1)

            out, hidden = self.gen.forward(inp, hidden, need_hidden=True)

        return samples

    def rollout_mc_search_leakgan(self, sentences, dis, given_num):

        batch_size, seq_len = sentences.size()

        goal_array = torch.zeros((batch_size, seq_len + 1, self.goal_out_size))

        work_hidden = self.gen.init_hidden(batch_size)
        mana_hidden = self.gen.init_hidden(batch_size)
        real_goal = self.gen.goal_init[:batch_size, :]
        out = 0

        if self.gpu:
            goal_array = goal_array.cuda()
            real_goal = real_goal.cuda()

        # get current state
        for i in range(given_num):
            # Get feature.
            dis_inp = torch.zeros(batch_size, seq_len).long()
            dis_inp[:, :i + 1] = sentences[:, :i + 1]  # cut sentences
            leak_inp = sentences[:, i]
            if self.gpu:
                dis_inp = dis_inp.cuda()
                leak_inp = leak_inp.cuda()
            feature = dis.get_feature(dis_inp).unsqueeze(0)

            # Get output of one token
            # cur_goal: batch_size * 1 * goal_out_size
            out, cur_goal, work_hidden, mana_hidden = self.gen(i, leak_inp, work_hidden, mana_hidden,
                                                               feature, real_goal, train=True)

            # Save goal and update last_goal
            goal_array[:, i, :] = cur_goal.squeeze(1)
            if i > 0 and i % self.step_size == 0:
                real_goal = torch.sum(goal_array[:, i - 3:i + 1, :], dim=1)
                if i / self.step_size == 1:
                    real_goal += self.gen.goal_init[:batch_size, :]

        samples = torch.zeros(batch_size, self.max_seq_len).long()
        samples[:, :given_num] = sentences[:, :given_num]

        # MC search
        for i in range(given_num, self.max_seq_len):
            # Sample one token
            out = torch.multinomial(torch.exp(out), 1).view(-1)  # [num_samples] (sampling from each row)
            samples[:, i] = out.data

            # Get feature
            dis_inp = samples
            if self.gpu:
                dis_inp = dis_inp.cuda()
            feature = dis.get_feature(dis_inp).unsqueeze(0)
            leak_inp = out

            # Get output of one token
            # cur_goal: batch_size * 1 * goal_out_size
            out, cur_goal, work_hidden, mana_hidden = self.gen(i, leak_inp, work_hidden, mana_hidden,
                                                               feature, real_goal, train=True)

            # Save goal and update last_goal
            goal_array[:, i, :] = cur_goal.squeeze(1)
            if i > 0 and i % self.step_size == 0:
                real_goal = torch.sum(goal_array[:, i - 3:i + 1, :], dim=1)
                if i / self.step_size == 1:
                    real_goal += self.gen.goal_init[:batch_size, :]

        if self.gpu:
            samples = samples.cuda()

        return samples

    def get_reward(self, sentences, rollout_num, dis, current_k=0):
        """
        get reward via Monte Carlo search
        :param sentences: size of batch_size * max_seq_len
        :param rollout_num:
        :param dis:
        :param current_k: current training gen
        :return: reward: [batch_size]
        """
        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards = torch.zeros([rollout_num * self.max_seq_len, batch_size]).float()
            if self.gpu:
                rewards = rewards.cuda()
            idx = 0
            for i in range(rollout_num):
                for given_num in range(1, self.max_seq_len + 1):
                    samples = self.rollout_mc_search(sentences, given_num)
                    out = dis.forward(samples)
                    out = F.softmax(out, dim=-1)
                    reward = out[:, current_k + 1]
                    rewards[idx] = reward
                    idx += 1

        # rewards = torch.mean(rewards, dim=0)
        rewards = torch.mean(rewards.view(batch_size, self.max_seq_len, rollout_num), dim=-1)
        return rewards

    def get_reward_leakgan(self, sentences, rollout_num, dis, current_k):
        """
        get reward via Monte Carlo search for LeakGAN
        :param sentences: size of batch_size * max_seq_len
        :param rollout_num:
        :param dis:
        :param current_k: current training gen

        :return: reward: batch_size * (max_seq_len / step_size)
        """
        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards = torch.zeros([rollout_num * (self.max_seq_len // self.step_size), batch_size]).float()
            if self.gpu:
                rewards = rewards.cuda()
            idx = 0
            for i in range(rollout_num):
                for t in range(self.max_seq_len // self.step_size):
                    given_num = t * self.step_size + 1  # 1, 5, 9, ..
                    samples = self.rollout_mc_search_leakgan(sentences, dis, given_num)
                    out = dis(samples)
                    out = F.softmax(out, dim=-1)
                    reward = out[:, current_k + 1]
                    rewards[idx] = reward
                    idx += 1

        rewards = rewards.view(batch_size, self.max_seq_len // self.step_size, rollout_num)
        rewards = torch.mean(rewards, dim=-1)
        return rewards

    def get_token_reward(self, sentences, rollout_num, dis, current_k, given_num):
        """
        get reward of each token in sequence via Monte Carlo search
        """
        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards = torch.zeros([rollout_num, batch_size]).float()
            idx = 0
            for i in range(rollout_num):
                samples = self.rollout_mc_search(sentences, given_num)
                out = dis(samples)
                out = F.softmax(out, dim=-1)
                reward = out[:, current_k + 1]
                rewards[idx] = reward
                idx += 1

        rewards = torch.Tensor(rewards).cuda()
        rewards = torch.sum(rewards, dim=0) / rollout_num
        return rewards

    def get_reward_csgan(self, target, rollout_num, csgan_clas):
        pass
