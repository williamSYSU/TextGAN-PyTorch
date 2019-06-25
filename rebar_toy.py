# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : rebar_toy.py
# @Time         : Created at 2019-06-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Parameters
VOCAB_SIZE = 1000
BATCH_SIZE = 32
EMBED_DIM = 16
HIDDEN_DIM = 10
EPOCH = 200
THETA = 1.0


def prepare_data(vocab_size=1000):
    data = torch.LongTensor(list(range(vocab_size)))
    target = torch.LongTensor([0] * vocab_size)
    target[torch.randperm(vocab_size)[:vocab_size // 2]] = 1

    perm = torch.randperm(vocab_size)
    data = data[perm]
    target = target[perm]

    return data, target


class NVIL(nn.Module):
    def __init__(self):
        super(NVIL, self).__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.baseline = nn.Linear(EMBED_DIM, 1)

    def forward(self, inp):
        emb = self.embed(inp)
        nvil = self.baseline(emb).squeeze(1)

        return nvil


class Toy(nn.Module):
    def __init__(self):
        super(Toy, self).__init__()

        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.embed2hid = nn.Linear(EMBED_DIM, VOCAB_SIZE)
        self.hid2out = nn.Linear(VOCAB_SIZE, 2)

    def get_params(self):
        params = list()

        params += self.embed.parameters()
        params += self.embed2hid.parameters()

        return params

    def forward(self, inp):
        emb = self.embed(inp)
        hidden = self.embed2hid(emb)

        pred = F.softmax(hidden, dim=-1)
        samples = torch.argmax(pred, dim=-1).detach()
        samples_onehot = F.one_hot(samples, VOCAB_SIZE).float()

        gumbel_u = F.softmax(self.add_gumbel_u(hidden), dim=-1)
        gumbel_v = F.softmax(self.add_gumbel_v(hidden, samples), dim=-1)

        hardlogQ = self.log_likelihood(samples_onehot, hidden)

        return hidden, pred, samples, samples_onehot, gumbel_u, gumbel_v, hardlogQ

    @staticmethod
    def add_gumbel_u(hidden, eps=1e-10):
        gumbel_u = torch.rand(hidden.size()).cuda()
        g_t = -torch.log(-torch.log(gumbel_u + eps) + eps)
        out = hidden + g_t
        return out

    @staticmethod
    def add_gumbel_v(hidden, samples, eps=1e-10):
        gumbel_v = torch.rand(hidden.size()).cuda()
        p = torch.exp(hidden)
        b = gumbel_v[torch.arange(gumbel_v.size(0)), samples].unsqueeze(1)
        v_1 = -torch.log(-(torch.log(gumbel_v + eps) / (p + eps)) - torch.log(b))
        v_2 = -torch.log(-torch.log(gumbel_v[torch.arange(gumbel_v.size(0)), samples] + eps) + eps)
        v_1_clo = v_1
        v_1_clo[torch.arange(gumbel_v.size(0)), samples] = v_2
        out = v_1_clo

        return out

    @staticmethod
    def log_likelihood(y, log_y_hat, eps=1e-10):
        """Computes log likelihood.

        Args:
          y: observed data
          log_y_hat: parameters of the variables

        Returns:
          log_likelihood
        """
        return torch.sum(y * torch.log(torch.clamp(F.softmax(log_y_hat, dim=-1), eps, 1)), 1)

    def cal_loss(self, label, pred_out, hard_out, soft_out, hardlogQ):
        criterion = nn.CrossEntropyLoss(reduction='none')

        pred_loss = -criterion(pred_out, label)
        hard_loss = -criterion(hard_out, label)
        soft_loss = -criterion(soft_out, label)

        de_hard_loss = hard_loss.detach()
        de_soft_loss = soft_loss.detach()

        f = - torch.mean(de_hard_loss * hardlogQ)
        h = THETA * torch.mean(de_soft_loss * hardlogQ - pred_loss + soft_loss)

        return f, h

    def cal_loss_v2(self, label, out):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, label)

        return loss

    def cal_loss_v3(self, label, pred_out, hard_out, soft_out, hardlogQ, nvil):
        criterion = nn.CrossEntropyLoss(reduction='none')

        pred_loss = -criterion(pred_out, label)
        hard_loss = -criterion(hard_out, label)
        soft_loss = -criterion(soft_out, label)

        de_hard_loss = hard_loss.detach()
        de_soft_loss = soft_loss.detach()

        f_target = (de_hard_loss - nvil) * hardlogQ + hard_loss
        f = - torch.mean(f_target)

        h = THETA * torch.mean(de_soft_loss * hardlogQ - pred_loss + soft_loss)

        return f, h


def train():
    model = Toy()
    nvil_model = NVIL()
    model = model.cuda()
    nvil_model = nvil_model.cuda()
    opt = torch.optim.Adam(model.get_params(), lr=1e-2)

    data, target = prepare_data(VOCAB_SIZE)

    for step in range(EPOCH):
        total_num = 0
        total_acc = 0
        total_loss = []
        for i in range(0, VOCAB_SIZE, BATCH_SIZE):
            inp, label = data[i:i + BATCH_SIZE].cuda(), target[i:i + BATCH_SIZE].cuda()
            hidden, _, _, samples_onehot, gumbel_u, gumbel_v, hardlogQ = model(inp)

            hidden_out = model.hid2out(hidden)
            pred_out = model.hid2out(gumbel_u)
            hard_out = model.hid2out(samples_onehot)
            soft_out = model.hid2out(gumbel_v)
            f, h = model.cal_loss(label, pred_out, hard_out, soft_out, hardlogQ)
            # pred_loss = model.cal_loss_v2(label, pred_out)

            # nvil = nvil_model(inp)
            # f, h = model.cal_loss_v3(label, pred_out, hard_out, soft_out, hardlogQ, nvil)

            opt.zero_grad()
            f_plus_h = []
            f_grad = torch.autograd.grad(f, list(model.get_params()), create_graph=True)
            h_grad = torch.autograd.grad(h, list(model.get_params()))
            # h_grad = torch.autograd.grad(h, list(model.get_params()), retain_graph=True)

            for a, b in zip(f_grad, h_grad):
                flag_a = torch.sum(torch.isnan(a))
                flag_b = torch.sum(torch.isnan(b))
                if flag_a > 0:
                    f_plus_h.append(b)
                elif flag_b > 0:
                    f_plus_h.append(a)
                else:
                    f_plus_h.append(a + b)

            for i, param in enumerate(model.get_params()):
                param.grad = f_plus_h[i]
            # opt.step()

            # TODO: NVIL loss
            # extra_grad = [i.view(-1).pow(2) for i in f_plus_h]
            # extra_grad = torch.mean(torch.cat(extra_grad, dim=0))
            # c = torch.autograd.grad(extra_grad, model.get_params() + list(nvil_model.parameters()))
            # opt.zero_grad()
            # for i, param in enumerate(model.get_params() + list(nvil_model.parameters())):
            #     param.grad = c[i]

            # pred_loss.backward()
            # (f+h).backward()
            opt.step()

            total_num += BATCH_SIZE
            total_acc += torch.sum((hard_out.argmax(dim=-1) == label)).item()
            total_loss.append((f + h).item())
            # total_acc += torch.sum((pred_out.argmax(dim=-1) == label)).item()
            # total_loss.append(pred_loss.item())

        print('epoch %d: loss: %.4f, acc: %.4f' % (step, np.mean(total_loss), total_acc / total_num))


if __name__ == '__main__':
    train()
