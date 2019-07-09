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
EPOCH = 10000
NUM_LAYER = 20
NVIL_LAYER = 1
ETA = 1.0

torch.cuda.set_device(1)


# ETA = 0.989


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
        self.emb2hid = nn.Linear(EMBED_DIM, EMBED_DIM)  # FC version, shared
        # self.fc_list = nn.ModuleList([nn.Linear(EMBED_DIM, EMBED_DIM) for _ in range(NUM_LAYER)])  # FC version

        # LSTM version
        # self.hidden_dim = EMBED_DIM
        # self.lstm = nn.LSTM(EMBED_DIM, EMBED_DIM, batch_first=True)

        # RMC version
        # self.hidden_dim = 1 * 2 * 256
        # self.lstm = RelationalMemory(mem_slots=1, head_size=256, input_size=EMBED_DIM,
        #                              num_heads=2, return_all_outputs=True)

        # self.lstm2hid = nn.Linear(self.hidden_dim, VOCAB_SIZE)    # LSTM version

        self.baseline = nn.Linear(EMBED_DIM, 1)  # FC version
        # self.baseline = nn.Linear(VOCAB_SIZE, 1)  # LSTM version

        self.eta = nn.Parameter(torch.zeros(27))  # LSTM: 13, Linear: 9, RMC: 27

    def forward(self, inp):
        # >>> FC version
        # emb = self.embed(inp)
        # for i, fc in enumerate(self.fc_list):
        #     if i == 0:
        #         hidden = torch.tanh(fc(emb))
        #     else:
        #         hidden = torch.tanh(fc(hidden))
        # nvil = self.baseline(hidden).squeeze(1)

        # >>> FC version, shared
        emb = self.embed(inp)
        for i in range(NVIL_LAYER):
            if i == 0:
                hidden = torch.tanh(self.emb2hid(emb))
            else:
                hidden = torch.tanh(self.emb2hid(hidden))
        nvil = self.baseline(hidden).squeeze(1)

        # >>> LSTM version
        # lstm_hid = None
        # # lstm_hid = self.init_hidden()
        # for i in range(NUM_LAYER):
        #     emb = self.embed(inp).unsqueeze(1)
        #     out, lstm_hid = self.lstm(emb, lstm_hid)
        #     hidden = self.lstm2hid(out.squeeze(1))
        #     inp = torch.multinomial(torch.exp(F.softmax(hidden, dim=-1)), 1).squeeze(1)
        # nvil = self.baseline(hidden).squeeze(1)
        # nvil = self.baseline(out).squeeze(1)

        return nvil

    @staticmethod
    def change(eta):
        eta = 2 * torch.sigmoid(eta)
        return eta

    def multiply_by_eta(self, grads):
        res = []
        for i, g in enumerate(grads):
            if g is None:
                res.append(g)
            else:
                res.append(g * self.change(self.eta[i]))
        return res

    def init_hidden(self, batch_size=BATCH_SIZE):
        """init RMC memory"""
        memory = self.lstm.initial_state(batch_size)
        memory = self.lstm.repackage_hidden(memory)  # detch memory at first
        return memory.cuda()

    def get_params(self):
        params = list()

        params += self.embed.parameters()
        params += self.emb2hid.parameters()
        params += self.baseline.parameters()
        return params


class Toy(nn.Module):
    def __init__(self):
        super(Toy, self).__init__()

        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)

        # FC version
        # self.embed2hid = nn.Linear(EMBED_DIM, VOCAB_SIZE)

        # LSTM version
        self.hidden_dim = EMBED_DIM
        self.lstm = nn.LSTM(EMBED_DIM, EMBED_DIM, batch_first=True)

        # RMC version
        # self.hidden_dim = 1 * 2 * 256
        # self.lstm = RelationalMemory(mem_slots=1, head_size=256, input_size=EMBED_DIM,
        #                              num_heads=2, return_all_outputs=True)

        self.lstm2hid = nn.Linear(self.hidden_dim, VOCAB_SIZE)
        self.hid2out = nn.Linear(VOCAB_SIZE, 2)  # last layer
        self.last_lstm = nn.LSTM(VOCAB_SIZE, VOCAB_SIZE, batch_first=True)

    def get_params(self):
        params = list()

        params += self.embed.parameters()
        # params += self.embed2hid.parameters()  # FC version
        params += self.lstm.parameters()  # LSTM version
        params += self.lstm2hid.parameters()  # LSTM version
        # params += self.hid2out.parameters()
        # params += self.last_lstm.parameters()

        return params

    # >>> one layer
    # def forward(self, inp):
    #     emb = self.embed(inp)
    #     hidden = self.embed2hid(emb)
    #     pred = F.softmax(hidden, dim=-1)
    #
    #     gumbel_u = F.softmax(self.add_gumbel_u(hidden), dim=-1)
    #     samples = torch.argmax(gumbel_u, dim=-1).detach()  # use gumbel_u !!!
    #     samples_onehot = F.one_hot(samples, VOCAB_SIZE).float()
    #
    #     gumbel_v = F.softmax(self.add_gumbel_v(hidden, samples), dim=-1)
    #     hardlogQ = self.log_likelihood(samples_onehot, hidden)
    #
    #     return hidden, pred, samples, samples_onehot, gumbel_u, gumbel_v, hardlogQ

    # >>> use u_to_v
    # def forward(self, inp):
    #     emb = self.embed(inp)
    #     hidden = self.embed2hid(emb)
    #
    #     pred = F.softmax(hidden, dim=-1)
    #
    #     u = torch.rand(hidden.size()).cuda()
    #     gumbel_u = F.softmax(hidden + (-torch.log(-torch.log(u + 1e-10) + 1e-10)), dim=-1)
    #     samples = torch.argmax(gumbel_u, dim=-1).detach()  # use gumbel_u !!!
    #     samples_onehot = F.one_hot(samples, VOCAB_SIZE).float()
    #
    #     v = self._u_to_v_poly(hidden, u, samples)
    #     gumbel_v = F.softmax(hidden + (-torch.log(-torch.log(v + 1e-10) + 1e-10)), dim=-1)
    #
    #     hardlogQ = self.log_likelihood(samples_onehot, hidden)
    #
    #     return hidden, pred, samples, samples_onehot, gumbel_u, gumbel_v, hardlogQ

    # >>> two layers
    # def forward(self, inp):
    #     # first layer
    #     emb = self.embed(inp)
    #     hidden = self.embed2hid(emb)
    #     pred = F.softmax(hidden, dim=-1)
    #     gumbel_u = F.softmax(self.add_gumbel_u(hidden), dim=-1)
    #     samples = torch.argmax(gumbel_u, dim=-1).detach()  # use gumbel_u !!!
    #     samples_onehot = F.one_hot(samples, VOCAB_SIZE).float()
    #     gumbel_v = F.softmax(self.add_gumbel_v(hidden, samples), dim=-1)
    #     hardlogQ = self.log_likelihood(samples_onehot, hidden)
    #
    #     # second layer
    #     emb_2 = self.embed(samples)
    #     hidden_2 = self.embed2hid(emb_2)
    #     pred_2 = F.softmax(hidden_2, dim=-1)
    #     gumbel_u_2 = F.softmax(self.add_gumbel_u(hidden_2), dim=-1)
    #     samples_2 = torch.argmax(gumbel_u_2, dim=-1).detach()  # use gumbel_u !!!
    #     samples_onehot_2 = F.one_hot(samples_2, VOCAB_SIZE).float()
    #     gumbel_v_2 = F.softmax(self.add_gumbel_v(hidden_2, samples_2), dim=-1)
    #     hardlogQ_2 = self.log_likelihood(samples_onehot_2, hidden_2)
    #
    #     # cat
    #     pred_out = torch.stack((pred, pred_2), dim=1)
    #     samples_out = torch.stack((samples, samples_2), dim=1)
    #     samples_onehot_out = torch.stack((samples_onehot, samples_onehot_2), dim=1)
    #     gumbel_u_out = torch.stack((gumbel_u, gumbel_u_2), dim=1)
    #     gumbel_v_out = torch.stack((gumbel_v, gumbel_v_2), dim=1)
    #     hardlogQ_out = torch.stack((hardlogQ, hardlogQ_2), dim=1)
    #
    #     return hidden_2, pred_out, samples_out, samples_onehot_out, gumbel_u_out, gumbel_v_out, hardlogQ_out

    # >>> multilayer, more than two layers
    # def step(self, inp):
    #     emb = self.embed(inp)
    #     hidden = self.embed2hid(emb)
    #
    #     pred = F.softmax(hidden, dim=-1)
    #     gumbel_u = F.softmax(self.add_gumbel_u(hidden), dim=-1)
    #     samples = torch.argmax(gumbel_u, dim=-1).detach()  # use gumbel_u !!!
    #     samples_onehot = F.one_hot(samples, VOCAB_SIZE).float()
    #     gumbel_v = F.softmax(self.add_gumbel_v(hidden, samples), dim=-1)
    #     hardlogQ = self.log_likelihood(samples_onehot, hidden)
    #
    #     return hidden, pred, samples, samples_onehot, gumbel_u, gumbel_v, hardlogQ
    #
    # def forward(self, inp):
    #     pred_out = []
    #     samples_out = []
    #     samples_onehot_out = []
    #     gumbel_u_out = []
    #     gumbel_v_out = []
    #     hardlogQ_out = []
    #
    #     for i in range(NUM_LAYER):
    #         hidden, pred, samples, samples_onehot, gumbel_u, gumbel_v, hardlogQ = self.step(inp)
    #         inp = samples
    #
    #         pred_out.append(pred)
    #         samples_out.append(samples)
    #         samples_onehot_out.append(samples_onehot)
    #         gumbel_u_out.append(gumbel_u)
    #         gumbel_v_out.append(gumbel_v)
    #         hardlogQ_out.append(hardlogQ)
    #
    #     # cat
    #     pred_out = torch.stack(pred_out, dim=1)
    #     samples_out = torch.stack(samples_out, dim=1)
    #     samples_onehot_out = torch.stack(samples_onehot_out, dim=1)
    #     gumbel_u_out = torch.stack(gumbel_u_out, dim=1)
    #     gumbel_v_out = torch.stack(gumbel_v_out, dim=1)
    #     hardlogQ_out = torch.stack(hardlogQ_out, dim=1)
    #
    #     return hidden, pred_out, samples_out, samples_onehot_out, gumbel_u_out, gumbel_v_out, hardlogQ_out

    # >>> LSTM, multilayer
    def step(self, inp, lstm_hid):
        emb = self.embed(inp).unsqueeze(1)
        out, lstm_hid = self.lstm(emb, lstm_hid)
        hidden = self.lstm2hid(out.squeeze(1))

        pred = F.softmax(hidden, dim=-1)
        gumbel_u = F.softmax(self.add_gumbel_u(hidden), dim=-1)
        # samples = torch.argmax(gumbel_u, dim=-1).detach()  # use gumbel_u !!!
        samples = torch.argmax(pred, dim=-1).detach()
        samples_onehot = F.one_hot(samples, VOCAB_SIZE).float()
        gumbel_v = F.softmax(self.add_gumbel_v(hidden, samples), dim=-1)
        hardlogQ = self.log_likelihood(samples_onehot, hidden)

        return hidden, lstm_hid, pred, samples, samples_onehot, gumbel_u, gumbel_v, hardlogQ

    def forward(self, inp):
        pred_out = []
        samples_out = []
        samples_onehot_out = []
        gumbel_u_out = []
        gumbel_v_out = []
        hardlogQ_out = []

        lstm_hid = None
        # lstm_hid = self.init_hidden()
        for i in range(NUM_LAYER):
            hidden, lstm_hid, pred, samples, samples_onehot, gumbel_u, gumbel_v, hardlogQ = self.step(inp, lstm_hid)
            inp = samples

            pred_out.append(pred)
            samples_out.append(samples)
            samples_onehot_out.append(samples_onehot)
            gumbel_u_out.append(gumbel_u)
            gumbel_v_out.append(gumbel_v)
            hardlogQ_out.append(hardlogQ)

        # cat
        pred_out = torch.stack(pred_out, dim=1)
        samples_out = torch.stack(samples_out, dim=1)
        samples_onehot_out = torch.stack(samples_onehot_out, dim=1)
        gumbel_u_out = torch.stack(gumbel_u_out, dim=1)
        gumbel_v_out = torch.stack(gumbel_v_out, dim=1)
        hardlogQ_out = torch.stack(hardlogQ_out, dim=1)

        return hidden, pred_out, samples_out, samples_onehot_out, gumbel_u_out, gumbel_v_out, hardlogQ_out

    def extract(self, inp):
        """
        :param inp: batch_size * len * vocab_size
        """

        if len(inp.shape) == 3:
            inp = self.reduction(inp, 'mean')

        # out = self.hid2out(inp)   # pass criterion
        out = F.softmax(self.hid2out(inp), dim=-1)  # no criterion

        return out

    def reduction(self, inp, reduction='mean'):
        if reduction == 'mean':
            out = torch.mean(inp, dim=1)
        elif reduction == 'lstm':
            out, _ = self.last_lstm(inp, None)
            out = out[:, -1]

        return out

    @staticmethod
    def add_gumbel_u(hidden, eps=1e-10):
        gumbel_u = torch.zeros(hidden.size()).cuda().uniform_(0, 1)
        g_t = -torch.log(-torch.log(gumbel_u + eps) + eps)
        # out = F.log_softmax(hidden, dim=-1) + g_t
        out = hidden + g_t
        return out

    @staticmethod
    def add_gumbel_v(hidden, samples, eps=1e-10):
        gumbel_v = torch.zeros(hidden.size()).cuda().uniform_(0, 1)
        p = torch.exp(hidden)
        b = gumbel_v[torch.arange(gumbel_v.size(0)), samples].unsqueeze(1)
        v_1 = -torch.log(-(torch.log(gumbel_v + eps) / (p + eps)) - torch.log(b))
        v_2 = -torch.log(-torch.log(gumbel_v[torch.arange(gumbel_v.size(0)), samples] + eps) + eps)

        v_1_clo = v_1.clone()
        v_1_clo[torch.arange(gumbel_v.size(0)), samples] = v_2.clone()
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

    @staticmethod
    def _u_to_v_poly(hidden, u, samples, eps=1e-8):
        """Convert u to tied randomness in v."""
        p = torch.exp(hidden)

        v_k = torch.pow(u, 1 / torch.clamp(p, min=eps))
        v_k = v_k.detach()
        v_k = torch.pow(v_k, p)

        v_true_k = v_k[torch.arange(v_k.size(0)), samples]

        v_i = u / torch.clamp(torch.pow(v_true_k.unsqueeze(1), p), min=eps)
        v_i = v_i.detach()
        v_i = v_i * torch.pow(v_true_k.unsqueeze(1), p)

        v_clo = v_i.clone()
        v_clo[torch.arange(v_clo.size(0)), samples] = v_true_k.clone()

        v = v_clo + (-v_clo + u).detach()
        return v

    def init_hidden(self, batch_size=BATCH_SIZE):
        """init RMC memory"""
        memory = self.lstm.initial_state(batch_size)
        memory = self.lstm.repackage_hidden(memory)  # detch memory at first
        return memory.cuda()


# REBAR
def cal_loss(label, pred_out, hard_out, soft_out, hardlogQ):
    criterion = nn.CrossEntropyLoss(reduction='none')

    # pass loss
    # pred_loss = -criterion(pred_out, label)
    # hard_loss = -criterion(hard_out, label)
    # soft_loss = -criterion(soft_out, label)

    # no criterion
    pred_loss = pred_out[torch.arange(pred_out.size(0)), label]
    hard_loss = hard_out[torch.arange(hard_out.size(0)), label]
    soft_loss = soft_out[torch.arange(soft_out.size(0)), label]

    de_hard_loss = hard_loss.detach()
    de_soft_loss = soft_loss.detach()

    f = - torch.mean(de_hard_loss * hardlogQ)
    h = ETA * torch.mean(de_soft_loss * hardlogQ - pred_loss + soft_loss)

    return f, h, hard_loss


# vanilla
def cal_loss_v2(label, out):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, label)

    return loss


# NVIL
def cal_loss_v3(label, pred_out, hard_out, soft_out, hardlogQ, nvil):
    criterion = nn.CrossEntropyLoss(reduction='none')

    pred_loss = -criterion(pred_out, label)
    hard_loss = -criterion(hard_out, label)
    soft_loss = -criterion(soft_out, label)

    de_hard_loss = hard_loss.detach()
    de_soft_loss = soft_loss.detach()

    f_target = (de_hard_loss - nvil) * hardlogQ
    # f_target = (de_hard_loss - nvil) * hardlogQ + hard_loss
    f = - torch.mean(f_target)

    h = torch.mean(de_soft_loss * hardlogQ - pred_loss + soft_loss)

    return f, h, -hard_loss


def optimize_vanilla(loss, opt):
    opt.zero_grad()
    loss.backward()
    opt.step()


# REBAR
def optimize_f_plus_h(model, f, h, opt):
    opt.zero_grad()
    f_plus_h = []
    f_grad = torch.autograd.grad(f, list(model.get_params()), create_graph=True)
    h_grad = torch.autograd.grad(h, list(model.get_params()))

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

    # (f+h).backward()
    opt.step()


# NVIL
def optimize_nvil(model, nvil_model, f, h, opt, nvil_opt):
    opt.zero_grad()
    f_plus_h = []
    f_grad = torch.autograd.grad(f, list(model.get_params()) + list(nvil_model.parameters()), create_graph=True,
                                 allow_unused=True)
    h_grad = torch.autograd.grad(h, list(model.get_params()) + list(nvil_model.parameters()), retain_graph=True,
                                 allow_unused=True)
    h_grad = nvil_model.multiply_by_eta(h_grad)

    for a, b in zip(f_grad, h_grad):
        if a is None:
            f_plus_h.append(b)
        elif b is None:
            f_plus_h.append(a)
        else:
            f_plus_h.append(a + b)

    for i, param in enumerate(list(model.get_params()) + list(nvil_model.parameters())):
        param.grad = f_plus_h[i]
    opt.step()

    # TODO: NVIL loss
    # nvil_opt.zero_grad()
    # extra_grad = [rebar_grad.view(-1).pow(2) for rebar_grad in f_plus_h]
    # extra_grad = torch.mean(torch.cat(extra_grad, dim=0))
    extra_grad = vectorize(f_plus_h, skip_none=True)
    extra_grad = torch.mean(extra_grad.pow(2))

    target_grad = torch.autograd.grad(extra_grad, list(nvil_model.parameters()))
    for i, param in enumerate(list(nvil_model.parameters())):
        if param.grad is not None:
            param.grad = param.grad.detach()
            param.grad = param.grad.zero_()
        param.grad = target_grad[i]
    nvil_opt.step()


def vectorize(grads, set_none_to_zero=False, skip_none=False):
    if set_none_to_zero:
        return torch.cat([g.view(-1) if g is not None else
                          torch.zeros(g.size()).view(-1) for g in grads], 0)
    elif skip_none:
        return torch.cat([g.view(-1) for g in grads if g is not None], 0)
    else:
        return torch.cat([g.view(-1) for g in grads], 0)


def train():
    model = Toy()
    nvil_model = NVIL()
    model = model.cuda()
    nvil_model = nvil_model.cuda()
    # opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    opt = torch.optim.Adam(model.get_params(), lr=1e-2)
    nvil_opt = torch.optim.Adam(nvil_model.parameters(), lr=1e-2)

    data, target = prepare_data(VOCAB_SIZE)

    for step in range(EPOCH):
        total_num = 0
        total_acc = 0
        total_loss = []
        for i in range(BATCH_SIZE, VOCAB_SIZE, BATCH_SIZE):
            inp, label = data[i - BATCH_SIZE:i].cuda(), target[i - BATCH_SIZE:i].cuda()
            hidden, _, _, samples_onehot, gumbel_u, gumbel_v, hardlogQ = model(inp)

            # hidden_out = model.hid2out(hidden)
            # pred_out = model.extract(gumbel_u)
            # hard_out = model.extract(samples_onehot)
            # soft_out = model.extract(gumbel_v)
            pred_out = model.extract(gumbel_u[:, -1])
            hard_out = model.extract(samples_onehot[:, -1])
            soft_out = model.extract(gumbel_v[:, -1])

            # vanillia
            # pred_loss = cal_loss_v2(label, pred_out)
            # optimize_vanilla(pred_loss, opt)

            # REBAR
            hardlogQ = torch.sum(hardlogQ, dim=1)  # multilayer
            f, h, hard_loss = cal_loss(label, pred_out, hard_out, soft_out, hardlogQ)
            optimize_f_plus_h(model, f, h, opt)

            # NVIL
            # hardlogQ = torch.sum(hardlogQ, dim=1)
            # nvil = nvil_model(inp)
            # f, h, hard_loss = cal_loss_v3(label, pred_out, hard_out, soft_out, hardlogQ, nvil)
            # optimize_nvil(model, nvil_model, f, h, opt, nvil_opt)

            total_num += BATCH_SIZE

            # REBAR and NVIL
            total_acc += torch.sum((hard_out.argmax(dim=-1) == label)).item()
            total_loss.append(torch.sum(hard_loss).item())
            # total_loss.append((f + h).item())

            # vanilla
            # total_acc += torch.sum((pred_out.argmax(dim=-1) == label)).item()
            # total_loss.append(pred_loss.item())

        print('epoch %d: loss: %.4f, acc: %.4f' % (step, np.mean(total_loss), total_acc / total_num))
        # print(nvil_model.change(nvil_model.eta).tolist())


if __name__ == '__main__':
    train()
