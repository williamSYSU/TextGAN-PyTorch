import torch
from torch.autograd import Variable
from math import ceil


def prepare_generator_batch(samples, start_letter=0, gpu=False):
    """
    Takes samples (a batch) and returns

    Inputs: samples, start_letter, cuda
        - samples: batch_size x seq_len (Tensor with a sample in each row)

    Returns: inp, target
        - inp: batch_size x seq_len (same as target, but with start_letter prepended)
        - target: batch_size x seq_len (Variable same as samples)
    """

    batch_size, seq_len = samples.size()

    inp = torch.zeros(batch_size, seq_len)
    target = samples
    inp[:, 0] = start_letter
    inp[:, 1:] = target[:, :seq_len - 1]

    inp = Variable(inp).type(torch.LongTensor)
    target = Variable(target).type(torch.LongTensor)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target


def prepare_discriminator_data(pos_samples, neg_samples, gpu=False):
    """
    Takes positive (target) samples, negative (generator) samples and prepares inp and target data for discriminator.

    Inputs: pos_samples, neg_samples
        - pos_samples: pos_size x seq_len
        - neg_samples: neg_size x seq_len

    Returns: inp, target
        - inp: (pos_size + neg_size) x seq_len
        - target: pos_size + neg_size (boolean 1/0)
    """

    inp = torch.cat((pos_samples, neg_samples), 0).type(torch.LongTensor)
    target = torch.ones(pos_samples.size()[0] + neg_samples.size()[0])
    target[pos_samples.size()[0]:] = 0

    # shuffle
    perm = torch.randperm(target.size()[0])
    target = target[perm]
    inp = inp[perm]

    inp = Variable(inp)
    target = Variable(target)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target


def prepare_senti_discriminator_data(pos_samples_list, neg_samples_list, k_label, gpu=False):
    """
    Prepare multi-class data for discriminator
    :param pos_samples_list: list: k_label x sample_size or k_label x (k_label x sample_size)
    :param neg_samples_list: list: k_label x sample_size
    :param k_label:
    :param gpu:
    :return:
        - inp: (pos_size + neg_size) x seq_len
        - target: pos_size + neg_size, 0: neg, 1,2,...: pos
    """
    pos_samples = torch.cat(pos_samples_list, dim=0)
    neg_samples = torch.cat(neg_samples_list, dim=0)

    # initial
    inp = torch.cat((pos_samples, neg_samples), dim=0).long()
    target = torch.zeros(pos_samples.size()[0] + neg_samples.size()[0]).long()
    s_size = pos_samples_list[0].size()[0]
    for i in range(k_label):
        target[s_size * i:s_size * (i + 1)] = i + 1

    # shuffle
    perm = torch.randperm(target.size()[0])
    inp = inp[perm]
    target = target[perm]

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target


def batchwise_sample(gen, num_samples, batch_size):
    """
    Sample num_samples samples batch_size samples at a time from gen.
    Does not require gpu since gen.sample() takes care of that.
    """

    samples = []
    for i in range(int(ceil(num_samples / float(batch_size)))):
        samples.append(gen.sample(batch_size))

    return torch.cat(samples, 0)[:num_samples]


def batchwise_oracle_nll(gen, oracle, num_samples, batch_size, max_seq_len, start_letter=0, gpu=False):
    s = batchwise_sample(gen, num_samples, batch_size)
    oracle_nll = 0
    for i in range(0, num_samples, batch_size):
        inp, target = prepare_generator_batch(s[i:i + batch_size], start_letter, gpu)
        oracle_loss = oracle.batchNLLLoss(inp, target) / max_seq_len
        oracle_nll += oracle_loss.data.item()

    return oracle_nll / (num_samples / batch_size)
