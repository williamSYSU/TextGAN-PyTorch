from __future__ import print_function
from math import ceil
import sys
import argparse
import time

import torch
import torch.optim as optim
import torch.nn as nn

from models import generator, discriminator
import helpers
import rollout
from helpers import Signal
import config as cfg
from models.Oracle import Oracle
from instructor.leakgan_instructor import LeakGANInstructor
from instructor.seqgan_instructor import SeqGANInstructor


def _print(content):
    print(content, end='')
    sys.stdout.flush()
    if not cfg.if_test:
        log.write(content)


def gen_optimize(opts, losses):
    for i, (opt, loss) in enumerate(zip(opts, losses)):
        opt.zero_grad()
        loss.backward(retain_graph=True if i < len(opts) - 1 else False)
        opt.step()


def gen_evaluate(gen, dis, oracle):
    gen.eval()
    dis.eval()
    with torch.no_grad():
        # sample from gen and compute oracle NLL
        oracle_loss = helpers.batchwise_oracle_nll(gen, dis, oracle, cfg.samples_num, cfg.batch_size,
                                                   cfg.max_seq_len, start_letter=cfg.start_letter, gpu=cfg.CUDA)

        # real_s = oracle.sample(cfg.samples_num)
        # inverse_nll = helpers.batchwise_gen_nll(gen, dis, real_s, cfg.batch_size, cfg.max_seq_len,
        #                                         gpu=cfg.CUDA)
        inverse_nll = helpers.batchwise_oracle_nll(oracle, dis, gen, cfg.samples_num, cfg.batch_size,
                                                   cfg.max_seq_len, start_letter=cfg.start_letter, gpu=cfg.CUDA)

        inverse_nll = 0

    return oracle_loss, inverse_nll


def train_generator_MLE(gen, gen_opt, oracle, dis, real_data_samples, epochs):
    """
    Max Likelihood Pretraining for the gen

    - gen_opt: [mana_opt, work_opt]
    """
    for epoch in range(epochs):
        sig.update()
        if sig.pre_sig:
            _print('epoch %d : ' % (epoch + 1))
            pre_mana_loss = 0
            pre_work_loss = 0

            t0 = time.time()
            for i in range(0, cfg.samples_num, cfg.batch_size):
                # =====Train=====
                gen.train()
                # dis.eval()  # !!!
                dis.train()

                # for opt in gen_opt:
                #     opt.zero_grad()

                inp, target = helpers.prepare_generator_batch(real_data_samples[i:i + cfg.batch_size],
                                                              start_letter=cfg.start_letter, gpu=cfg.CUDA)

                # mana_loss, work_loss = gen.pretrain_loss(target, dis)
                loss = gen.batchNLLLoss(inp, target)

                # update parameters
                # gen_optimize(gen_opt, [mana_loss, work_loss])
                gen_optimize([gen_opt], [loss])

                # pre_mana_loss += mana_loss.data.item()
                # pre_work_loss += work_loss.data.item()
                pre_mana_loss += loss.data.item()

                if (i / cfg.batch_size) % ceil(
                        ceil(cfg.samples_num / float(cfg.batch_size)) / 10.) == 0:  # roughly every 10% of an epoch
                    _print('.')

            # each loss in a batch is loss per sample
            # pre_mana_loss = pre_mana_loss / ceil(cfg.samples_num / float(cfg.batch_size))
            pre_mana_loss = pre_mana_loss / ceil(cfg.samples_num / float(cfg.batch_size) * cfg.max_seq_len)
            pre_work_loss = pre_work_loss / ceil(cfg.samples_num / float(cfg.batch_size))

            # =====Test=====
            oracle_loss, inverse_nll = gen_evaluate(gen, dis, oracle)

            t1 = time.time()

            _print(' pre_mana_loss = %.4f, pre_work_loss = %.4f, oracle_sample_NLL = %.4f, inverse_NLL = %.4f, '
                   'time = %.4f\n' % (
                       pre_mana_loss, pre_work_loss, oracle_loss, inverse_nll, t1 - t0))

        else:
            _print('\n>>> Stop by pre signal, skip to adversarial training...')
            break


def train_generator_PG(gen, gen_opt, oracle, dis, g_step, current_k=0):
    """
    The gen is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """

    rollout_func = rollout.ROLLOUT(gen, cfg.CUDA)
    adv_mana_loss = 0
    adv_work_loss = 0

    for step in range(g_step):
        with torch.no_grad():
            gen_s = gen.sample(cfg.batch_size, dis, train=True)  # !!! train=True, the only place
            inp, target = helpers.prepare_generator_batch(gen_s, start_letter=cfg.start_letter, gpu=cfg.CUDA)

        # =====Train=====
        gen.train()
        # dis.eval()  # !!!
        dis.train()

        # rewards = rollout_func.get_reward_leakgan(target, cfg.rollout_num, dis,
        #                                           current_k).cpu()  # reward with MC search
        rewards = rollout_func.get_reward(target, cfg.rollout_num, dis, current_k)

        # mana_loss, work_loss = gen.adversarial_loss(target, rewards, dis)
        pg_loss = gen.batchPGLoss(inp, target, rewards)

        # update parameters
        # gen_optimize(gen_opt, [mana_loss, work_loss])
        gen_optimize([gen_opt], [pg_loss])

        # adv_mana_loss += mana_loss.data.item()
        # adv_work_loss += work_loss.data.item()
        adv_mana_loss += pg_loss.data.item()

    # =====Test=====
    oracle_loss, inverse_nll = gen_evaluate(gen, dis, oracle)

    _print(' adv_mana_loss = %.4f, adv_work_loss = %.4f, oracle_sample_NLL = %.4f\n' % (
        adv_mana_loss / g_step, adv_work_loss / g_step, oracle_loss))
    _print('inverse_NLL = %.4f\n' % (inverse_nll))


def train_discriminator(dis, dis_opt, gen, oracle, d_steps, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """
    # prepare data for validate
    with torch.no_grad():
        pos_val = oracle.sample(cfg.samples_num)
        neg_val = gen.sample(cfg.samples_num, dis)

        val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val)

    # loss_fn = nn.BCELoss()
    loss_fn = nn.CrossEntropyLoss()
    for d_step in range(d_steps):
        # prepare data for training
        with torch.no_grad():
            oracle_samples = oracle.sample(cfg.samples_num)
            gen_samples = gen.sample(cfg.samples_num, dis)
            dis_inp, dis_target = helpers.prepare_discriminator_data(oracle_samples, gen_samples)

        # training
        for epoch in range(epochs):
            _print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1))
            total_loss = 0
            total_acc = 0
            train_size = 2 * cfg.samples_num

            for i in range(0, train_size, cfg.batch_size):
                # =====Train=====
                dis.train()
                gen.eval()

                inp, target = dis_inp[i:i + cfg.batch_size], dis_target[i:i + cfg.batch_size]

                if cfg.CUDA:
                    inp = inp.cuda()
                    target = target.cuda()

                dis_opt.zero_grad()
                out = dis.batchClassify(inp)

                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.item()
                total_acc += torch.sum((out.argmax(dim=-1) == target)).item()

                if (i / cfg.batch_size) % ceil(ceil(train_size / float(
                        cfg.batch_size)) / 10.) == 0:  # roughly every 10% of an epoch
                    _print('.')

            total_loss /= ceil(train_size / float(cfg.batch_size))
            total_acc /= float(train_size)

            # =====Test=====
            dis.eval()
            with torch.no_grad():
                val_size = 2 * cfg.k_label * cfg.samples_num
                val_acc = 0
                for i in range(0, val_size, 8 * cfg.batch_size):  # 8 * batch_size for faster
                    inp, target = val_inp[i:i + 8 * cfg.batch_size], val_target[i:i + 8 * cfg.batch_size]

                    if cfg.CUDA:
                        inp = inp.cuda()
                        target = target.cuda()

                    val_pred = dis.batchClassify(inp)
                    val_acc += torch.sum((val_pred.argmax(dim=-1) == target)).item() / val_size

            _print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f\n' % (
                total_loss, total_acc, val_acc))


def test_func():
    leakgan = generator.LeakGenerator(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                                      cfg.padding_idx, cfg.goal_size, cfg.goal_out_size, cfg.step_size, cfg.CUDA)

    dis = discriminator.LEAKDiscriminator(cfg.dis_embed_dim, cfg.vocab_size, cfg.dis_filter_sizes,
                                          cfg.dis_num_filters,
                                          cfg.k_label, cfg.padding_idx, gpu=cfg.CUDA)

    # dis = discriminator.GRUDiscriminator(cfg.dis_embed_dim, cfg.vocab_size, cfg.dis_hidden_dim, cfg.goal_out_size,
    #                                      cfg.max_seq_len, cfg.k_label, cfg.padding_idx, cfg.CUDA)

    oracle = generator.LSTMGenerator(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                                     cfg.padding_idx, gpu=cfg.CUDA, oracle_init=True)
    rollout_func = rollout.ROLLOUT(leakgan, cfg.CUDA)

    # leakgan.load_state_dict(torch.load('./pretrain_generator/NUM10000/gen0_MLE_pretrain_EMB32_HID32_VOC5000_SEQ20_leak.pkl'))

    if cfg.CUDA:
        leakgan = leakgan.cuda()
        dis = dis.cuda()
        oracle = oracle.cuda()

    mana_params, work_params = leakgan.split_params()
    mana_opt = optim.Adam(mana_params, lr=cfg.gen_lr)
    work_opt = optim.Adam(work_params, lr=cfg.gen_lr)

    # 生成器生成样本
    print('begin sample')
    # total_time = 0
    # for _ in range(10):
    t0 = time.time()
    with torch.no_grad():
        gen_samples = leakgan.sample(cfg.samples_num, dis)
        # gen_samples = leakgan.sample(1000, dis)
    t1 = time.time()
    # print(gen_samples.shape)
    print('time-leakgan sample: ', t1 - t0)
    # total_time += t1 - t0
    # print('time-leakgan sample: ', total_time/10)

    # 生成真实样本
    t0 = time.time()
    with torch.no_grad():
        real_samples = oracle.sample(cfg.samples_num)
    print('time-oracle sample: ', t1 - t0)

    # 生成数据和生成器算NLL
    t0 = time.time()
    with torch.no_grad():
        self_nll = helpers.batchwise_gen_nll(leakgan, dis, real_samples, cfg.batch_size, cfg.max_seq_len,
                                             gpu=cfg.CUDA)
        print('self_nll: ', self_nll)
    t1 = time.time()
    print('time-gen nll: ', t1 - t0)

    # 计算重复率
    # t0 = time.time()
    # print('sent_distance: ', helpers.sent_distance(gen_samples))
    # t1 = time.time()
    # print('time-sent distence: ', t1 - t0)

    # 计算NLL_oracle
    # b = 1
    # with torch.no_grad():
    #     t0 = time.time()
    #     oracle_loss = helpers.batchwise_oracle_nll(leakgan, dis, oracle, cfg.samples_num, b * cfg.batch_size,
    #                                                cfg.max_seq_len,
    #                                                start_letter=cfg.start_letter, gpu=cfg.CUDA)
    #     t1 = time.time()
    # print('time-get oracle nll: ', t1 - t0)

    # 预训练过程
    # f0 = time.time()
    # for i in range(0, cfg.samples_num, cfg.batch_size):
    #     t0 = time.time()
    #     _, target = helpers.prepare_generator_batch(real_samples[i:i + cfg.batch_size],
    #                                                 start_letter=cfg.start_letter,
    #                                                 gpu=cfg.CUDA)
    #
    #     pre_mana_loss, pre_work_loss = leakgan.pretrain_loss(target, dis, cfg.start_letter)
    #     print('pretrain_generator loss: ', pre_mana_loss.item(), pre_work_loss.item())
    #     t1 = time.time()
    #     # print('time-pretrain_generator leakgan: ', t1 - t0)
    #
    #     t0 = time.time()
    #     mana_opt.zero_grad()
    #     work_opt.zero_grad()
    #     pre_mana_loss.backward(retain_graph=True)
    #     mana_opt.step()
    #
    #     mana_opt.zero_grad()
    #     work_opt.zero_grad()
    #     pre_work_loss.backward()
    #     work_opt.step()
    #     t1 = time.time()
    #     # print('time-backward pretrain_generator: ', t1 - t0)
    #
    # f1 = time.time()
    # print('total time-pretrain_generator leakgan: ', f1 - f0)

    # 对抗训练过程
    # t0 = time.time()
    # rewards = rollout_func.get_reward_leakgan(real_samples[0:64], 4, dis, 0)
    # t1 = time.time()
    # print('time-rollout: ', t1 - t0)
    #
    # t0 = time.time()
    # adv_mana_loss, adv_work_loss = leakgan.adversarial_loss(inp, target, rewards, dis, cfg.start_letter)
    # print('adversarial loss: ', adv_mana_loss.data, adv_work_loss.data)
    # t1 = time.time()
    # print('time-adversarial train: ', t1 - t0)
    #
    # t0 = time.time()
    # mana_opt.zero_grad()
    # pre_mana_loss.backward()
    # mana_opt.step()
    # work_opt.zero_grad()
    # pre_work_loss.backward()
    # work_opt.step()
    # t1 = time.time()
    # print('time-backward adversarial: ', t1 - t0)


def test_func_2():
    pass


# MAIN
if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--mle_epoch', default=cfg.MLE_train_epoch, type=int)
    parser.add_argument('--adv_epoch', default=cfg.ADV_train_epoch, type=int)
    parser.add_argument('--inter_epoch', default=cfg.inter_epoch, type=int)
    parser.add_argument('--k_label', default=cfg.k_label, type=int)
    parser.add_argument('--samples_num', default=cfg.samples_num, type=int)
    parser.add_argument('--batch_size', default=cfg.batch_size, type=int)
    parser.add_argument('--adv_g_step', default=cfg.ADV_g_step, type=int)
    parser.add_argument('--rollout_num', default=cfg.rollout_num, type=int)
    parser.add_argument('--d_step', default=cfg.d_step, type=int)
    parser.add_argument('--d_epoch', default=cfg.d_epoch, type=int)
    parser.add_argument('--adv_d_step', default=cfg.ADV_d_step, type=int)
    parser.add_argument('--adv_d_epoch', default=cfg.ADV_d_epoch, type=int)
    parser.add_argument('--gen_lr', default=cfg.gen_lr, type=float)
    parser.add_argument('--dis_lr', default=cfg.dis_lr, type=float)

    parser.add_argument('--vocab_size', default=cfg.vocab_size, type=int)
    parser.add_argument('--max_seq_len', default=cfg.max_seq_len, type=int)
    parser.add_argument('--start_letter', default=cfg.start_letter, type=int)
    parser.add_argument('--gen_embed_dim', default=cfg.gen_embed_dim, type=int)
    parser.add_argument('--gen_hidden_dim', default=cfg.gen_hidden_dim, type=int)
    parser.add_argument('--dis_embed_dim', default=cfg.dis_embed_dim, type=int)
    parser.add_argument('--dis_hidden_dim', default=cfg.dis_hidden_dim, type=int)

    parser.add_argument('--cuda', default=cfg.CUDA, type=int)
    parser.add_argument('--device', default=cfg.device, type=int)
    parser.add_argument('--if_save', default=cfg.if_save, type=int)
    parser.add_argument('--if_test', default=cfg.if_test, type=int)
    parser.add_argument('--if_reward', default=cfg.if_reward, type=int)
    parser.add_argument('--ora_pretrain', default=cfg.oracle_pretrain, type=int)
    parser.add_argument('--gen_pretrain', default=cfg.gen_pretrain, type=int)
    parser.add_argument('--dis_pretrain', default=cfg.dis_pretrain, type=int)
    parser.add_argument('--seq_update', default=cfg.seq_update, type=int)
    parser.add_argument('--no_log', default=cfg.no_log, type=int)
    parser.add_argument('--log_file', default=cfg.log_filename, type=str)
    parser.add_argument('--tips', default=cfg.tips, type=str)
    opt = parser.parse_args()
    cfg.init_param(opt)

    # ==========begin test==========
    if cfg.if_test:
        print('Begin test...')
        if cfg.test_idx == 1:
            test_func()
        elif cfg.test_idx == 2:
            test_func_2()
        print('End test...')
        exit(0)
    # ==========end test==========

    # Create log file
    # if not cfg.if_test:
    log = open(cfg.log_filename + '.txt', 'w')
    dict_file = open(cfg.log_filename + '_dict.txt', 'w')

    _print(100 * '=' + '\n')
    _print('> training arguments:\n')
    for arg in vars(opt):
        _print('>>> {0}: {1}\n'.format(arg, getattr(opt, arg)))
    _print(100 * '=' + '\n')

    # ==========Load Signal==========
    sig = Signal(cfg.signal_file)

    # ==========Load Oracle==========
    # oracle
    oracle = Oracle(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                    cfg.padding_idx, gpu=cfg.CUDA)
    if cfg.oracle_pretrain:
        oracle.load_state_dict(torch.load(cfg.oracle_state_dict_path))
        # oracle_samples = torch.load(cfg.oracle_samples_path.format(cfg.samples_num))

    # gen
    # gen = generator.LeakGenerator(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
    #                               cfg.padding_idx, cfg.goal_size, cfg.goal_out_size, cfg.step_size,
    #                               cfg.CUDA)
    gen = generator.LSTMGenerator(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                                  cfg.padding_idx, gpu=cfg.CUDA, oracle_init=True)

    dis = discriminator.LEAKDiscriminator(cfg.dis_embed_dim, cfg.vocab_size, cfg.dis_filter_sizes, cfg.dis_num_filters,
                                          cfg.k_label, cfg.padding_idx, gpu=cfg.CUDA)
    # dis = discriminator.GRUDiscriminator(cfg.dis_embed_dim, cfg.vocab_size, cfg.dis_hidden_dim, cfg.goal_out_size,
    #                                         cfg.max_seq_len, cfg.k_label, cfg.padding_idx, cfg.CUDA)

    if cfg.CUDA:
        oracle = oracle.cuda()
        gen = gen.cuda()
        dis = dis.cuda()

    # ==========Interleaved Pre-Training==========
    dis_optimizer = optim.Adam(dis.parameters(), lr=cfg.dis_lr)

    # mana_params, work_params = gen.split_params()
    # mana_opt = optim.Adam(mana_params, lr=cfg.gen_lr)
    # work_opt = optim.Adam(work_params, lr=cfg.gen_lr)
    # gen_opt = [mana_opt, work_opt]
    gen_opt = optim.Adam(gen.parameters(), lr=cfg.gen_lr)

    oracle_samples = oracle.sample(cfg.samples_num)
    for inter_num in range(cfg.inter_epoch):
        _print('\n>>> Interleaved Round %d...\n' % inter_num)
        sig.update()  # update signal
        if sig.pre_sig:
            # ==========DISCRIMINATOR PRE-TRAINING==========
            _print('\nStarting Discriminator Training...\n')
            if cfg.dis_pretrain:
                _print(
                    'Load pretrain_generator discriminator: {}\n'.format(cfg.pretrained_dis_path.format(cfg.k_label)))
                dis.load_state_dict(torch.load(cfg.pretrained_dis_path.format(cfg.k_label)))
            else:
                train_discriminator(dis, dis_optimizer, gen, oracle, cfg.d_step, cfg.d_epoch)

                if cfg.if_save:
                    torch.save(dis.state_dict(), cfg.pretrained_dis_path.format(cfg.k_label))
                    print('Save pretrain_generator discriminator: {}\n'.format(
                        cfg.pretrained_dis_path.format(cfg.k_label)))

            # ==========GENERATOR MLE TRAINING==========
            _print('\nStarting Generator MLE Training...\n')
            if cfg.gen_pretrain:
                _print('Load MLE pretrain_generator gen: {}\n'.format(cfg.pretrained_gen_path))
                gen.load_state_dict(torch.load(cfg.pretrained_gen_path))
            else:
                _print('Generator MLE training...\n')
                train_generator_MLE(gen, gen_opt, oracle, dis, oracle_samples, cfg.MLE_train_epoch)

                if cfg.if_save:
                    torch.save(gen.state_dict(), cfg.pretrained_gen_path)
                    print('Save MLE pretrain_generator gen: {}\n'.format(cfg.pretrained_gen_path))
        else:
            _print('\n>>> Stop by pre_signal! Skip to adversarial training...\n')
            break

    # ==========ADVERSARIAL TRAINING==========
    _print('\nStarting Adversarial Training...\n')

    oracle_loss = helpers.batchwise_oracle_nll(gen, dis, oracle, cfg.samples_num,
                                               cfg.batch_size,
                                               cfg.max_seq_len, start_letter=cfg.start_letter, gpu=cfg.CUDA)
    _print('Generator: Initial Oracle Sample Loss : %.4f\n' % oracle_loss)

    for epoch in range(cfg.ADV_train_epoch):
        _print('\n--------\nEPOCH %d\n--------\n' % (epoch + 1))
        sig.update()
        if sig.adv_sig:
            # TRAIN GENERATOR
            _print('\nAdversarial Training Generator: \n')
            train_generator_PG(gen, gen_opt, oracle, dis, cfg.ADV_g_step)

            # TRAIN DISCRIMINATOR
            _print('\nAdversarial Training Discriminator : \n')
            train_discriminator(dis, dis_optimizer, gen, oracle, cfg.ADV_d_step, cfg.ADV_d_epoch)
        else:
            _print('\n>>> Stop by adv_signal! Finishing adversarial training...\n')
            break

        # if epoch % 10 == 0 and epoch > 0:
        #     with torch.no_grad():
        #         samples = gen_list[0].sample(1000, dis)
        #     _print('\nSentences distances: {}'.format(helpers.sent_distance(samples)))

    # ==========Save and close==========
    # close log file
    try:
        log.close()
    except:
        pass
