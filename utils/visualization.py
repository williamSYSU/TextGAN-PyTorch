# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : visualization.py
# @Time         : Created at 2019-03-19
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import matplotlib.pyplot as plt
import shutil

title_dict = {0: 'gen_pre_loss',
              1: 'oracle_NLL',
              2: 'gen_NLL',
              3: 'BLEU-3'}

color_list = ['green', 'red', 'skyblue', 'blue', 'violet', 'sienna', 'blueviolet']


def show_plt(data, step, title, if_save=False):
    x = [i for i in range(step)]
    # plt.clf()
    # plt.title(title)
    plt.plot(x, data, color=color_list[color_id], label=title)
    if if_save:
        plt.savefig('savefig/' + title + '.png')


def get_dict_data(filename):
    with open(filename, 'r') as fin:
        res_dict = eval(fin.read())
        return (res_dict['gen_oracle_nll'],
                res_dict['gen_mle_nll'],
                res_dict['gen_pg_loss'],
                res_dict['dis_train_loss'],
                res_dict['dis_train_acc'],
                res_dict['dis_val_acc'])


def get_log_data_leak(filename):
    with open(filename, 'r') as fin:
        all_lines = fin.read().strip().split('\n')

        gen_oracle_nll = []
        gen_mana_loss = []
        gen_work_loss = []
        dis_train_loss = []
        dis_train_acc = []
        dis_val_acc = []

        for line in all_lines:
            items = line.split()
            # print(items)
            try:
                if 'adv_mana_loss' in items:
                    gen_mana_loss.append(float(items[2][:-1]))
                    gen_work_loss.append(float(items[5][:-1]))
                    gen_oracle_nll.append(float(items[8][:-1]))

                if 'd-step' in items:
                    dis_train_loss.append(float(items[8][:-1]))
                    dis_train_acc.append(float(items[11][:-1]))
                    dis_val_acc.append(float(items[14][:-1]))

                if 'pre_mana_loss' in items:
                    gen_mana_loss.append(float(items[6][:-1]))
                    gen_work_loss.append(float(items[9][:-1]))
                    gen_oracle_nll.append(float(items[12][:-1]))
            except:
                break

    # print(gen_work_loss)
    return (gen_oracle_nll, gen_mana_loss, gen_work_loss,
            dis_train_loss, dis_train_acc, dis_val_acc)


def get_log_data_rel(filename):
    with open(filename, 'r') as fin:
        all_lines = fin.read().strip().split('\n')

        gen_pre_loss = []
        oracle_nll = []
        gen_nll = []
        bleu3 = []

        data_dict = {'pre_loss': gen_pre_loss, 'oracle_NLL': oracle_nll, 'gen_NLL': gen_nll, 'BLEU-3': bleu3}

        for line in all_lines:
            items = line.split()
            # print(items)
            try:
                for key in data_dict.keys():
                    if key in items:
                        data_dict[key].append(float(items[items.index(key) + 2][:-1]))
            except:
                break

    # print(gen_work_loss)
    return gen_pre_loss, oracle_nll, gen_nll, bleu3


def get_log_data(model, filename):
    if model == 'leakgan':
        return get_log_data_leak(filename)
    elif model == 'relgan':
        return get_log_data_rel(filename)


def show_data(all_data, filename, index=0, if_save=False):
    if 0 <= index < len(title_dict):
        show_plt(all_data[index], len(all_data[index]), filename, if_save)
    else:
        print('Error index!!!')


if __name__ == '__main__':
    log_file_root = 'log/'
    log_file_list = ['log_0528_0853']

    color_id = 0
    index = 1
    if_save = False
    model = 'relgan'
    plot_title = log_file_list
    # plot_title = ['RMC, temp=1']

    plt.clf()
    plt.title(title_dict[index])
    all_data_list = []
    for idx, item in enumerate(log_file_list):
        log_file = log_file_root + item + '.txt'

        # save log file
        shutil.copyfile(log_file, 'save/' + log_file)
        all_data = get_log_data(model, log_file)
        show_data(all_data, filename=plot_title[idx], index=index, if_save=if_save)
        color_id += 1

    plt.legend()
    plt.show()
