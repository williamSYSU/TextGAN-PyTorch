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

# title_dict = {0: 'gen{}_oracle_nll',
#               1: 'gen{}_mle_nll',
#               2: 'gen{}_pg_loss',
#               3: 'dis_train_loss',
#               4: 'dis_train_acc',
#               5: 'dis_val_acc'}
# title_dict = {0: 'gen{}_oracle_nll',
#               1: 'gen{}_mana_loss',
#               2: 'gen{}_work_loss',
#               3: 'dis_train_loss',
#               4: 'dis_train_acc',
#               5: 'dis_val_acc'}
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


# show source code log loader
def get_source_data(filename):
    with open(filename, 'r') as fin:
        all_lines = fin.read().strip().split('\n')

        oracle_nll = []

        for line in all_lines:
            items = line.split('\t')
            try:
                if 'nll:' in items:
                    oracle_nll.append(float(items[3]))
            except:
                pass

    return [oracle_nll]


def show_data(all_data, filename, index=0, if_save=False):
    if 0 <= index < len(title_dict):
        show_plt(all_data[index], len(all_data[index]), filename, if_save)
    else:
        print('Error index!!!')


if __name__ == '__main__':
    log_file_root = 'log/'
    # log_file_list = ['log_0323_1558', 'log_0323_1559', 'log_0323_1949', 'log_0323_2011', 'log_0323_2133']
    # log_file_list = ['log_0323_2318', 'log_0323_2319', 'log_0323_2319_2', 'log_0323_2320']
    # log_file_list = ['log_0323_2325', 'log_0323_2326', 'log_0323_2320']
    # log_file_list = ['log_0324_1201','log_0324_1201_2','log_0324_1201_3','log_0324_1201_4']
    # log_file_list = ['log_0324_1152','log_0324_1153','log_0324_1153_2','log_0324_1153_3']
    # log_file_list = ['log_0324_2009','log_0324_2009_2','log_0324_2009_3','log_0324_2010_2']
    # log_file_list = ['log_0324_2233','log_0324_2233_2','log_0324_2234','log_0324_2236']
    # log_file_list = ['log_0325_0944','log_0325_0944_2','log_0325_0944_3','log_0325_0945']
    # log_file_list = ['log_0325_1439','log_0325_1439_2','log_0325_1439_3','log_0325_1439_4']
    # log_file_list = ['log_0325_2324', 'log_0325_2324_2', 'log_0325_2324_3', 'log_0325_2324_4']
    # log_file_list = ['log_0329_2152', 'log_0329_2152_2', 'log_0329_2152_3', 'log_0329_2152_4']
    # log_file_list = ['log_0331_1201', 'log_0331_1201_2', 'log_0331_1204']
    # log_file_list = ['log_0331_1439', 'log_0331_1439_2', 'log_0331_1440', 'log_0331_1440_2']
    # log_file_list = ['log_0331_1925', 'log_0331_1925_2']
    # log_file_list = ['log_0403_1224_2', 'log_0403_1247', 'log_0403_1433', 'log_0403_1633']
    # log_file_list = ['log_0404_2136']
    # log_file_list = ['log_0409_1011','log_0409_1032_2','log_0409_1109','log_0409_2031']
    # log_file_list = ['log_0409_2332', 'log_0410_1032']
    # log_file_list = ['source/log_0404_1422']
    # log_file_list = ['log_0410_1216','log_0410_2320','log_0417_1529_2']
    # log_file_list = ['log_0417_1648','log_0417_1748','log_0418_1502','log_0418_2320']
    # log_file_list = ['log_0419_0951','log_0419_1047','log_0421_1044']
    # log_file_list = ['log_0421_2152']
    # log_file_list = ['log_0504_2110', 'log_0504_2112', 'log_0504_2150', 'log_0504_2220']
    # log_file_list = ['log_0505_1646', 'log_0505_1711']
    # log_file_list = ['log_0506_1103', 'log_0506_1159']
    # log_file_list = ['log_0506_1726','log_0507_1037']
    # log_file_list = ['log_0508_1209', 'log_0508_1744', 'log_0508_1729']
    # log_file_list = ['log_0508_1744', 'log_0508_2005', 'log_0508_2008']
    # log_file_list = ['log_0508_2157','log_0509_1228']
    # log_file_list = ['log_0509_1509', 'log_0509_1604','log_0513_1156','log_0513_1534']
    # log_file_list = ['log_0513_1746','log_0513_1546']
    # log_file_list = ['log_0516_2336', 'log_0517_0156_2', 'log_0517_0030']

    # log_file_list = ['log_0523_1207', 'log_0523_0901', 'log_0523_0856']
    # plot_title = ['temp = 1', 'temp = 2', 'temp = 5']

    # log_file_list = ['log_0523_0859', 'log_0523_0859_2', 'log_0523_0906']
    # log_file_list = ['log_0527_1208', 'log_0527_1527', 'log_0527_1528', 'log_0527_1544', 'log_0527_1555']
    log_file_list = ['log_0528_0853']

    color_id = 0
    index = 1
    if_save = False
    model = 'relgan'
    # plot_title = ['RMC, temp=100', 'RMC, temp=1000', 'LSTM-512, temp=100']
    plot_title = log_file_list

    plt.clf()
    plt.title(title_dict[index])
    all_data_list = []
    for idx, item in enumerate(log_file_list):
        log_file = log_file_root + item + '.txt'

        # save log file
        shutil.copyfile(log_file, 'save/' + log_file)
        all_data = get_log_data(model, log_file)
        # show_data(all_data, filename=item, index=index, if_save=if_save)
        show_data(all_data, filename=plot_title[idx], index=index, if_save=if_save)
        color_id += 1

    plt.legend()
    plt.show()
