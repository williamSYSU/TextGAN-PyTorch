# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : SentiGAN-william
# @FileName     : data_illustrator.py
# @Time         : Created at 2019-03-19
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import os
import matplotlib.pyplot as plt

title_dict = {0: 'gen{}_oracle_nll',
              1: 'gen{}_mle_nll',
              2: 'gen{}_pg_loss',
              3: 'dis_train_loss',
              4: 'dis_train_acc',
              5: 'dis_val_acc'}

color_list = ['green', 'red', 'skyblue', 'blue', 'violet', 'sienna', 'blueviolet']


def show_plt(data, step, title, if_save=False):
    x = [i for i in range(step)]
    # plt.clf()
    # plt.title(title)
    plt.plot(x, data, color=color_list[color_id], label=title)
    # if if_save:
    #     plt.savefig('savefig/' + title + '.png')
    # else:
    #     plt.show()


def get_data(filename):
    with open(filename, 'r') as fin:
        res_dict = eval(fin.read())
        return (res_dict['gen_oracle_nll'],
                res_dict['gen_mle_nll'],
                res_dict['gen_pg_loss'],
                res_dict['dis_train_loss'],
                res_dict['dis_train_acc'],
                res_dict['dis_val_acc'])


def get_incomplete_data(filename):
    with open(filename, 'r') as fin:
        all_lines = fin.read().strip().split('\n')

        gen_oracle_nll = []
        gen_mle_nll = []
        gen_pg_loss = []
        dis_train_loss = []
        dis_train_acc = []
        dis_val_acc = []

        for line in all_lines:
            items = line.split()
            # print(items)
            try:
                if 'oracle_sample_NLL' in items and 'epoch' not in items:
                    gen_oracle_nll.append(float(items[2]))
                    gen_pg_loss.append(float(items[5]))

                if 'd-step' in items:
                    dis_train_loss.append(float(items[8][:-1]))
                    dis_train_acc.append(float(items[11][:-1]))
                    dis_val_acc.append(float(items[14][:-1]))

                if 'average_train_NLL' in items:
                    gen_mle_nll.append(float(items[6][:-1]))
                    gen_oracle_nll.append(float(items[9]))
            except:
                break

    return (gen_oracle_nll, gen_mle_nll, gen_pg_loss,
            dis_train_loss, dis_train_acc, dis_val_acc)


def show_data(all_data, filename, index=0, gen=0, k_label=2, if_save=False):
    if 0 <= index < 3 and k_label == 2:
        gen_list = ([], [])
        for idx, item in enumerate(all_data[index]):
            if idx % 2 == 0:
                gen_list[0].append(item)
            else:
                gen_list[1].append(item)

        # show_plt(gen_list[gen], len(gen_list[gen]), filename + ':' + title_dict[index].format(gen), if_save)
        show_plt(gen_list[gen], len(gen_list[gen]), filename, if_save)
    elif 3 <= index < 6 or 0 <= index < 6 and k_label == 1:
        # show_plt(all_data[index], len(all_data[index]), filename + ': ' + title_dict[index], if_save)
        show_plt(all_data[index], len(all_data[index]), filename, if_save)
    else:
        print('Error index!!!')


if __name__ == '__main__':
    # log_file = 'log/log_0323_2318.txt'
    # all_data = get_data(log_file)
    # all_data = get_incomplete_data(log_file)
    # log_name = os.path.splitext(log_file)[0].split('/')[1]
    # show_data(all_data, log_name, index=0, gen=0, k_label=2)

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
    log_file_list = ['log_0329_2152', 'log_0329_2152_2', 'log_0329_2152_3', 'log_0329_2152_4']

    color_id = 0
    index = 0
    gen = 0
    plt.clf()
    plt.title(title_dict[index].format(gen))
    all_data_list = []
    for item in log_file_list:
        log_file = log_file_root + item + '.txt'
        # all_data = get_data(log_file)
        all_data = get_incomplete_data(log_file)

        # for idx in range(0, 6):
        #     for gen in range(0, 2):
        #         show_data(all_data, item, index=idx, gen=gen, k_label=2, if_save=True)
        show_data(all_data, item, index=index, gen=gen, k_label=2, if_save=False)
        color_id += 1

    plt.legend()
    plt.show()

# with open(log_file, 'r') as fin:
#     lines = fin.read().strip().split('\n')
#     gen_oracle_nll = []
#     train_loss = []
#     train_acc = []
#     val_acc = []
#     step = 0
#     for line in lines:
#         if line[:6] == 'd-step':
#             items = line.split()
#             train_loss.append(float(items[8][:-1]))
#             train_acc.append(float(items[11][:-1]))
#             val_acc.append(float(items[14][:-1]))
#
#             step += 1
#
#             # writer.add_scalar('train_loss', float(items[8][:-1]), step)
#             # writer.add_scalar('train_acc', float(items[11][:-1]), step)
#             # writer.add_scalar('val_acc', float(items[14]), step)
#             # step += 1
#
#     # print(train_loss, train_acc, val_acc)
#     x = [i for i in range(step)]
#     plt.title(log_file + '(Penalty)')
#     plt.xlabel('step')
#
#     plt.plot(x, train_loss[:], 'r--', label='train_loss')
#     plt.plot(x, train_acc[:], 'g--', label='train_acc')
#     plt.plot(x, val_acc[:], 'b--', label='val_acc')
#     plt.legend(bbox_to_anchor=[0.8,1])
#     # plt.savefig('log/train_loss.png')
#     plt.show()
