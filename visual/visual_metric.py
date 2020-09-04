# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : visual_metric.py
# @Time         : Created at 2019-11-26
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import matplotlib.pyplot as plt
import numpy as np

color_list = ['#2980b9', '#e74c3c', '#1abc9c', '#9b59b6']


def plt_x_y_data(x, y, title, c_id):
    plt.plot(x, y, color=color_list[c_id], label=title)


def get_log_data(filename):
    with open(filename, 'r') as fin:
        all_lines = fin.read().strip().split('\n')
        data_dict = {'NLL_oracle': [], 'NLL_gen': [], 'NLL_div': []}

        for line in all_lines:
            items = line.split()
            try:
                for key in data_dict.keys():
                    if key in items:
                        data_dict[key].append(float(items[items.index(key) + 2][:-1]))
            except:
                break

    return data_dict


if __name__ == '__main__':
    log_file_root = 'log/'
    # Custom your log files in lists, no more than len(color_list)
    log_file_list = ['jsdgan_vanilla_oracle', 'catgan_vanilla_oracle']
    legend_text = ['JSDGAN', 'CatGAN']

    color_id = 0
    title = 'Synthetic data'
    if_save = True
    length = 100

    plt.clf()
    plt.title(title)
    all_data_list = []
    for idx, item in enumerate(log_file_list):
        log_file = log_file_root + item + '.txt'

        # save log file
        all_data = get_log_data(log_file)
        idxs = np.argsort(-np.array(all_data['NLL_oracle']))
        plt_x_y_data(np.array(all_data['NLL_oracle'])[idxs][:length], np.array(all_data['NLL_div'])[idxs][:length],
                     legend_text[idx], color_id)
        color_id += 1

    plt.legend()
    # plt.tight_layout()
    plt.xlabel(r'${\rm NLL_{\rm oracle}}$')
    plt.ylabel(r'${\rm NLL_{\rm div}}$')
    if if_save:
        plt.savefig('../savefig/synthetic_oracle_div.png')
    plt.show()
