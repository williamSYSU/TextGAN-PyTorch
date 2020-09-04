# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os

title_dict = {
    'NLL_oracle': 'NLL_oracle',
    'NLL_gen': 'NLL_gen',
    'NLL_div': 'NLL_div',
    'nll_oracle': 'nll_oracle',
    'nll_div': 'nll_div',
    'temp': 'temp',
}

color_list = ['#2980b9', '#e74c3c', '#1abc9c', '#9b59b6']
ls_list = ['--', '-']
marker_list = [None, None]


def plt_data(data, length, title, c_id, ls, marker, start=0):
    x = np.arange(start, start + length, 1)
    data = data[start:start + length]
    plt.plot(x, data, color=color_list[c_id], label=title, lw=1.0, ls=ls, marker=marker)
    if length < 100:
        plt.xticks(np.arange(start, start + length + 1, 5))


def get_log_data(filename):
    with open(filename, 'r') as fin:
        all_lines = fin.read().strip().split('\n')
        data_dict = {'NLL_oracle': [], 'NLL_gen': [], 'NLL_div': [], 'temp': []}

        for line in all_lines:
            items = line.split()
            try:
                for key in data_dict.keys():
                    if '>>>' not in items and key in items:
                        target = items[items.index(key) + 2]
                        if ',' in target:
                            target = target[:-1]
                        data_dict[key].append(float(target))
            except:
                break

    return data_dict


if __name__ == '__main__':
    os.chdir('..')
    log_file_root = 'savefig/figure_log/'
    log_file_list = ['exp_temp5', 'evo_temp5_nll']
    legend_text = ['Exponential temperature', 'Evolutionary temperature']

    data_name = 'temp'
    if_save = True
    color_id = 0
    all_data_list = []
    length = 20
    start = 1250

    plt.clf()
    if length < 100:
        plt.figure(figsize=(4, 3))
    assert data_name in title_dict.keys(), 'Error data name'
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    for idx, item in enumerate(log_file_list):
        log_file = log_file_root + item + '.txt'

        # save log file
        all_data = get_log_data(log_file)
        plt_data(all_data[title_dict[data_name]], length, legend_text[idx], color_id, start=start, ls=ls_list[idx],
                 marker=marker_list[idx])
        color_id += 1
    if length > 100:
        plt.legend(prop={'size': 7})
        plt.xlabel(r"training iterations", fontsize=7)
        plt.ylabel(r"temperature", fontsize=7)
    plt.tight_layout()
    if if_save:
        plt.savefig('savefig/temp_curve_{}.pdf'.format(length))
    plt.show()
