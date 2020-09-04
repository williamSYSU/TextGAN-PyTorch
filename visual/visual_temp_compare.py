# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

title_dict = {
    'NLL_oracle': 'NLL_oracle',
    'NLL_gen': 'NLL_gen',
    'NLL_div': 'NLL_div',
    'nll_oracle': 'nll_oracle',
    'nll_div': 'nll_div',
    'temp': 'temp',
}

color_list = ['#e74c3c', '#f1c40f', '#1abc9c', '#9b59b6']


def plt_data(data, title, c_id):
    pre_x = np.arange(0, 150, 10)
    adv_x = np.arange(150, 2000, 40)
    x = np.concatenate((pre_x, adv_x))

    plt.plot(x, data, color=color_list[c_id], label=title)
    # plt.xticks(np.arange(0, 2000, 500))


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
    # log_file_root = '../log/'
    log_file_root = 'savefig/figure_log/'
    log_file_list = ['catgan_temp1_final', 'catgan_temp5_final', 'relgan_temp1_final', 'relgan_temp5_final']
    legend_text = [r'CatGAN ($\tau_{\rm{tar}}$=1)', r'CatGAN ($\tau_{\rm{tar}}$=5)', r'RelGAN ($\tau_{\rm{tar}}$=1)',
                   r'RelGAN ($\tau_{\rm{tar}}$=5)']

    data_name_list = ['NLL_oracle', 'NLL_div']
    if_save = True

    plt.clf()
    plt.figure(figsize=(8, 3.5))
    for cur_id, data_name in enumerate(data_name_list):
        assert data_name in title_dict.keys(), 'Error data name'

        plt.subplot(12 * 10 + cur_id + 1)
        if cur_id == 0:
            # plt.title(r"$\rm{NLL}_{\rm{oracle}}$")
            plt.ylabel(r"$\rm{NLL}_{\rm{oracle}}$", fontsize=12)
            plt.plot([150, 150], [8.3, 9.4], 'k--')
        else:
            # plt.title(r"$\rm{NLL}_{\rm{div}}$")
            plt.ylabel(r"$\rm{NLL}_{\rm{div}}$", fontsize=12)
            plt.plot([150, 150], [3.3, 5], 'k--')
        plt.xlabel("training iterations", fontsize=12)

        color_id = 0
        all_data_list = []
        for idx, item in enumerate(log_file_list):
            log_file = log_file_root + item + '.txt'

            # save log file
            all_data = get_log_data(log_file)
            if 'catgan' in log_file or 'relgan' in log_file:
                temp = all_data[title_dict[data_name]]
                last = list(np.array(temp)[range(15, 108, 2)])
                res = temp[:15] + last
                plt_data(res, legend_text[idx], color_id)
            else:
                plt_data(all_data[title_dict[data_name]], legend_text[idx], color_id)
            color_id += 1

    plt.legend()
    plt.tight_layout()
    if if_save:
        plt.savefig('savefig/temp_figure.pdf')
    plt.show()
