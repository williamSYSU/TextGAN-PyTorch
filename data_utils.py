# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : SentiGAN-william
# @FileName     : data_utils.py
# @Time         : Created at 2019-03-16
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import generator
import discriminator
import config as cfg

K_LABEL = 2
save_path = './save'


def create_senti_oracle():
    for i in range(1, K_LABEL + 1):
        oracle = generator.Generator(cfg.gen_embed_dim, cfg.gen_hidden_dim,
                                     cfg.vocab_size, cfg.max_seq_len, oracle_init=True)

        oracle_state_dict_path = cfg.oracle_state_dict_path.format(
            i, cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len
        )
        if cfg.gen_pretrain:
            oracle.load_state_dict(torch.load(oracle_state_dict_path))
        else:
            torch.save(oracle.state_dict(), oracle_state_dict_path)

        samples = oracle.sample(cfg.samples_num)
        torch.save(samples, cfg.oracle_samples_path.format(i, cfg.samples_num))

        large_samples = oracle.sample(2 * cfg.samples_num)
        torch.save(large_samples, cfg.oracle_samples_path.format(i, 2 * cfg.samples_num))


if __name__ == '__main__':
    create_senti_oracle()
