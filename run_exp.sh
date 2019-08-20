#!/usr/bin/env bash

for _ in {1..5}; do
  python main.py --run_model "csgan" --if_test 0 --gen_pretrain 0 --if_real_data 1 --dataset mr15 --temperature 100 --loss_type "nsgan" --mu_type "nsgan rsgan" --eval_type "Ra" --adv_epoch 500 --device 7
done

# Oracle data, catgan and evocatgan
#python main.py --run_model "catgan" --if_test 0 --gen_pretrain 0 --temperature 1 --loss_type "nsgan" --mu_type "nsgan rsgan" --eval_type "Ra" --max_seq_len 20 --device 0
#python main.py --run_model "catgan" --if_test 0 --gen_pretrain 0 --temperature 1 --loss_type "nsgan" --mu_type "nsgan rsgan" --eval_type "Ra" --max_seq_len 40 --device 1
#python main.py --run_model "catgan" --if_test 0 --gen_pretrain 0 --temperature 2 --loss_type "nsgan" --mu_type "nsgan rsgan" --eval_type "Ra" --max_seq_len 20 --device 2
#python main.py --run_model "catgan" --if_test 0 --gen_pretrain 0 --temperature 2 --loss_type "nsgan" --mu_type "nsgan rsgan" --eval_type "Ra" --max_seq_len 40 --device 3
#python main.py --run_model "evocatgan" --if_test 0 --gen_pretrain 0 --temperature 1 --loss_type "nsgan" --mu_type "nsgan rsgan" --eval_type "Ra" --max_seq_len 20 --device 4
#python main.py --run_model "evocatgan" --if_test 0 --gen_pretrain 0 --temperature 1 --loss_type "nsgan" --mu_type "nsgan rsgan" --eval_type "Ra" --max_seq_len 40 --device 5
#python main.py --run_model "evocatgan" --if_test 0 --gen_pretrain 0 --temperature 2 --loss_type "nsgan" --mu_type "nsgan rsgan" --eval_type "Ra" --max_seq_len 20 --device 6
#python main.py --run_model "evocatgan" --if_test 0 --gen_pretrain 0 --temperature 2 --loss_type "nsgan" --mu_type "nsgan rsgan" --eval_type "Ra" --max_seq_len 40 --device 7
