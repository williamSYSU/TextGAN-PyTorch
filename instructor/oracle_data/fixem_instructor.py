import os
import random
from itertools import chain

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchtext
from tqdm import tqdm, trange


import config as cfg
from instructor.oracle_data.instructor import BasicInstructor
from instructor.real_data.fixem_instructor import FixemGANInstructor
from utils.gan_loss import GANLoss
from utils.text_process import text_file_iterator
from utils.data_loader import DataSupplier, GANDataset
from utils.nn_helpers import create_noise, number_of_parameters
from utils.helpers import create_oracle
from utils.create_embeddings import EmbeddingsTrainer, load_embedding
from models.FixemGAN_G import Generator
from models.FixemGAN_D import Discriminator


# TO DO:
# 1. test oracle
# 1. train embedding if not exists (if oracle, then always retrain ))
# 3. train epochs and each 10 epochs print metrics
# 4. save? or save each 10 epochs
# 5. fix bleu score
# 6. add new interested scores (IOC, NLL on GPT) (split quick metric and slow metric)
# 7. logger
# 11. make run_fixem clean
# 10. cat_oracle ?

# afterwards:
# check target real/fake to be right (Uniform or const)
# random data portion generator?


class FixemGANInstructor(BasicInstructor, FixemGANInstructor):
    def __init__(self, opt):
        super(FixemGANInstructor, self).__init__(opt)
        # check if embeddings already exist for current oracle

        if cfg.oracle_pretrain:
            if not os.path.exists(cfg.oracle_state_dict_path):
                create_oracle()
            self.oracle.load_state_dict(
                torch.load(cfg.oracle_state_dict_path, map_location='cuda:{}'.format(cfg.device)))

        if cfg.CUDA:
            self.oracle = self.oracle.cuda()

        if not os.path.exists(cfg.pretrain_embedding_path):
            # train embedding on available dataset or oracle
            print(f"Didn't find embeddings in {cfg.pretrain_embedding_path}")
            print("Will train new one, it may take a while...")
            with open(cfg.oracle_samples_path.format(cfg.w2v_samples_num), 'w') as f:
                for sample in tqdm(giant_samples):
                    f.write(" ".join(str(int(idx)) for idx in sample))
                    f.write("\n")

            sources = [cfg.oracle_samples_path.format(cfg.w2v_samples_num)]
            EmbeddingsTrainer(sources, cfg.pretrain_embedding_path).make_embeddings()

        # Metrics
        self.all_metrics = [self.nll_oracle]
