import os
import random
from itertools import chain

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import trange, tqdm


import config as cfg
from instructor.real_data.instructor import BasicInstructor
from utils.gan_loss import GANLoss
from utils.text_process import text_file_iterator
from utils.data_loader import DataSupplier, GenDataIter, GANDataset
from utils.cat_data_loader import CatClasDataIter
from utils.nn_helpers import create_noise, number_of_parameters
from utils.create_embeddings import EmbeddingsTrainer, load_embedding
from models.generators.FixemGAN_G import Generator
from models.discriminators.FixemGAN_D import Discriminator


class FixemGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(FixemGANInstructor, self).__init__(opt)
        # check if embeddings already exist
        if not os.path.exists(cfg.pretrain_embedding_path):
            # train embedding on available datasets
            self.build_embedding()

        w2v = load_embedding(cfg.pretrain_embedding_path)

        if cfg.run_model == 'fixemgan':
            labels, train_data = zip(*[(0, line) for line in text_file_iterator(cfg.train_data)])

        if cfg.run_model == 'cat_fixemgan':
            labels, train_data = zip(
                *chain(
                    *[[(i, line) for line in text_file_iterator(cfg.cat_train_data.format(i))]
                    for i in range(cfg.k_label)]
                )
            )

        self.train_data_supplier = DataSupplier(train_data, labels, w2v, cfg.batch_size, cfg.batches_per_epoch)

        self.dis = Discriminator(cfg.discriminator_complexity)
        self.log.info(f"discriminator total tranable parameters: {number_of_parameters(self.dis.parameters())}")
        self.gen = Generator(cfg.generator_complexity, cfg.noise_size, w2v, cfg.w2v_embedding_size)
        self.log.info(f"generator total tranable parameters: {number_of_parameters(self.gen.parameters())}")

        if cfg.CUDA:
            self.dis = self.dis.cuda()
            self.gen = self.gen.cuda()

        self.G_criterion = GANLoss(cfg.loss_type, which_net=None, which_D=None, CUDA=cfg.CUDA)
        self.D_criterion = GANLoss(cfg.loss_type, which_net=None, which_D=None, target_real_label=0.8, target_fake_label=0.2, CUDA=cfg.CUDA)

    def build_embedding(self):
        self.log.info(f"Didn't find embeddings in {cfg.pretrain_embedding_path}")
        self.log.info("Will train new one, it may take a while...")
        sources = list(Path(cfg.texts_pile).glob('*.txt'))
        EmbeddingsTrainer(sources, cfg.pretrain_embedding_path).make_embeddings()

    def generator_train_one_batch(self):
        self.gen.optimizer.zero_grad()
        noise = create_noise(cfg.batch_size, cfg.noise_size, cfg.k_label)
        if cfg.CUDA:
            noise = tuple(tt.cuda() for tt in noise)
        fakes = self.gen(*noise)

        real_fake_predicts, label_predicts = self.dis(fakes)
        loss = self.G_criterion.G_loss_fixem(real_fake_predicts, label_predicts, noise[1], fakes)

        loss.backward()
        self.gen.optimizer.step()

        generator_acc = float(
            np.array(real_fake_predicts.detach().cpu().numpy() > 0.5, dtype=int).mean()
        )
        return generator_acc

    def discriminator_train_one_batch(self, real_vector, labels):
        # important to have equal batch size for fake and real vectors
        this_batch_size = real_vector.shape[0]

        # create input
        noise = create_noise(cfg.batch_size, cfg.noise_size, cfg.k_label)
        if cfg.CUDA:
            noise = tuple(tt.cuda() for tt in noise)
        fake = self.gen(*noise).detach()
        text_input_vectors = torch.cat((real_vector, fake))

        # optmizer step
        self.dis.optimizer.zero_grad()
        real_fake_predicts, label_predicts = self.dis(text_input_vectors)
        loss = self.D_criterion.D_loss_fixem(real_fake_predicts, label_predicts[:this_batch_size], labels)
        loss.backward()
        self.dis.optimizer.step()

        real_fake_predicts = real_fake_predicts.clone().detach()
        real_fake_predicts = real_fake_predicts.chunk(2) #splitting to realand fake parks

        discriminator_acc = float(
                torch.cat((
                    real_fake_predicts[0] > 0.5,
                    real_fake_predicts[1] < 0.5
                )).mean(dtype=float)
        )
        return discriminator_acc

    def _run(self):
        for i in trange(cfg.max_epochs):
            for labels, text_vector in tqdm(self.train_data_supplier, leave=False):
                if cfg.CUDA:
                    labels, text_vector = labels.cuda(), text_vector.cuda()
                discriminator_acc = self.discriminator_train_one_batch(text_vector, labels)

                generator_acc = 1 - 2 * (discriminator_acc - 0.5)
                # run the generator until generator acc not get high enought
                while self.one_more_batch_for_generator(generator_acc):
                    generator_acc = self.generator_train_one_batch()

            if cfg.run_model == 'fixemgan':
                print('calculating_metrics')
                scores = self.cal_metrics(fmt_str=True)
            if cfg.run_model == 'cat_fixemgan':
                scores = '\n\n'.join([self.cal_metrics_with_label(label_i=label_i, fmt_str=True) for label_i in range(cfg.k_label)])
            self.log.info(f'epoch: {i}')
            self.log.info(f'{scores}')

    def one_more_batch_for_generator(
        self, generator_acc, leave_in_generator_min=0.1, leave_in_generator_max=0.9
    ):
        generator_acc = min(leave_in_generator_max, generator_acc)
        generator_acc = max(leave_in_generator_min, generator_acc)
        if random.random() > generator_acc:
            return True
        return False

    def sample_for_metrics(self):
        gen_tokens = self.gen.sample(cfg.samples_num, 4 * cfg.batch_size)
        gen_tokens = [sample.split() for sample in gen_tokens]
        gen_tokens_s = self.gen.sample(cfg.small_sample_num, 8 * cfg.batch_size)
        gen_tokens_s = [sample.split() for sample in gen_tokens_s]
        return GenDataIter(gen_tokens), gen_tokens, gen_tokens_s

    def sample_for_metrics_with_label(self, label_i):
        gen_tokens = self.gen.sample(cfg.samples_num, 8 * cfg.batch_size, label_i=label_i)
        gen_tokens = [sample.split() for sample in gen_tokens]
        gen_tokens_s = self.gen.sample(cfg.small_sample_num, 8 * cfg.batch_size, label_i=label_i)
        gen_tokens_s = [sample.split() for sample in gen_tokens_s]
        return GenDataIter(gen_tokens), gen_tokens, gen_tokens_s, CatClasDataIter([gen_tokens], label_i)
