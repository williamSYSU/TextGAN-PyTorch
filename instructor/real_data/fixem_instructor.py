import os
import random
from itertools import chain

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchtext
from tqdm import trange


import config as cfg
from instructor.real_data.instructor import BasicInstructor
from utils.gan_loss import GANLoss
from utils.text_process import text_file_iterator
from utils.data_loader import DataSupplier, GANDataset
from utils.nn_helpers import create_noise, number_of_parameters
from utils.create_embeddings import EmbeddingsTrainer, load_embedding
from models.FixemGAN_G import Generator
from models.FixemGAN_D import Discriminator


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

        self.disc = Discriminator(cfg.discriminator_complexity)
        print(
            "discriminator total tranable parameters:",
            number_of_parameters(self.dis.parameters())
        )
        self.gen = Generator(cfg.generator_complexity, cfg.noise_size, w2v, cfg.w2v_embedding_size)
        print(
            "generator total tranable parameters:",
            number_of_parameters(self.gen.parameters())
        )

        if cfg.CUDA:
            self.dis = self.dis.cuda()
            self.gen = self.gen.cuda()

        self.G_criterion = GANLoss(cfg.loss_type, which_net=None, which_D=None, CUDA=cfg.CUDA)
        self.D_criterion = GANLoss(cfg.loss_type, which_net=None, which_D=None, target_real_label=0.8, target_fake_label=0.2, CUDA=cfg.CUDA)

        self.all_metrics = [self.bleu, self.self_bleu, self.ioc, self.gpt_nll]

    def build_embedding(self):
        print(f"Didn't find embeddings in {cfg.pretrain_embedding_path}")
        print("Will train new one, it may take a while...")
        sources = list(Path(cfg.texts_pile).glob('*.txt'))
        EmbeddingsTrainer(sources, cfg.pretrain_embedding_path).make_embeddings()

    def generator_train_one_batch(self):
        self.generator.optimizer.zero_grad()
        noise = create_noise(cfg.batch_size, cfg.noise_size, cfg.k_label)
        if cfg.CUDA:
            noise = tuple(tt.cuda() for tt in noise)
        fakes = self.generator(*noise)

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
        fake = self.generator(*noise).detach()
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
            for labels, text_vector in self.train_data_supplier:
                if cfg.CUDA:
                    labels, text_vector = labels.cuda(), text_vector.cuda()
                discriminator_acc = self.discriminator_train_one_batch(text_vector, labels)

                generator_acc = 1 - 2 * (discriminator_acc - 0.5)
                # run the generator until generator acc not get high enought
                while self.one_more_batch_for_generator(generator_acc):
                    generator_acc = self.generator_train_one_batch()


            samples = self.gene.sample(20, 20)
            for sample in samples:
                print(sample)

            # if (i + 1) % 10 == 0:
            if cfg.run_model == 'fixemgan':
                scores = self.cal_metrics(fmt_str=True)
            if cfg.run_model == 'cat_fixemgan':
                scores = ' '.join([self.cal_metrics_with_label(label_i=label_i, fmt_str=True) for label_i in range(cfg.k_label)])
            print('epoch:', i, scores)


    def one_more_batch_for_generator(
        self, generator_acc, leave_in_generator_min=0.1, leave_in_generator_max=0.9
    ):
        generator_acc = min(leave_in_generator_max, generator_acc)
        generator_acc = max(leave_in_generator_min, generator_acc)
        if random.random() > generator_acc:
            return True
        return False


    def cal_metrics_with_label(self, label_i, fmt_str=False):
        assert type(label_i) == int, 'missing label'
        with torch.no_grad():
            # Prepare data for evaluation
            # eval_samples = self.generator.sample(cfg.samples_num, 8 * cfg.batch_size, label_i=label_i)
            # gen_data = GenDataIter(eval_samples)
            # gen_tokens = tensor_to_tokens(eval_samples, self.idx2word_dict)
            gen_tokens = self.gen.sample(cfg.samples_num, 8 * cfg.batch_size, label_i=label_i)
            gen_tokens = [sample.split() for sample in gen_tokens]
            # gen_tokens_s = tensor_to_tokens(self.gen.sample(200, 200, label_i=label_i), self.idx2word_dict)
            gen_tokens_s = self.gen.sample(200, 200, label_i=label_i)
            gen_tokens_s = [sample.split() for sample in gen_tokens_s]
            # clas_data = CatClasDataIter([eval_samples], label_i)

            # Reset metrics
            self.bleu.reset(test_text=gen_tokens, real_text=self.test_data_list[label_i].tokens)
            # self.nll_gen.reset(self.gen, self.train_data_list[label_i].loader, label_i)
            # self.nll_div.reset(self.gen, gen_data.loader, label_i)
            self.self_bleu.reset(test_text=gen_tokens_s, real_text=gen_tokens)
            # self.clas_acc.reset(self.clas, clas_data.loader)
            # self.ppl.reset(gen_tokens)
            self.ioc.reset(test_text=gen_tokens)
            self.gpt_nll.reset(test_text=gen_tokens)

        if fmt_str:
            return ', '.join(['%s = %s' % (metric.get_name(), metric.get_score()) for metric in self.all_metrics])

        return [metric.get_score() for metric in self.all_metrics]

    def sample_for_metrics(self):
        gen_tokens = self.gen.sample(cfg.samples_num, 4 * cfg.batch_size)
        gen_tokens = [sample.split() for sample in gen_tokens]
        gen_tokens_s = self.gen.sample(200, 200)
        gen_tokens_s = [sample.split() for sample in gen_tokens_s]
        return None, gen_tokens, gen_tokens_s
