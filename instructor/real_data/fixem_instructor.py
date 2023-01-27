import os

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchtext
from tqdm import trange


import config as cfg
from instructor.real_data.instructor import BasicInstructor
from utils.text_process import text_file_iterator
from utils.data_loader import DataSupplier, GANDataset
from utils.nn_helpers import create_noise, number_of_parameters
from utils.create_embeddings import EmbeddingsTrainer, load_embedding
from models.FixemGAN_G import Generator
from models.FixemGAN_D import Discriminator


# TO DO:
# 1. train embedding if not exists (if oracle, then always retrain )
# 2. create data generator (categorical and non categorical) (based on given dataset)
# 3. create disc and gen
# 4. train epochs and each 10 epochs print metrics
# 5. show metrics
# 6. save? or save each 10 epochs

# chack target real/fake to be right (Uniform or const)


class FixemGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(FixemGANInstructor, self).__init__(opt)
        # check if embeddings already exist for current oracle
        if not os.path.exists(cfg.pretrain_embedding_path):
            # train embedding on available dataset or oracle
            sources = list(Path(texts_pile).glob('*.txt'))
            EmbeddingsTrainer(sources, cfg.pretrain_embedding_path).make_embeddings()

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

        self.train_data_supplier = DataSupplier(train_data, labels, w2v, True, cfg.batch_size, cfg.batches_per_epoch)

        self.discriminator = Discriminator(cfg.discriminator_complexity)
        print(
            "discriminator total tranable parameters:",
            number_of_parameters(self.discriminator.parameters())
        )
        self.generator = Generator(cfg.generator_complexity, cgf.noise_size, w2v)
        print(
            "generator total tranable parameters:",
            number_of_parameters(self.generator.parameters())
        )

        self.G_criterion = GANLoss(cfg.run_model, which_net=None, which_D=None, )
        self.D_criterion = GANLoss(cfg.run_model, which_net=None, which_D=None, target_real_label=0.8, target_fake_label=0.2)

        self.all_metrics = [self.bleu, self.self_bleu]

    def generator_train_one_batch(self):
        self.generator.optimizer.zero_grad()
        noise = create_noise(cfg.batch_size, cfg.noise_size. cfg.k_label)
        fakes = self.generator(*noise)

        real_fake_predicts, label_predicts = self.discriminator(fakes)
        loss = self.G_criterion.G_loss_fixem(real_fake_predicts, label_predicts, fakes)
        loss.backward()
        self.generator.optimizer.step()

        generator_acc = float(
            np.array(real_fake_predicts.detach().cpu().numpy() > 0.5, dtype=int).mean()
        )
        return generator_acc

    def discriminator_train_one_batch(self, real_vector, labels):
        # important to have equal batch size for fake and real vectors
        this_batch_size = real_vector.shape[0]

        # create input
        noise = create_noise(cfg.batch_size, cfg.noise_size. cfg.k_label)
        fake = self.generator(*noise).detach()
        text_input_vectors = torch.cat((real_vector, fake))

        # optmizer step
        discriminator.optimizer.zero_grad()
        real_fake_predicts, label_predicts = self.discriminator(text_input_vectors)
        loss = self.D_criterion.D_loss_fixem(real_fake_predicts, label_predicts[:this_batch_size], labels)
        loss.backward()
        discriminator.optimizer.step()

        discriminator_acc = torch.cat(
            (
                real_fake_predicts.chunk(2)[0] > 0.5,
                real_fake_predicts.chunk(2)[1] < 0.5
            )
        )
        return discriminator_acc


    def _run(self):
        for i in trange(cfg.max_epochs):
            for labels, text_vector in self.train_data_supplier:
                discriminator_acc = self.discriminator_train_one_batch(text_vector, labels)

                generator_acc = 1 - 2 * (discriminator_acc - 0.5)
                # run the generator until generator acc not get high enought
                while self.one_more_batch_for_generator(generator_acc):
                    generator_acc = self.generator_train_one_batch()

            if cfg.run_model == 'fixemgan':
                scores = self.cal_metrics(fmt_str=True)
            if cfg.run_model == 'cat_fixemgan':
                scores = self.cal_metrics_with_label(fmt_str=True)

            print('epoch:', i, scores)


    def one_more_batch_for_generator(
        self, generator_acc, leave_in_generator_min=0.1, leave_in_generator_max=0.9
    ):
        generator_acc = min(leave_in_generator_max, generator_acc)
        generator_acc = max(leave_in_generator_min, generator_acc)
        if random.random() > generator_acc:
            return True
        return False


    def cal_metrics(self, fmt_str=False):
        """
        Calculate metrics
        :param fmt_str: if return format string for logging
        """
        with torch.no_grad():
            # Prepare data for evaluation
            gen_tokens = self.generator.sample(cfg.samples_num, 4 * cfg.batch_size)
            gen_tokens_s = self.generator.sample(200, 200)

            # Reset metrics
            self.bleu.reset(test_text=gen_tokens, real_text=self.test_data.tokens)
            # self.nll_gen.reset(self.gen, self.train_data.loader)
            # self.nll_div.reset(self.gen, gen_data.loader)
            self.self_bleu.reset(test_text=gen_tokens_s, real_text=gen_tokens)
            # self.ppl.reset(gen_tokens)

        if fmt_str:
            return ', '.join(['%s = %s' % (metric.get_name(), metric.get_score()) for metric in self.all_metrics])
        else:
            return [metric.get_score() for metric in self.all_metrics]
