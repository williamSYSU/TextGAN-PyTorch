from dataclasses import dataclass

import torch
import torch.nn as nn
from utils.nn_helpers import (
    get_optimizer,
    create_noise,
    Concatenate,
    Reshape,
    MyConvLayerNorm,
    MyConvTransposeLayer,
    PositionalEncoding,
    MyLSTMLayerNorm,
    Dummy,
)

import config as cfg
from models.generators.generator import LSTMGenerator



class Generator(LSTMGenerator):
    def __init__(self, complexity, noise_size, w2v, w2v_embedding_size):
        super(Generator, self).__init__(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.target_len, cfg.padding_idx)
        alpha = 0.2
        added_dim_pe = 0
        include_batch_norm = True
        include_transformer = False
        include_lstm = True
        self.noise_size = noise_size
        self.w2v = w2v
        self.embedding_size = w2v_embedding_size

        self.main = nn.Sequential(
            # 1 layer
            Concatenate(1),
            nn.Linear(cfg.noise_size + cfg.k_label, cfg.target_len // 2 // 2 * complexity),
            nn.BatchNorm1d(cfg.target_len // 2 // 2 * complexity),
            nn.LeakyReLU(alpha),
            Reshape(complexity, cfg.target_len // 2 // 2),
            # 2 layer
            MyConvLayerNorm(complexity, complexity, alpha=alpha),
            # 3 layer
            MyConvTransposeLayer(
                complexity,
                complexity,
                stride=2,
                output_padding=1,
                alpha=alpha,
                include_batch_norm=include_batch_norm,
            ),
            # 4 layer
            MyConvTransposeLayer(
                complexity,
                complexity,
                stride=2,
                output_padding=1,
                alpha=alpha,
                include_batch_norm=include_batch_norm,
            ),
            # adding/concatenating positional encoding
            PositionalEncoding(
                dim_pe=complexity,
                max_len=cfg.target_len,
                concatenate_pe=False,
            ),
            # 5 layer
            MyConvLayerNorm(
                complexity + added_dim_pe,
                complexity,
                alpha=alpha,
                include_batch_norm=include_batch_norm,
            ),
            # adding/concatenating positional encoding
            PositionalEncoding(
                dim_pe=complexity,
                max_len=cfg.target_len,
                concatenate_pe=False,
            ),
            # 6 layer
            MyTransformerEncoderLayer(
                d_model=complexity + added_dim_pe,
                n_layers=3,
            )
            if include_transformer
            else Dummy(),
            # 7 layer
            MyConvTransposeLayer(
                complexity + added_dim_pe,
                complexity,
                alpha=alpha,
                include_batch_norm=include_batch_norm,
            ),
            # 8 layer
            MyLSTMLayerNorm(
                complexity,
                complexity//2,
            ) if include_lstm else Dummy(),

            # 9 layer
            MyConvTransposeLayer(
                complexity,
                complexity,
                alpha=alpha,
                include_batch_norm=include_batch_norm,
            ),
            # 10 layer
            MyLSTMLayerNorm(
                complexity,
                complexity//2,
            ) if include_lstm else Dummy(),
            # 11 layer
            MyConvTransposeLayer(
                complexity,
                complexity,
                alpha=alpha,
                include_batch_norm=include_batch_norm,
            ),
            # 12 layer
            nn.Conv1d(
                complexity,
                cfg.w2v_embedding_size,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        self.optimizer = get_optimizer(self.parameters())

    def forward(self, noise, target_labels):
        target_labels = torch.nn.functional.one_hot(target_labels, num_classes=cfg.k_label)
        x = self.main([noise, target_labels])
        return x

    def sample(self, num_samples, batch_size, label_i = 'random', start_letter=cfg.start_letter):
        noise = create_noise(num_samples, self.noise_size, cfg.k_label)
        if label_i != 'random':
            noise = (noise[0], torch.tensor(label_i).expand_as(noise[1]))

        if cfg.CUDA:
            noise = tuple(tt.cuda() for tt in noise)
        fakes = self.forward(*noise)
        fakes = fakes.detach().cpu().numpy()
        assert len(fakes.shape) == 3
        return [self.recover_sentence(fake) for fake in fakes]

    def recover_sentence(self, fake):
        fake = fake.T
        tokens = []
        for token_vector in fake:
            token = self.w2v.wv.most_similar([token_vector])[0][0]
            if token == cfg.padding_token:
                continue
            tokens.append(token)
        return " ".join(tokens).strip()
