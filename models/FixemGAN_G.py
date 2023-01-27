from dataclasses import dataclass

import torch.nn as nn
from utils.nn_helpers import get_optimizer, Concatenate, Reshape, MyConvLayerNorm, MyConvTransposeLayer, PositionalEncoding, MyLSTMLayerNorm


@dataclass
class GeneratorParameters:
    complexity: int = 512
    concatenate_pe: bool = False
    leacky_ReLU_alpha: float = 0.2
    batch_norm: bool = True
    transformer: bool = False
    lstm: bool = True
    transformer_layers: int = 3


class Generator(nn.Module):
    def __init__(self, parameters: GeneratorParameters, embedding_size: int, verbose=False):
        super(Generator, self).__init__()
        complexity = parameters.complexity
        alpha = parameters.leacky_ReLU_alpha
        added_dim_pe = parameters.complexity if parameters.concatenate_pe else 0
        include_batch_norm = parameters.batch_norm
        include_transformer = parameters.transformer
        include_lstm = parameters.lstm

        self.embedding_size = embedding_size
        self.real_fake_criterion = nn.BCELoss()
        self.label_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.main = nn.Sequential(
            # 1 layer
            Concatenate(1),
            nn.Linear(NOISE_SIZE + DEPTH, TARGET_LEN // 2 // 2 * complexity),
            nn.BatchNorm1d(TARGET_LEN // 2 // 2 * complexity),
            nn.LeakyReLU(alpha),
            Reshape(complexity, TARGET_LEN // 2 // 2),
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
                dim_pe=parameters.complexity,
                max_len=TARGET_LEN,
                concatenate_pe=parameters.concatenate_pe,
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
                max_len=TARGET_LEN,
                concatenate_pe=parameters.concatenate_pe,
            ),
            # 6 layer
            MyTransformerEncoderLayer(
                d_model=complexity + added_dim_pe,
                n_layers=parameters.transformer_layers,
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
            ) if include_lstm else Dummy(),,
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
                EMBEDDING_SIZE,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        self.optimizer = get_optimizer()
        self.to(device)
        if verbose:
            print("total parameters:", number_of_parameters(self.parameters()))

    def forward(self, noise, target_labels):
        target_labels = torch.nn.functional.one_hot(target_labels, num_classes=DEPTH)
        x = self.main([noise, target_labels])
        return x
