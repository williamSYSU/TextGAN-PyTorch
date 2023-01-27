import torch.nn as nn

from utils.nn_helpers import get_optimizer, MyConvLayer, MyTransformerEncoderLayer, Flatten

from models.discriminator import CNNDiscriminator

class Discriminator(CNNDiscriminator):
    def __init__(self, complexity):
        alpha = 0.2
        drop_rate = 0.0
        include_transformer = False

        self.main = nn.Sequential(
            # 1 layer
            MyConvLayer(cfg.w2v_embedding_size, complexity, alpha=alpha, drop_rate=drop_rate),
            # 2 layer
            MyConvLayer(
                complexity,
                complexity,
                alpha=alpha,
                drop_rate=drop_rate,
            ),
            # 3 layer
            MyConvLayer(complexity, complexity, alpha=alpha, drop_rate=drop_rate),
            # MyLSTMLayer(complexity, complexity//2),
            # 4 layer
            MyConvLayer(complexity, complexity, alpha=alpha, drop_rate=drop_rate),
            # 5 layer

            MyTransformerEncoderLayer(
                d_model=complexity,
                n_layers=3,
            )
            if include_transformer
            else Dummy(),

            # 6 layer
            MyConvLayer(complexity, complexity, alpha=alpha, drop_rate=drop_rate),
            # MyLSTMLayer(complexity, complexity//2),
            # 7 layer
            MyConvLayer(
                complexity,
                complexity,
                stride=2,
                padding=1,
                alpha=alpha,
                drop_rate=drop_rate,
            ),

            MyConvLayer(
                complexity,
                complexity,
                stride=2,
                padding=1,
                alpha=alpha,
                drop_rate=drop_rate,
            ),
            # 8 layer
            Flatten(),
            nn.Linear(complexity * cfg.max_seq_len // 2 // 2, complexity),
            nn.LeakyReLU(alpha),
            nn.Dropout(drop_rate),
        )

        self.real_fake = nn.Sequential(
            nn.Linear(complexity, 1),
        )
        self.labels = nn.Sequential(
            nn.Linear(complexity, cfg.k_label),
        )
        self.optimizer = get_optimizer()
        # maybe it will help!
        # self.init_params()

    @property
    def nb_of_parameters(self):
        return number_of_parameters(self.parameters())

    def forward(self, x):
        x = self.main(x)
        return self.real_fake(x), self.labels(x)
