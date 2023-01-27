import torch.nn as nn

from utils.nn_helpers import get_optimizer, MyConvLayer, MyTransformerEncoderLayer, Flatten

@dataclass
class DiscriminatorParameters:
    complexity: int = 512
    alpha: float = 0.2
    drop_rate: float = 0.0
    transformer: bool = False
    transformer_layers: int = 6


class Discriminator(nn.Module):
    def __init__(self, parameters: DiscriminatorParameters, verbose=False):
        super(Discriminator, self).__init__()
        complexity = parameters.complexity
        alpha = parameters.alpha
        drop_rate = parameters.drop_rate
        include_transformer = parameters.transformer

        self.main = nn.Sequential(
            # 1 layer
            MyConvLayer(EMBEDDING_SIZE, complexity, alpha=alpha, drop_rate=drop_rate),
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
                n_layers=parameters.transformer_layers,
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
            nn.Linear(complexity * TARGET_LEN // 2 // 2, complexity),
            nn.LeakyReLU(alpha),
            nn.Dropout(drop_rate),
        )

        self.real_fake = nn.Sequential(
            nn.Linear(complexity, 1),
        )
        self.labels = nn.Sequential(
            nn.Linear(complexity, DEPTH),
        )
        self.optimizer = get_optimizer()

    @property
    def nb_of_parameters(self):
        return number_of_parameters(self.parameters())

    def forward(self, x):
        x = self.main(x)
        return self.real_fake(x), self.labels(x)
