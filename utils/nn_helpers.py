import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.distributions.uniform import Uniform

import math

import matplotlib.pyplot as plt


def create_noise(sample_size, noise_size):
    return (
        torch.randn(sample_size, noise_size).to(device),
        torch.randint(0, DEPTH, (sample_size,)).sort().values.to(device),
    )


def multiply_shape(shape):
    if len(shape) == 1:
        return shape[0]
    return shape[0] * multiply_shape(shape[1:])


def number_of_parameters(parameters):
    nb_of_vars = 0
    for parameter in parameters:
        nb_of_vars += multiply_shape(tuple(parameter.shape))
    return nb_of_vars


def get_optimizer(parameters, lr=0.0001, betas=(0.5, 0.999)):
    return optim.Adam(parameters, lr=lr, betas=betas)


class PositionalEncoding(nn.Module):
    def __init__(self, dim_pe: int, max_len: int = TARGET_LEN, concatenate_pe=False):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_pe, 2) * (-math.log(dim_pe // 4) / dim_pe)
        )
        pe = torch.zeros(max_len, 1, dim_pe)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = torch.transpose(pe, 0, 1)
        pe = torch.transpose(pe, 1, 2)
        # plt.imshow(pe[0], cmap="hot", interpolation="nearest")
        # plt.show()
        self.register_buffer("pe", pe)
        self.concatenate_pe = concatenate_pe

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        pe = self.pe.repeat(x.size(0), 1, 1)
        # input (N,C,L) - N bathc size, C - channels, L - length
        return torch.cat((x, pe), 1) if self.concatenate_pe else x + pe


class Reshape(nn.Module):
    def __init__(self, *out_shape):
        super(Reshape, self).__init__()
        self.out_shape = out_shape

    def forward(self, input_batch):
        """Turch batched flat vector to out_shape"""
        return torch.reshape(input_batch, (-1, *self.out_shape))


class Concatenate(nn.Module):
    def __init__(self, dim):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, input_batch):
        return torch.cat(input_batch, dim=1)


class Dummy(nn.Module):
    """For shadowing some layers."""

    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, x):
        return x


class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead=1, n_layers=1):
        super(MyTransformerEncoderLayer, self).__init__()
        self.tranformer_layers = nn.Sequential(
            *tuple(
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, batch_first=True
                )
                for _ in range(n_layers)
            )
        )

    def forward(self, x):
        x = self.tranformer_layers(torch.transpose(x, 1, 2))
        return torch.transpose(x, 1, 2)


class MyConvTransposeLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        output_padding=0,
        alpha=0.2,
        include_batch_norm=True,
        padding=1,
        kernel_size=3,
    ):
        super(MyConvTransposeLayer, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.LeakyReLU(alpha),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm1d(out_channels) if include_batch_norm else Dummy(),
            nn.LeakyReLU(alpha),
        )

    def forward(self, x):
        return self.conv_layer(x)


class MyConvLayerNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        output_padding=0,
        alpha=0.2,
        include_batch_norm=True,
        padding=1,
        kernel_size=3,
    ):
        super(MyConvLayerNorm, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm1d(out_channels) if include_batch_norm else Dummy(),
            nn.LeakyReLU(alpha),
        )

    def forward(self, x):
        return self.conv_layer(x)


class MyLSTMLayerNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        alpha=0.2,
    ):
        super(MyLSTMLayerNorm, self).__init__()
        self.lstm = nn.LSTM(
            batch_first=True,
            bidirectional=True,
            input_size=in_channels,
            hidden_size=out_channels,
        )
        self.layers = nn.Sequential(
            nn.BatchNorm1d(2 * out_channels),
            nn.LeakyReLU(alpha),
        )
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x, (hn, cn) = self.lstm(x)
        x = torch.transpose(x, 1,2)
        x = self.layers(x)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self.start_dim = 1

    def forward(self, input_tensor):
        return torch.flatten(input_tensor, start_dim=1)


class MyConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding="same",
        alpha=0.2,
        drop_rate=0.2,
    ):
        super(MyConvLayer, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.LeakyReLU(alpha),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.conv_layer(x)


class MyLSTMLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        alpha=0.2,
        drop_rate=0.2,
    ):
        super(MyLSTMLayer, self).__init__()
        self.lstm = nn.LSTM(
            batch_first=True,
            bidirectional=True,
            input_size=in_channels,
            hidden_size=out_channels,
        )
        self.layers = nn.Sequential(
            nn.LeakyReLU(alpha),
            nn.Dropout(drop_rate),
        )
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x, (hn, cn) = self.lstm(x)
        x = torch.transpose(x, 1,2)
        x = self.layers(x)
        return x



def DiversityLoss():
    cs2 = torch.nn.CosineSimilarity(dim=2)
    def cos_sim_loss(generated):
        batch_size = generated.shape[0]
        generated = generated.repeat(batch_size, 1, 1, 1)
        generatedTranspose = torch.transpose(generated, 0, 1)
        loss = cs2(generated, generatedTranspose)
        ind = np.diag_indices(loss.shape[0])
        loss[ind[0], ind[1], :] = 0 # set 0 to similarity of message to itself
        loss = loss.mean(axis=2).max(axis=0).values.mean()
        return loss
    return cos_sim_loss
