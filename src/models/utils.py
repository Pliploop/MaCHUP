from audiocraft.modules.codebooks_patterns import CodebooksPatternProvider, Pattern, PatternLayout, LayoutCoord
import typing as tp
import torch
from pytorch_lightning import LightningModule
from torch import nn
import random
import math


class EmptyPatternProvider(CodebooksPatternProvider):

    def __init__(self, n_q: int, delays: tp.Optional[tp.List[int]] = None, empty_initial: int = 0):
        super().__init__(n_q)
        if delays is None:
            delays = list(range(n_q))
        self.delays = delays
        self.empty_initial = empty_initial

    def get_pattern(self, timesteps: int) -> Pattern:
        out: PatternLayout = [[]]
        max_delay = max(self.delays)
        if self.empty_initial:
            out += [[] for _ in range(self.empty_initial)]
        for t in range(0, timesteps):
            v = []
            for q, delay in enumerate(self.delays):
                v.append(LayoutCoord(t, q))
            out.append(v)

        return Pattern(out, n_q=self.n_q, timesteps=timesteps)


class PositionalEncoding(LightningModule):

    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 1024, batch_first: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        if self.batch_first:
            x = x + self.pe.permute(1,0,2)[:,:x.size(1),:]
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)
        


class LearnedPositionalEncoding(LightningModule):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Define a learnable parameter for positional encodings
        self.positional_encodings = nn.Parameter(
            torch.randn(max_len, 1, d_model))

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # Get the positional encodings
        pe = self.positional_encodings[:x.size(0)]

        # Add positional encodings to the input
        x = x + pe

        return self.dropout(x)
