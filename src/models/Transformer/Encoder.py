from torch import nn
import torch
import math
from pytorch_lightning import LightningModule



class Embed(LightningModule):
    
    def __init__(self, embedding_behaviour, embedding_sizes, n_codebooks, card, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding_behaviour = embedding_behaviour
        self.embedding_sizes = embedding_sizes
        self.n_codebooks = n_codebooks
        self.card = card

        self.emb = nn.ModuleList([nn.Embedding(self.card + 1, self.embedding_sizes[codebook]) for codebook in range(self.n_codebooks)])

    def forward(self,indices):
        B, K, T = indices.shape

        embeddings = [self.emb[k](indices[:, k]) for k in range(K)] ## shape B,T,E
        print(embeddings[0].shape)
        if self.embedding_behaviour == 'sum':
            
            input_ = sum(embeddings)
        else:
            input_ = torch.cat(embeddings, dim=-1)

        return input_


class PositionalEncoding(LightningModule):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class LearnedPositionalEncoding(LightningModule):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Define a learnable parameter for positional encodings
        self.positional_encodings = nn.Parameter(torch.randn(max_len, 1, d_model))

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

class TransformerEncoder(LightningModule):

    def __init__(
            self,
            n_codebooks = 4,
            embedding_size = [512,256,128,64],
            card = 1024,
            embedding_behaviour = 'concat',
            position_encoder = PositionalEncoding,
            encoder_kind = "linear",
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.n_codebooks = n_codebooks
        self.embedding_size = embedding_size
        self.card = card
        self.embedding_behaviour = embedding_behaviour
        if self.embedding_behaviour == "concat":
            self.d_model = sum(self.embedding_size)
        else:
            self.d_model = self.embedding_size[0]
        
        self.position_encoder = position_encoder(self.d_model)

        self.emb = Embed(embedding_behaviour=self.embedding_behaviour,embedding_sizes=self.embedding_size, card=self.card, n_codebooks=self.n_codebooks)
        self.linears = nn.ModuleList([nn.Linear(self.embedding_size[codebook], self.card) for codebook in range(self.n_codebooks)])
        
        self.encoder_kind = encoder_kind
        
        if self.encoder_kind == "linear":
            self.transformer = None
        else:
            self.transformer = None

    def forward(self,indices):
        # x is of shape B,n_q,T
        B, K, T = indices.shape

        ## masking here?

        ## pattern generation here

        
        input_ = self.emb(indices)



        ## norm in here

        ## Positional encoding here

        ## Transformer encoder here

        return input_


    
