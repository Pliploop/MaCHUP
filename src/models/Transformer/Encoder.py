from torch import nn
import torch
import math
from pytorch_lightning import LightningModule
from linear_attention_transformer import LinearAttentionTransformerLM
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
from linformer import Linformer



class Embed(LightningModule):

    def __init__(self, embedding_behaviour, embedding_sizes, n_codebooks, card, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding_behaviour = embedding_behaviour
        self.embedding_sizes = embedding_sizes
        self.n_codebooks = n_codebooks
        self.card = card

        self.emb = nn.ModuleList([nn.Embedding(
            self.card + 3, self.embedding_sizes[codebook]) for codebook in range(self.n_codebooks)])

        ## +3 for pad, pattern tokens, and mask tokens

    def forward(self, indices):
        B, K, T = indices.shape

        embeddings = [self.emb[k](indices[:, k])
                      for k in range(K)]  # shape B,T,E
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


class Encoder(LightningModule):
    """"Default transformer encoder. Default behaviour is according to encodecMAE (or similar):
    
        sum embeddings of all tokens and conduct masking afterwards (with or without codebook pattern generator).
    
    Other possible behaviours :
    
        Pattern + Masking before summing embeddings. Meaning the masking mask would include all embeddings. Allows for structured masking patterns like in patchout
        Pattern + Masking before flattening embeddings. Allows for structured patterns in masking and discarding embeddings *BUT* results in 4x longer sequence

    
    """

    def __init__(
            self,
            n_codebooks=4,
            embedding_size=[512, 256, 128, 64],
            card=1024,
            embedding_behaviour='concat',
            position_encoder="sinusoidal",
            sequence_len=2048,
            layers = 6,
            n_heads = 8,
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.n_codebooks = n_codebooks
        self.embedding_size = embedding_size
        self.card = card
        self.sequence_len = sequence_len
        self.embedding_behaviour = embedding_behaviour
        if self.embedding_behaviour == "concat":
            self.d_model = sum(self.embedding_size)
        else:
            self.d_model = self.embedding_size[0]

        if position_encoder == 'sinusoidal':
            self.position_encoder = PositionalEncoding(
                self.d_model, max_len=self.sequence_len)
        else:
            self.position_encoder = LearnedPositionalEncoding(
                self.d_model, mex_len=self.sequence_len)

        self.emb = Embed(embedding_behaviour=self.embedding_behaviour,
                         embedding_sizes=self.embedding_size, card=self.card, n_codebooks=self.n_codebooks)
        self.linears = nn.ModuleList([nn.Linear(
            self.embedding_size[codebook], self.card) for codebook in range(self.n_codebooks)])

        self.n_heads = n_heads
        self.layers = layers

        self.transformer = None
        
        self.norm_in = LayerNorm(self.d_model)

        self.class_token = nn.Parameter(torch.randn(1, 1, self.d_model))


    def forward(self, codes, padding_mask = None, mask_before = False):
        # indices is of shape B,n_q,T
        B, K, T = codes.shape
        
        
        if mask_before:                                        # masking before, i.e all timesteps are sent to be encoded but allows for structured masking algos.
            pass


        input_ = self.emb(codes)
        print(input_.shape)

        
        if not mask_before:
            pass


        # shape B,T,d_model
        class_token = self.class_token.expand(B, 1, self.d_model)  # Expand class token for batch
        input_ = torch.cat([class_token, input_], dim=1)
        input_ = self.position_encoder(input_)
        input_ = self.norm_in(input_)
        print(input_.shape)
        output_ = self.transformer(input_, src_key_padding_mask = padding_mask)
        # shape B,T,d_model

        

        assert output_.shape == (B,T+1,self.d_model), f"{output_.shape}"

        return output_



class LinearEncoder(Encoder):
    """"Does not work yet because of paddding mask implementation"""
    def __init__(self, n_codebooks=4, embedding_size=[512, 256, 128, 64], card=1024, embedding_behaviour='concat', position_encoder="sinusoidal", sequence_len=2048, layers=6, n_heads=8, *args, **kwargs) -> None:
        super().__init__(n_codebooks, embedding_size, card, embedding_behaviour, position_encoder, sequence_len, layers, n_heads, *args, **kwargs)
        self.norm = LayerNorm(self.d_model)
        self.transformer = Linformer(self.d_model, self.sequence_len, self.layers)
        

class VanillaEncoder(Encoder):
    def __init__(self, n_codebooks=4, embedding_size=[512, 256, 128, 64], card=1024, embedding_behaviour='concat', position_encoder="sinusoidal", sequence_len=2048, layers=6, n_heads=8, *args, **kwargs) -> None:
        super().__init__(n_codebooks, embedding_size, card, embedding_behaviour, position_encoder, sequence_len, layers, n_heads, *args, **kwargs)
        self.norm = LayerNorm(self.d_model)
        self.transformer_layer = TransformerEncoderLayer(self.d_model, self.n_heads, activation="gelu", batch_first=True)
        self.transformer = TransformerEncoder(self.transformer_layer, self.layers, norm=self.norm)



