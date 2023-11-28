
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
from linformer import Linformer
from src.models.utils import LearnedPositionalEncoding, PositionalEncoding
from pytorch_lightning import LightningModule


class Decoder(nn.Module):
    """"Default transformer decoder. Default behaviour is according to encodecMAE (or similar):


    """

    def __init__(
            self,
            n_codebooks=4,
            card=1024,
            embedding_size=[512, 256, 128, 64],
            sequence_len=2048,
            layers=4,
            n_heads=8,
            embedding_behaviour='concat',
            position_encoder='sinusoidal',
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.n_codebooks = n_codebooks
        self.sequence_len = sequence_len
        self.card = card
        self.embedding_size = embedding_size

        self.embedding_behaviour = embedding_behaviour
        if self.embedding_behaviour == "concat":
            self.d_model = sum(self.embedding_size)
        else:
            self.d_model = self.embedding_size[0]

        self.linears = nn.ModuleList([nn.Linear(
            self.d_model, self.card + 3) for codebook in range(self.n_codebooks)])  # +3 is because of special tokens pad, mask, and pattern

        self.n_heads = n_heads
        self.layers = layers

        self.transformer = None
        self.position_encoder = position_encoder

        if self.position_encoder == 'sinusoidal':
            self.position_encoder = PositionalEncoding(
                self.d_model, max_len=self.sequence_len)
        else:
            self.position_encoder = LearnedPositionalEncoding(
                self.d_model, max_len=self.sequence_len)

    def forward(self, embeddings, padding_mask=None):
        # indices is of shape B,n_q,T
        B, T, d_model = embeddings.shape

        embeddings = self.position_encoder(embeddings)

        output_ = self.transformer(
            embeddings, src_key_padding_mask=padding_mask)
        # shape B,T,d_model

        logits = torch.stack([self.linears[k](output_) for k in range(
            self.n_codebooks)], dim=1).permute(0, 3, 1, 2)

        return logits

    def adapt_sequence_len(self,new_sequence_len):
        self.sequence_len = new_sequence_len
        # if self.position_encoder == "sinusoidal":
        #     self.position_encoder = PositionalEncoding(
        #         self.d_model, max_len=self.sequence_len
        #     )
        # else:
        #     self.position_encoder = LearnedPositionalEncoding(
        #         self.d_model, max_len=self.sequence_len
        #     )

class LinearDecoder(Decoder):
    def __init__(self, n_codebooks=4, card=1024, embedding_size=[512, 256, 128, 64], sequence_len=2048, layers=4, n_heads=8, embedding_behaviour="concat", *args, **kwargs) -> None:
        super().__init__(n_codebooks, card, embedding_size, sequence_len,
                         layers, n_heads, embedding_behaviour, *args, **kwargs)
        self.norm = LayerNorm(self.d_model)
        self.transformer = Linformer(
            self.d_model, self.sequence_len, self.layers)


class VanillaDecoder(Decoder):
    def __init__(self, n_codebooks=4, card=1024, embedding_size=[512, 256, 128, 64], sequence_len=2048, layers=4, n_heads=8, embedding_behaviour="concat", *args, **kwargs) -> None:
        super().__init__(n_codebooks, card, embedding_size, sequence_len,
                         layers, n_heads, embedding_behaviour, *args, **kwargs)
        self.norm = LayerNorm(self.d_model)
        self.transformer_layer = TransformerEncoderLayer(
            self.d_model, self.n_heads, activation="gelu", batch_first=True)
        self.transformer = TransformerEncoder(
            self.transformer_layer, self.layers, norm=self.norm)
