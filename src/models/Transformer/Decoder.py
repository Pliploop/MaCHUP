from torch import nn
import torch
import math
from pytorch_lightning import LightningModule
from linear_attention_transformer import LinearAttentionTransformerLM
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
from linformer import Linformer




class Decoder(LightningModule):
    """"Default transformer decoder. Default behaviour is according to encodecMAE (or similar):
    
    
    """

    def __init__(
            self,
            n_codebooks=4,
            card = 1024,
            embedding_size=[512, 256, 128, 64],
            sequence_len=2048,
            layers = 4,
            n_heads = 8,
            embedding_behaviour = 'concat',
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
            self.d_model, self.card + 3) for codebook in range(self.n_codebooks)]) ## +3 is because of special tokens pad, mask, and pattern

        self.n_heads = n_heads
        self.layers = layers

        self.transformer = None
       

    def forward(self, embeddings):
        # indices is of shape B,n_q,T
        B, T, d_model = embeddings.shape
        

        output_ = self.transformer(embeddings)
        # shape B,T,d_model

        logits = torch.stack([self.linears[k](output_) for k in range(self.n_codebooks)], dim=1).permute(0, 3, 1, 2)

        return logits



class LinearDecoder(Decoder):
    def __init__(self, n_codebooks=4, card = 1024, embedding_size=[512, 256, 128, 64], sequence_len=2048, layers=4, n_heads=8, embedding_behaviour="concat", *args, **kwargs) -> None:
        super().__init__(n_codebooks,card, embedding_size, sequence_len, layers, n_heads, embedding_behaviour, *args, **kwargs)
        self.norm = LayerNorm(self.d_model)
        self.transformer = Linformer(self.d_model, self.sequence_len, self.layers)
        

class VanillaDecoder(Decoder):
    def __init__(self, n_codebooks=4, card=1024, embedding_size=[512, 256, 128, 64], sequence_len=2048, layers=4, n_heads=8, embedding_behaviour="concat", *args, **kwargs) -> None:
        super().__init__(n_codebooks, card, embedding_size,sequence_len, layers, n_heads, embedding_behaviour, *args, **kwargs)
        self.norm = LayerNorm(self.d_model)
        self.transformer_layer = TransformerEncoderLayer(self.d_model, self.n_heads, activation="gelu")
        self.transformer = TransformerEncoder(self.transformer_layer, self.layers, norm=self.norm)



