from torch import nn
from pytorch_lightning import LightningModule
from src.models.Encodec import Encodec
from src.models.Transformer.Encoder import TransformerEncoder


class MuMRVQ (LightningModule):

    def __init__(self, encodec: LightningModule = Encodec(), transformer_encoder: LightningModule = TransformerEncoder(), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encodec = encodec
        self.transformer_encoder = transformer_encoder
    
    def forward(self,x):
        return self.transformer_encoder(self.encodec(x))