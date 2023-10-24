from torch import nn
from pytorch_lightning import LightningModule
from src.models.Encodec import Encodec
from src.models.Transformer.Encoder import Encoder
from audiocraft.modules.codebooks_patterns import DelayedPatternProvider,UnrolledPatternProvider,VALLEPattern


class MuMRVQ (LightningModule):

    def __init__(self, encodec: LightningModule, encoder: LightningModule, pattern="delay", n_codebooks = 4, pattern_special_token = 1025, mask_special_token = 1026, mask_before = False, debug = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encodec = encodec
        self.transformer_encoder = encoder

        self.pattern_correspondence = {
            "delay" : DelayedPatternProvider,
            "unrolled" : UnrolledPatternProvider,
            "valle" : VALLEPattern,
            "none"  : None
        }

        self.pattern_special_token = pattern_special_token
        self.mask_special_token = mask_special_token
        self.pattern_provider = self.pattern_correspondence[pattern](n_q = n_codebooks)
        self.mask_before = mask_before
        self.debug = debug
        
    
    def forward(self,x):
        ## x is an array of mono 24kHz audio truncated to a certain length.
        
        codes = self.encodec(x)
        if self.debug : print(f"codes shape : {codes.shape}")

        if self.pattern_provider:
            codes, indices, pattern_mask = self.get_pattern(codes)             # pattern application
        
        if self.debug : print(f"codes after pattern : {codes.shape}")
        ## pattern and masking here probably
        
        if self.mask_before:
            pass

        encoded = self.transformer_encoder(codes)

        if not self.mask_before:
            pass

        ## keep encoded for Contrastive loss

        ## reinject masked indices here, not applicable if masked before encoding

        if not self.mask_before:
            pass ## reinjection of mask tokens

        ## decoder

        ## keep decoded for contrastive loss

        ## classification head

        ## cross-entropy between codes and decoded/encoded

        out_ = encoded

        return out_

    def get_pattern(self,codes):
        B,K,T = codes.shape
        pattern = self.pattern_provider.get_pattern(T)
        new_codes, indices, mask = pattern.build_pattern_sequence(codes,self.pattern_special_token)
        return new_codes, indices, mask
