from torch import nn
from pytorch_lightning import LightningModule
from src.models.Encodec import Encodec
from src.models.Transformer.Encoder import Encoder
from audiocraft.modules.codebooks_patterns import DelayedPatternProvider, UnrolledPatternProvider, VALLEPattern
from src.models.utils import EmptyPatternProvider
import torch
import numpy as np


class MuMRVQ (LightningModule):

    def __init__(self,
    
    encoder: LightningModule,
    decoder: LightningModule,
    encodec = None,
    pattern="delay",
    n_codebooks=4,
    sequence_length = 1024,
    pattern_special_token=1024,
    mask_special_token=1025,
    pad_special_token = 1026,
    mask_before=False,debug=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encodec = encodec
        self.transformer_encoder = encoder
        self.transformer_decoder = decoder
        self.sequence_length =  sequence_length

        self.pattern = pattern
        self.pattern_correspondence = {
            "delay": DelayedPatternProvider,
            "unrolled": UnrolledPatternProvider,
            "valle": VALLEPattern,
            "none": None
        }

        self.pattern_special_token = pattern_special_token
        self.mask_special_token = mask_special_token
        self.pad_special_token = pad_special_token
        
        self.pattern_provider = self.pattern_correspondence[pattern]
        if self.pattern_provider:
            self.pattern_provider = self.pattern_provider(
            n_q=n_codebooks)
        self.mask_before = mask_before
        self.debug = debug
          
        self.first_run = True

    def forward(self, x):
        # x is an array of mono 24kHz audio truncated to a certain length. ['original_len'] denotes the  number of samples in the original array (if truncated) ande can be used to construct the padding mask later on
        ## e.g 

        if isinstance(x,dict):                                       ## when loading from a dataloader:
            wav = x['wav']
            lens = x['original_lens']
        else:                                                        ## when running inference on an array for testing purposes
            wav = x
            lens = None

        codes = self.encodec(wav)                                 ## SHAPE : [B, n_q, T]

        if self.pattern_provider:
            codes, indices, pattern_mask = self.get_pattern(
                codes)                                              # pattern application

        padding_mask, codes = self.create_padding_mask(codes,lens)         # SHAPE : [B,T] : boolean padding mask

        
        if self.first_run:
            print(f'codes: {codes.shape}')
            print(f'padding_mask: {padding_mask.shape}')

        padding_mask, codes = self.pad_codes_to_length(codes, padding_mask)

        
        if self.first_run:
            print(f'codes: {codes.shape}')
            print(f'padding_mask: {padding_mask.shape}')


        encoded = self.transformer_encoder(codes, padding_mask, self.mask_before)    ## Input and output SHAPE : [B,T,d_model], batch_first = True
        # keep encoded for Contrastive loss

        decoded_logits = self.transformer_decoder(encoded)      ## SHAPE : [B,n_q,T,card]
        # decoded_logits  = decoded_logits.masked_fill(torch.isnan(decoded_logits), -1e20)

        # keep decoded for contrastive loss

        # cross-entropy between codes and decoded/encoded

        
        # encoded = encoded.masked_fill(torch.isnan(encoded), -1e20)


        self.first_run = False

        return {"logits": decoded_logits,
                "encoded": encoded,
                "codes": codes,
                "padding_mask" : padding_mask}

    def get_pattern(self, codes):
        B, K, T = codes.shape
        pattern = self.pattern_provider.get_pattern(T)
        new_codes, indices, mask = pattern.build_pattern_sequence(
            codes, self.pattern_special_token)
        return new_codes, indices, mask

    def create_padding_mask(self,codes,original_lens):
        padding_mask = torch.zeros((codes.shape[0],codes.shape[2]+1), dtype=bool, device=codes.device) ## +1 because of class token to be added later on
        if original_lens is not None:
            if original_lens.dim() == 1:
                original_lens = original_lens.unsqueeze(0)
            
            encoded_lens = original_lens//self.encodec.model.encoder.hop_length
            for i, l in enumerate(encoded_lens.squeeze()):
                print(l)
                padding_mask[i,l:]=1
                codes[i,:,l:] = self.pad_special_token
        return padding_mask, codes

    def pad_codes_to_length(self,codes, padding_mask):
        target_padding_mask_shape = (padding_mask.shape[0], self.sequence_length) ## Accounting for future class token
        target_codes_shapes = (codes.shape[0], codes.shape[1], self.sequence_length)
        if codes.shape[-1] > self.sequence_length:
            padding_mask = padding_mask[:,:self.sequence_length]
            codes = codes[:,:,:self.sequence_length]
        else:
            new_padding_mask = torch.ones(target_padding_mask_shape).bool().to(codes.device)
            new_codes = torch.full(target_codes_shapes,self.pad_special_token).to(codes.device)
            new_padding_mask[:,:padding_mask.shape[-1]] = padding_mask
            new_codes[:,:,:codes.shape[-1]] = codes
            codes = new_codes
            padding_mask = new_padding_mask

        return padding_mask, codes

    def training_step(self,batch,batch_idx):
        x = batch
        waveform = x['wav']
        lens = x['original_lens']
        out_ = self(batch)
        logits = out_['decoded_logits']
        encoded = out_['encoded']
        codes = out_['codes']
        padding_mask = out_['padding_mask']

        ## loss computations here
        
    def configure_optimizers(self):
        return None
