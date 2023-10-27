from encodec import EncodecModel
from encodec.utils import convert_audio

import torchaudio
import torch

from pytorch_lightning import LightningModule

from torch import nn


class Encodec(LightningModule):

    def __init__(self,  sample_rate = 24000, frozen = True, model_bandwidth = 3,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frozen = frozen
        self.model_bandwidth = model_bandwidth
        if sample_rate == 24000 :
            self.model = EncodecModel.encodec_model_24khz()
        else:
            self.model = EncodecModel.encodec_model_48khz()
        # The number of codebooks used will be determined bythe bandwidth selected.
        # E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
        # Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
        # For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
        # of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
        self.model.set_target_bandwidth(self.model_bandwidth)
        self.hop_length = self.model.encoder.hop_length

    def forward(self,wav):
        ## wav is a batched waveform.unsqueeze if not:
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)
        
        if self.frozen :
            with torch.no_grad():

                encoded_frames = self.model.encode(wav)
        else:
            encoded_frames = self.model.encode(wav)
        
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]

        return codes