from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from pytorch_lightning import LightningModule
from audiocraft.modules.codebooks_patterns import (
    DelayedPatternProvider,
    UnrolledPatternProvider,
    VALLEPattern,
)
import torch
import torch.optim as optim
import wandb
import matplotlib.pyplot as plt
from pytorch_lightning.cli import OptimizerCallable
import torch.nn.functional as F
from src.models.losses import SupConLoss, MySupConLoss
import torch.autograd.profiler as profiler


class MaCHUP(LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encodec: nn.Module,
        optimizer: OptimizerCallable = None,
        pattern="delay",
        n_codebooks=4,
        sequence_len=1024,
        pattern_special_token=1024,
        mask_special_token=1025,
        pad_special_token=1026,
        mask_before=False,
        debug=False,
        masked_loss_ratio=0.9,
        masked_objective=True,
        window_size=50,
        adapt_sequence_len=True,
        contrastive_to_masked_ratio=0.5,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.encodec = encodec
        self.transformer_encoder = encoder
        self.transformer_decoder = decoder
        self.sequence_len = sequence_len
        self.window_size = window_size
        self.adapt_sequence_len = adapt_sequence_len
        self.contrastive_to_masked_ratio = contrastive_to_masked_ratio

        self.pattern = pattern
        self.pattern_correspondence = {
            "delay": DelayedPatternProvider,
            "unrolled": UnrolledPatternProvider,
            "valle": VALLEPattern,
            "none": None,
        }
        self.masked_loss_ratio = masked_loss_ratio

        self.pattern_special_token = pattern_special_token
        self.mask_special_token = mask_special_token
        self.pad_special_token = pad_special_token

        self.pattern_provider = self.pattern_correspondence[pattern]
        if self.pattern_provider:
            self.pattern_provider = self.pattern_provider(n_q=n_codebooks)
        self.mask_before = mask_before
        self.masked_objective = masked_objective
        self.optimizer = optimizer
        self.first_run = True

        self.supconloss = MySupConLoss(temperature=0.5)
        self.extended_bacth_size = 0
        self.simple_batrch_size = 0

        # 1 loss per codebook for unmasked and masked regions. for sequence length L, scale masked vs unmasked by delta/M and (1-delta)/M
        # per codebook should be weighted by gamma_q, arbitrary?

    def forward(self, x):
        # x is an array of mono 24kHz audio truncated to a certain length. ['original_len'] denotes the  number of samples in the original array (if truncated) ande can be used to construct the padding mask later on
        # e.g
        if self.first_run:
            torch.autograd.set_detect_anomaly(True)

        if isinstance(x, dict):  # when loading from a dataloader:
            wav = x["wav"]
            lens = x["original_lens"]
        else:  # when running inference on an array for testing purposes
            wav = x
            lens = None

        B, N, channels, T = wav.shape

        if self.first_run:
            print(f'tensor shape before augmentation batching: {wav.shape}')

        # wav = wav.permute(1, 0, 2,3).contiguous().view(N * B, channels, T)
        wav = wav.contiguous().view(N*B, channels, T)

        if self.first_run:
            print(f'tensor shape after augmentation batching: {wav.shape}')

        codes = self.encodec(wav)  # SHAPE : [B * N, n_q, T]

        if self.adapt_sequence_len and self.first_run:
            self.transformer_encoder.adapt_sequence_len(codes.shape[-1])
            self.transformer_decoder.adapt_sequence_len(codes.shape[-1])
            self.sequence_len = codes.shape[-1]

        # if self.pattern_provider:
        #     codes, indices, pattern_mask = self.get_pattern(
        #         codes
        #     )  # pattern application (not needed because bidirectional)

        padding_mask, codes = self.create_padding_mask(
            codes, lens
        )  # SHAPE : [B,T] : boolean padding mask

        if self.first_run:
            print(f"codes: {codes.shape}")
            print(f"padding_mask: {padding_mask.shape}")

        padding_mask, codes = self.pad_codes_to_length(codes, padding_mask)
        codes_are_padded = codes == self.pad_special_token

        if self.first_run:
            print(f"codes: {codes.shape}")
            print(f"padding_mask: {padding_mask.shape}")

        if self.first_run or codes.shape[-1]*B*N != self.extended_batch_size:
            self.contrastive_matrix = self.get_contrastive_matrix(
                T=codes.shape[-1], B=B, N=N, W=self.window_size, device=codes.device)

        encoded, unmasked_encoded, codes_mask, encoded_padding_mask, contrastive_matrix_dict = self.transformer_encoder(
            codes=codes,
            padding_mask=padding_mask,
            mask_before=self.mask_before,
            contrastive_matrix=self.contrastive_matrix
        )

    # Input and output SHAPE : [B,T,d_model], batch_first = True
    # note : it is within the encoder that the masking happens and the class token is added. i.e for a masking ratio of 0.5 and a sequence length of 1024
    # Masking is applied randomly and masked inputs are discarded : shape [B,512,d_model], same is done for padding_mask [B,512]
    # class token is added : shape [B,513,d_model], padding_mask is catted with a 0 at the start : [B,513]
    # this is encoded with the transformer encoder : [B,513,d_model]
    # class token is temporarily removed from embeddings and padding mask for computing, masked tokens are added back as a shared embedding : [B,1024,d_model], [B,512,d_model]
    # class token is added back in embeddings and padding_mask [B,1025,512], [B,1025,512]

        codes = codes.clone()
        codes[codes_are_padded] = self.pad_special_token

        masked_codes = codes.clone()
        # purely for logging and visual purposes
        masked_codes[codes_mask] = -1000

        # keep encoded for Contrastive loss

        decoded_logits = self.transformer_decoder(
            unmasked_encoded, padding_mask=encoded_padding_mask
        )  # SHAPE : [B,n_q,T,card]

        # keep decoded for contrastive loss
        # cross-entropy between codes and decoded/encoded

        return {
            "logits": decoded_logits,
            "encoded": encoded,
            "codes": codes,
            "masked_codes": masked_codes,
            "codes_mask": codes_mask,
            "padding_mask": padding_mask,
            "contrastive_matrix_dict": contrastive_matrix_dict
        }

    def get_pattern(self, codes):
        B, K, T = codes.shape
        pattern = self.pattern_provider.get_pattern(T)
        new_codes, indices, mask = pattern.build_pattern_sequence(
            codes, self.pattern_special_token
        )
        return new_codes, indices, mask

    def create_padding_mask(self, codes, original_lens):
        # +1 because of class token to be added later on
        padding_mask = torch.zeros(
            (codes.shape[0], codes.shape[2] + 1), dtype=bool, device=codes.device
        )
        if original_lens is not None:
            if original_lens.dim() == 1:
                original_lens = original_lens.unsqueeze(0)

            encoded_lens = original_lens // self.encodec.model.encoder.hop_length
            for i, l in enumerate(encoded_lens.squeeze()):
                padding_mask[i, l:] = 1
                codes[i, :, l:] = self.pad_special_token
        return padding_mask, codes

    def pad_codes_to_length(self, codes, padding_mask):
        # Accounting for future class token
        target_padding_mask_shape = (
            padding_mask.shape[0], self.sequence_len+1)
        target_codes_shapes = (
            codes.shape[0], codes.shape[1], self.sequence_len+1)
        if codes.shape[-1] > self.sequence_len:
            padding_mask = padding_mask[:, : self.sequence_len]
            codes = codes[:, :, : self.sequence_len]
        else:
            new_padding_mask = torch.ones(
                target_padding_mask_shape, device=codes.device).bool()
            new_codes = torch.full(target_codes_shapes,
                                   self.pad_special_token, device=codes.device)
            new_padding_mask[:, : padding_mask.shape[-1]] = padding_mask
            new_codes[:, :, : codes.shape[-1]] = codes
            codes = new_codes
            padding_mask = new_padding_mask

        return padding_mask, codes

    def get_contrastive_matrix(self, B, T, N, W, device=None):

        total_samples = B * T * N

        # Create a matrix indicating whether samples are from the same sequence

        diag_block = torch.ones((T, T), device=device)
        same_sequence = torch.block_diag(*((B * N) * [diag_block]))

        # Create a window mask
        window_width = W // 2 + 1
        window_mask = torch.diag(torch.ones((total_samples), device=device), 0)

        for k in range(1, window_width):
            window_mask += torch.diag(torch.ones((total_samples - k),
                                      device=device), k)
            window_mask += torch.diag(torch.ones((total_samples - k),
                                      device=device), -k)

        # Apply the window mask
        window_mask = window_mask[:total_samples, :total_samples]

        self.extended_batch_size = total_samples
        self.simple_batch_size = B
        # Return the contrastive matrix
        return (same_sequence + window_mask)*same_sequence, (B, N, T, W)

    def get_contrastive_loss(self, encoded, contrastive_matrix_removed):

        Bext, T, d = encoded.shape
        only_class_tokens = encoded[:, 0, :].contiguous().view(-1, d)
        only_class_tokens_matrix = contrastive_matrix_removed[::T, ::T]

        positive_mask = (contrastive_matrix_removed == 2)
        negative_mask = (contrastive_matrix_removed == 0)
        neutral_mask = (contrastive_matrix_removed == 1)

        positive_class = (only_class_tokens_matrix == 2)
        negative_class = (only_class_tokens_matrix == 0)

        encoded = encoded.contiguous().view(-1, d)

        # split_size = encoded.shape[0]//2
        # f1,f2 = torch.split(encoded,[split_size,split_size], dim=0)
        # encoded  = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        # IMPORTANT NOTE : THIS IS HOW THE AUTHOR IMPLEMENTS IT. BUT THIS DOESN'T WORK BECAUSE IT CONSIDERS ALL VIEWS AS SAME INTANCES WHICH WE DON'T DO

        loss = self.supconloss(encoded, positive_mask=positive_mask,
                               negative_mask=negative_mask, neutral_mask=neutral_mask)
        class_loss = self.supconloss(
            only_class_tokens, positive_mask=positive_class, negative_mask=negative_class)

        return loss, class_loss

    def training_step(self, batch, batch_idx):
        torch.autograd.set_detect_anomaly(True)
        x = batch
        waveform = x["wav"]
        lens = x["original_lens"]

        if self.first_run:
            with profiler.profile(with_stack=True, profile_memory=True) as prof:
                out_ = self(batch)
            print(prof.key_averages(group_by_stack_n=5).table(
                sort_by='self_cpu_time_total', row_limit=20))

        else:
            out_ = self(batch)

        logits = out_["logits"]
        encoded = out_["encoded"]
        codes = out_["codes"]
        codes_mask = out_['codes_mask']
        padding_mask = out_["padding_mask"]
        masked_codes = out_['masked_codes']
        contrastive_matrix_dict = out_['contrastive_matrix_dict']

        all_losses, masked_unmasked, per_codebook = self.get_detailed_losses(
            logits[:, :, :, 1:], codes.clone().long(), codes_mask)
        masked_loss = self.masked_loss_ratio * masked_unmasked['masked_loss'] + (
            1-self.masked_loss_ratio) * masked_unmasked['unmasked_loss']
        contrastive_loss, contrastive_class_loss = self.get_contrastive_loss(
            encoded, contrastive_matrix_dict['contrastive_matrix_masked_removed'])

        for k in all_losses.keys():
            self.log(k, all_losses[k], sync_dist=True)

        for k in masked_unmasked.keys():
            self.log(k, masked_unmasked[k], sync_dist=True)

        for k in per_codebook.keys():
            self.log(k, per_codebook[k], sync_dist=True)

        self.log("contrastive loss", contrastive_loss,
                 prog_bar=True, sync_dist=True)
        self.log("contrastive global loss",
                 contrastive_class_loss, sync_dist=True)

        self.log("train_crossentropy_simple", masked_loss,
                 prog_bar=True, sync_dist=True)
        if self.logger is not None and self.global_step % 1000 == 0:

            fig, ax = plt.subplots(2, 1)
            ax[0].imshow(masked_codes[0, :, :20].cpu().numpy(),
                         vmin=-1000, vmax=1000, cmap="plasma")
            ax[1].imshow(codes[0, :, :20].cpu().numpy(),
                         vmin=-1000, vmax=1000, cmap='plasma')

            self.logger.log_image(
                "masked and unmasked tokens", [wandb.Image(fig)])
            plt.close(fig)

            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(
                contrastive_matrix_dict['original_contrastive_matrix'].cpu(), cmap="plasma")
            ax[1].imshow(
                contrastive_matrix_dict['contrastive_matrix_masked_blackout'].cpu(), cmap="plasma")
            ax[2].imshow(
                contrastive_matrix_dict['contrastive_matrix_masked_removed'].cpu(), cmap="plasma")

            self.logger.log_image(
                "contrastive matrix, masked and unmasked", [wandb.Image(fig)])
            plt.close(fig)

        loss = self.contrastive_to_masked_ratio * contrastive_loss + \
            (1-self.contrastive_to_masked_ratio) * masked_loss
        self.first_run = False

        return loss

    def validation_step(self, batch) -> STEP_OUTPUT:
        torch.autograd.set_detect_anomaly(True)
        x = batch
        out_ = self(batch)
        logits = out_["logits"]
        encoded = out_["encoded"]
        codes = out_["codes"]
        codes_mask = out_['codes_mask']
        contrastive_matrix_dict = out_['contrastive_matrix_dict']

        # simple_crossentropy = self.cross_entropy_simple(logits[:,:,:,1:],codes.clone().long())

        # loss computations here
        # per-codebook loss of masked vs non-masked tokens (so 8 parameters total)
        # contrastive loss - TBD

        all_losses, masked_unmasked, per_codebook = self.get_detailed_losses(
            logits[:, :, :, 1:], codes.clone().long(), codes_mask)
        masked_loss = self.masked_loss_ratio * masked_unmasked['masked_loss'] + (
            1-self.masked_loss_ratio) * masked_unmasked['unmasked_loss']
        contrastive_loss, contrastive_class_loss = self.get_contrastive_loss(
            encoded, contrastive_matrix_dict['contrastive_matrix_masked_removed'])

        for k in all_losses.keys():
            self.log(k, all_losses[k], sync_dist=True)

        for k in masked_unmasked.keys():
            self.log(k, masked_unmasked[k], sync_dist=True)

        for k in per_codebook.keys():
            self.log(k, per_codebook[k], sync_dist=True)

        self.log("val_crossentropy_simple", masked_loss,
                 prog_bar=True, sync_dist=True)
        self.log("val_contrastive loss", contrastive_loss,
                 prog_bar=True, sync_dist=True)
        self.log("val_contrastive global loss",
                 contrastive_class_loss, sync_dist=True)

        if self.masked_objective:
            loss = contrastive_loss + masked_loss
            # this is too simple, some balancing will be needed here when multiple objectives will be combined.
        else:
            loss = contrastive_loss

        return loss

    def configure_optimizers(self):
        if self.optimizer is None:
            optimizer = optim.AdamW(
                self.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        else:
            optimizer = self.optimizer(self.parameters())
        return optimizer

    def get_detailed_losses(self, logits, codes, codes_mask):

        B, C, Q, T = logits.shape
        all_losses = {}
        masked_unmasked = {}
        per_codebook = {}
        loss = F.cross_entropy(
            logits, codes, reduction='none', ignore_index=self.pad_special_token)

        masked_loss = loss*codes_mask
        unmasked_loss = loss*(~codes_mask)

        for q in range(Q):
            all_losses[f'masked_loss_q{q}'] = torch.sum(
                masked_loss[:, q, :])/torch.sum(codes_mask)
            all_losses[f'unmasked_loss_q{q}'] = torch.sum(
                unmasked_loss[:, q, :])/torch.sum(~codes_mask)
            per_codebook[f'global_loss_q{q}'] = all_losses[f'masked_loss_q{q}'] + \
                all_losses[f'unmasked_loss_q{q}']

        masked_loss = torch.sum(masked_loss)/torch.sum(codes_mask)
        unmasked_loss = torch.sum(unmasked_loss)/torch.sum(~codes_mask)
        masked_unmasked['masked_loss'] = masked_loss
        masked_unmasked['unmasked_loss'] = unmasked_loss
        return all_losses, masked_unmasked, per_codebook
