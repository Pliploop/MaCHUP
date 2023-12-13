
from typing import Any, Dict
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
from src.models.losses import MySupConLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MaCHUP(LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encodec: nn.Module,
        optimizer: OptimizerCallable = None,
        n_codebooks=4,
        sequence_len=1024,
        mask_special_token=1025,
        pad_special_token=1026,
        mask_before=False,
        debug=False,
        masked_loss_ratio=0.9,
        masked_objective=True,
        contrastive_objective=True,
        window_size=50,
        adapt_sequence_len=True,
        contrastive_to_masked_ratio=0.5,
        global_vs_local_contrastive_loss_ratio=0.1,
        global_class_vs_average_contrastive_loss_ratio=1,
        contrastive_temperature=0.5,
        reduce_lr_monitor='masked loss',
        only_global_contrastive=False,
        use_embeddings=False,
        resume_from_checkpoint=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.encodec = encodec
        self.transformer_encoder = encoder
        self.transformer_decoder = decoder
        self.d_model = self.transformer_encoder.d_model
        self.sequence_len = sequence_len
        self.window_size = window_size
        self.adapt_sequence_len = adapt_sequence_len

        # loss flag and triggers
        self.contrastive_to_masked_ratio = contrastive_to_masked_ratio
        self.global_vs_local_contrastive_loss_ratio = global_vs_local_contrastive_loss_ratio
        self.global_class_vs_average_contrastive_loss_ratio = global_class_vs_average_contrastive_loss_ratio
        self.reduce_lr_monitor = reduce_lr_monitor
        self.only_global_contrastive = only_global_contrastive
        self.masked_loss_ratio = masked_loss_ratio

        self.use_encodec_embeddings = use_embeddings

        self.mask_before = mask_before
        self.masked_objective = masked_objective
        self.contrastive_objective = contrastive_objective

        self.optimizer = optimizer

        self.mask_special_token = mask_special_token
        self.pad_special_token = pad_special_token

        self.supconloss = MySupConLoss(temperature=contrastive_temperature)
        self.proj_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 128)
        )

        self.contrastive_matrix = None
        self.first_run = True
        self.extended_batch_size = 0
        self.simple_batch_size = 0
        
        if resume_from_checkpoint:
            self.load_from_checkpoint(resume_from_checkpoint)
            
    def forward(self, x):
        # x is an array of mono 24kHz audio truncated to a certain length. ['original_len'] denotes the  number of samples in the original array (if truncated) ande can be used to construct the padding mask later on
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

        wav = wav.contiguous().view(N*B, channels, T)

        if self.first_run:
            print(f'tensor shape after augmentation batching: {wav.shape}')

        codes, embeddings = self.encodec.get_encodec_output(wav)

        if self.adapt_sequence_len and self.first_run:
            self.transformer_encoder.adapt_sequence_len(codes.shape[-1])
            self.transformer_decoder.adapt_sequence_len(codes.shape[-1])
            self.sequence_len = codes.shape[-1]

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
            print(f"codes_are_padded: {codes_are_padded.any()}")

        if self.only_global_contrastive:
            T = 0
        else:
            T = codes.shape[-1]

        if (self.first_run or T*B*N != self.extended_batch_size) and self.contrastive_objective:
            print(
                f'creating contrastive matrix for batch size {B*N} and sequence length {codes.shape[-1]} with {N} augmentations per sample')
            print(f'previous batch size was {self.extended_batch_size}')

            self.contrastive_matrix = self.get_contrastive_matrix(
                T=T, B=B, N=N, W=self.window_size, device=codes.device)
            
        if self.first_run:
            print(self.contrastive_matrix[0].shape)
            print(self.contrastive_matrix[1])

        encoded, unmasked_encoded, codes_mask, encoded_padding_mask, contrastive_matrix_dict = self.transformer_encoder(
            codes=codes,
            padding_mask=padding_mask,
            mask_before=self.mask_before,
            contrastive_matrix=self.contrastive_matrix,
            embeddings=embeddings,
            use_embeddings=self.use_encodec_embeddings
        )

        codes = codes.clone()
        codes[codes_are_padded] = self.pad_special_token

        masked_codes = codes.clone()
        # purely for logging and visual purposes
        masked_codes[codes_mask] = -1000

        decoded_logits = self.transformer_decoder(
            unmasked_encoded, padding_mask=encoded_padding_mask
        )  # SHAPE : [B,n_q,T,card]

        projected = self.proj_head(encoded)

        return {

            "encodec_embeddings": embeddings,
            "codes": codes,
            "masked_codes": masked_codes,
            "codes_mask": codes_mask,
            "padding_mask": padding_mask,
            "encoded": encoded,
            "projected": projected,
            "logits": decoded_logits,
            "contrastive_matrix_dict": contrastive_matrix_dict,
        }

    def finetune_forward(self, x):
        
        if isinstance(x, dict):  # when loading from a dataloader:
            wav = x["wav"]
            lens = x["original_lens"]
        else:  # when running inference on an array for testing purposes
            wav = x
            lens = None

        codes, embeddings = self.encodec.get_encodec_output(wav)

        if self.adapt_sequence_len and self.first_run:
            self.transformer_encoder.adapt_sequence_len(codes.shape[-1])
            self.transformer_decoder.adapt_sequence_len(codes.shape[-1])
            self.sequence_len = codes.shape[-1]

        padding_mask, codes = self.create_padding_mask(codes, lens)
        padding_mask, codes = self.pad_codes_to_length(codes, padding_mask)
        codes_are_padded = codes == self.pad_special_token

        if self.first_run:
            print(f"codes: {codes.shape}")
            print(f"padding_mask: {padding_mask.shape}")

        encoded, _, _, encoded_padding_mask, _ = self.transformer_encoder.finetune_forward(
            codes=codes,
            padding_mask=padding_mask,
            use_embeddings=self.use_encodec_embeddings,
            embeddings=embeddings
        )

        codes = codes.clone()
        codes[codes_are_padded] = self.pad_special_token

        projected = self.proj_head(encoded)

        self.first_run = False

        return {
            "encoded": encoded,
            "projected": projected,
            "codes": codes,
            "padding_mask": padding_mask,
            "encodec_embeddings": embeddings
        }

    def create_padding_mask(self, codes, original_lens):
        padding_mask = torch.zeros(
            (codes.shape[0], codes.shape[2]), dtype=bool, device=codes.device
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
        target_padding_mask_shape = (
            padding_mask.shape[0], self.sequence_len)
        target_codes_shapes = (
            codes.shape[0], codes.shape[1], self.sequence_len)
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

    def get_contrastive_matrix(self, B, T, N, W, device=None, window_stride=1):

        total_samples = B * T * N

        # Create a matrix indicating whether samples are from the same sequence

        diag_block = torch.ones((T, T), device=device)
        same_sequence = torch.block_diag(*((B * N) * [diag_block]))

        # Create a window mask
        window_width = W // 2 + 1
        window_mask = torch.diag(torch.ones((total_samples), device=device), 0)

        indices = torch.arange(total_samples, device=device).unsqueeze(0)

        # Create the window mask
        window_mask = (indices - indices.T).abs() < window_width

        # Mask out with stride, every row that is not a multiple of the stride
        window_mask[torch.arange(total_samples) % window_stride != 0] = 0

        # Add the transpose of the mask to itself to make it symmetric
        window_mask += window_mask.clone().T

        # Set all elements above zero to one
        window_mask[window_mask > 0] = 1

        # Apply the window mask
        window_mask = window_mask[:total_samples, :total_samples]

        self.extended_batch_size = total_samples
        self.simple_batch_size = B
        # Return the contrastive matrix
        return (same_sequence + window_mask)*same_sequence, (B, N, T, W)

    def get_contrastive_loss(self, encoded, contrastive_matrix_removed):

        Bext, T, d = encoded.shape
        only_class_tokens = encoded[:, 0, :].contiguous().view(-1, d)
        only_average_pool = torch.mean(encoded[:, 1:, :], dim=1)
        encoded = encoded.contiguous().view(-1, d)

        if not self.only_global_contrastive:
            only_class_tokens_matrix = contrastive_matrix_removed[::T, ::T]
            positive_mask = (contrastive_matrix_removed == 2)
            negative_mask = torch.ones_like(positive_mask).bool()
            neutral_mask = (contrastive_matrix_removed == 1)

            loss = self.supconloss(encoded, positive_mask=positive_mask,
                                   negative_mask=negative_mask, neutral_mask=neutral_mask)

        else:
            only_class_tokens_matrix = contrastive_matrix_removed

        positive_class = (only_class_tokens_matrix == 2)
        negative_class = torch.ones_like(positive_class).bool()

        assert only_average_pool.shape == only_class_tokens.shape, print(
            only_average_pool.shape, only_class_tokens.shape)

        if self.logger is not None and self.global_step % 100 == 0:
            if not self.only_global_contrastive:
                similarity_matrix = self.supconloss.get_similarities(encoded)
                similarity_matrix = similarity_matrix * \
                    (~(torch.eye(
                        similarity_matrix.shape[0], device=similarity_matrix.device).bool())).int()

                fig, ax = plt.subplots(1, 1)
                ax.imshow(similarity_matrix.detach(
                ).cpu().numpy(), cmap="plasma")
                self.logger.log_image(
                    "similarity matrix", [wandb.Image(fig)])
                plt.close(fig)

                # log a smaller subset of the similarity matrix
                fig, ax = plt.subplots(1, 1)
                ax.imshow(similarity_matrix.detach().cpu().numpy()[
                    :8 * T, :8 * T], cmap="plasma")
                self.logger.log_image(
                    "similarity matrix (smaller subset)", [wandb.Image(fig)])
                plt.close(fig)

            class_similarity_matrix = self.supconloss.get_similarities(
                only_class_tokens)
            avg_similarity_matrix = self.supconloss.get_similarities(
                only_average_pool)

            class_similarity_matrix = class_similarity_matrix * \
                (~(torch.eye(
                    class_similarity_matrix.shape[0], device=class_similarity_matrix.device).bool())).int()

            avg_similarity_matrix = avg_similarity_matrix * \
                (~(torch.eye(
                    avg_similarity_matrix.shape[0], device=avg_similarity_matrix.device).bool())).int()

            fig, ax = plt.subplots(1, 1)
            ax.imshow(class_similarity_matrix.detach(
            ).cpu().numpy(), cmap="plasma")
            self.logger.log_image(
                "class similarity matrix", [wandb.Image(fig)])
            plt.close(fig)

            fig, ax = plt.subplots(1, 1)
            ax.imshow(avg_similarity_matrix.detach(
            ).cpu().numpy(), cmap="plasma")
            self.logger.log_image(
                "average pool similarity matrix", [wandb.Image(fig)])
            plt.close(fig)

        class_loss = self.supconloss(
            only_class_tokens, positive_mask=positive_class, negative_mask=negative_class)

        average_pool_loss = self.supconloss(
            only_average_pool, positive_mask=positive_class, negative_mask=negative_class)

        global_loss = self.global_class_vs_average_contrastive_loss_ratio * class_loss + \
            (1-self.global_class_vs_average_contrastive_loss_ratio) * average_pool_loss

        if self.only_global_contrastive:
            return global_loss, global_loss, class_loss, average_pool_loss

        return loss, global_loss, class_loss, average_pool_loss

    def training_step(self, batch, batch_idx):
        torch.autograd.set_detect_anomaly(True)
        x = batch

        # if self.first_run:
        #     with profiler.profile(with_stack=True, profile_memory=True) as prof:
        #         out_ = self(batch)
        #     print(prof.key_averages(group_by_stack_n=5).table(
        #         sort_by='self_cpu_time_total', row_limit=20))

        # else:
        out_ = self(batch)

        logits = out_["logits"]
        encoded = out_["encoded"]
        projected = out_["projected"]
        codes = out_["codes"]
        codes_mask = out_['codes_mask']
        padding_mask = out_["padding_mask"]
        masked_codes = out_['masked_codes']
        contrastive_matrix_dict = out_['contrastive_matrix_dict']
        encodec_embeddings = out_['encodec_embeddings']

        all_losses, masked_unmasked, per_codebook = self.get_detailed_losses(
            logits[:, :, :, 1:], codes.clone().long(), codes_mask)
        masked_loss = self.masked_loss_ratio * masked_unmasked['masked_loss'] + (
            1-self.masked_loss_ratio) * masked_unmasked['unmasked_loss']

        if self.contrastive_objective:
            contrastive_loss, contrastive_global_loss, contrastive_class_loss, contrastive_avg_loss = self.get_contrastive_loss(
                projected, contrastive_matrix_dict['contrastive_matrix_masked_removed'])

            # weigh the contrastive loss by a factor of self.global_vs_local_contrastive_loss_ratio
            contrastive_loss = contrastive_global_loss * self.global_vs_local_contrastive_loss_ratio + \
                contrastive_loss * \
                (1-self.global_vs_local_contrastive_loss_ratio)
        else:
            contrastive_loss, contrastive_global_loss, contrastive_class_loss, contrastive_avg_loss = 0, 0, 0, 0

        for k in all_losses.keys():
            self.log(k, all_losses[k], sync_dist=True, on_step=True)

        for k in masked_unmasked.keys():
            self.log(k, masked_unmasked[k], sync_dist=True, on_step=True)

        for k in per_codebook.keys():
            self.log(k, per_codebook[k], sync_dist=True,    on_step=True)

        self.log("contrastive loss", contrastive_loss,
                 prog_bar=True, sync_dist=True, on_step=True)
        self.log("contrastive global loss",
                 contrastive_global_loss, sync_dist=True, on_step=True)
        self.log("contrastive class loss",
                 contrastive_class_loss, sync_dist=True, on_step=True)
        self.log("contrastive avg loss",
                 contrastive_avg_loss, sync_dist=True, on_step=True)

        self.log("masked loss", masked_loss,
                 prog_bar=True, sync_dist=True, on_step=True)

        self.log(
            "lr", self.trainer.optimizers[0].param_groups[0]['lr'], sync_dist=True, on_step=True)

        if self.logger is not None:

            # if self.global_step % 100 == 0:
            #     # random sample a portion of encodec embeddings along dimension 1
            #     perm = torch.randperm(encodec_embeddings.shape[1])
            #     idx = perm[:int(encodec_embeddings.shape[1] * (1-self.transformer_encoder.mask_p))]
            #     encodec_embeddings = encodec_embeddings[:,idx,:]
            #     encodec_embeddings = encodec_embeddings.contiguous().view(-1,encodec_embeddings.shape[-1])

            #     print("encodec embeddings shape :")
            #     print(encodec_embeddings.shape)

            # encodec_embeddings = encodec_embeddings.contiguous().view(-1,encodec_embeddings.shape[-1])

            # embeddings_sim = self.supconloss.get_similarities(encodec_embeddings)
            # fig, ax = plt.subplots(1, 1)
            # ax.imshow(embeddings_sim.detach().cpu().numpy(), cmap="plasma")
            # self.logger.log_image(
            #     "frozen encodec embeddings similarity matrix", [wandb.Image(fig)])
            # plt.close(fig)

            if self.global_step % 1000 == 0:
                fig, ax = plt.subplots(2, 1)
                ax[0].imshow(masked_codes[0, :, :20].cpu().numpy(),
                             vmin=-1000, vmax=1000, cmap="plasma")
                ax[1].imshow(codes[0, :, :20].cpu().numpy(),
                             vmin=-1000, vmax=1000, cmap='plasma')

                self.logger.log_image(
                    "masked and unmasked tokens", [wandb.Image(fig)])
                plt.close(fig)

                if self.contrastive_objective:
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

        if self.masked_objective:
            loss = self.contrastive_to_masked_ratio * contrastive_loss + \
                (1-self.contrastive_to_masked_ratio) * masked_loss
        else:
            loss = contrastive_loss

        self.first_run = False
        
        
        self.log('overall_loss', loss, prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch):
        out_ = self(batch)
        logits = out_["logits"]
        projected = out_["projected"]
        codes = out_["codes"]
        codes_mask = out_['codes_mask']
        contrastive_matrix_dict = out_['contrastive_matrix_dict']

        all_losses, masked_unmasked, per_codebook = self.get_detailed_losses(
            logits[:, :, :, 1:], codes.clone().long(), codes_mask)
        masked_loss = self.masked_loss_ratio * masked_unmasked['masked_loss'] + (
            1-self.masked_loss_ratio) * masked_unmasked['unmasked_loss']

        if self.contrastive_objective:
            contrastive_loss, contrastive_global_loss, contrastive_class_loss, contrastive_avg_loss = self.get_contrastive_loss(
                projected, contrastive_matrix_dict['contrastive_matrix_masked_removed'])

            # weigh the contrastive loss by a factor of self.global_vs_local_contrastive_loss_ratio
            contrastive_loss = contrastive_global_loss * self.global_vs_local_contrastive_loss_ratio + \
                contrastive_loss * \
                (1-self.global_vs_local_contrastive_loss_ratio)
        else:
            contrastive_loss, contrastive_global_loss, contrastive_class_loss, contrastive_avg_loss = 0, 0, 0, 0

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
            loss = self.contrastive_to_masked_ratio * contrastive_loss + \
                (1-self.contrastive_to_masked_ratio) * masked_loss
        else:
            loss = contrastive_loss

        self.log('overall_loss', loss, prog_bar=False, sync_dist=True)

        return loss

    def configure_optimizers(self):
        if self.optimizer is None:
            optimizer = optim.AdamW(
                self.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        else:
            optimizer = self.optimizer(self.parameters())

        scheduler = ReduceLROnPlateau(
            optimizer=optimizer, factor=0.5, patience=20, min_lr=1e-6, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": self.reduce_lr_monitor, }

    def get_detailed_losses(self, logits, codes, codes_mask):

        B, C, Q, T = logits.shape
        all_losses = {}
        masked_unmasked = {}
        per_codebook = {}
        loss = F.cross_entropy(
            logits, codes, reduction='none', ignore_index=self.pad_special_token)

        masked_loss = loss*codes_mask
        unmasked_loss = loss*(~codes_mask)

        codes_mask_sum = torch.sum(codes_mask)
        inverse_codes_mask_sum = torch.sum(~codes_mask)

        for q in range(Q):
            all_losses[f'masked_loss_q{q}'] = torch.sum(
                masked_loss[:, q, :]) / codes_mask_sum
            all_losses[f'unmasked_loss_q{q}'] = torch.sum(
                unmasked_loss[:, q, :]) / inverse_codes_mask_sum
            per_codebook[f'global_loss_q{q}'] = all_losses[f'masked_loss_q{q}'] + \
                all_losses[f'unmasked_loss_q{q}']

        masked_loss = torch.sum(masked_loss)/codes_mask_sum
        unmasked_loss = torch.sum(unmasked_loss)/inverse_codes_mask_sum
        masked_unmasked['masked_loss'] = masked_loss
        masked_unmasked['unmasked_loss'] = unmasked_loss
        return all_losses, masked_unmasked, per_codebook

    # def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    #     checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if not k.startswith('encodec')}
    #     # no need to save the encodec state dict as the encodec model is frozen
    #     return super().on_save_checkpoint(checkpoint)
    
    # def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    #     # add the current encodec state dict to the checkpoint
    #     # get the encodec state dict from the current state dict of the model
    #     encodec_state_dict = {k: v for k, v in self.state_dict().items() if k.startswith('encodec')}
    #     checkpoint['state_dict'].update(encodec_state_dict)
    #     print(checkpoint['state_dict'])
    #     return super().on_load_checkpoint(checkpoint)