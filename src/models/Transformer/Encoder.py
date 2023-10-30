import random
import torch
from torch import nn
from linformer import Linformer
from src.models.utils import LearnedPositionalEncoding, PositionalEncoding
from torch.nn import LayerNorm, TransformerEncoder, TransformerEncoderLayer


class Embed(nn.Module):
    def __init__(
        self, embedding_behaviour, embedding_sizes, n_codebooks, card, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embedding_behaviour = embedding_behaviour
        self.embedding_sizes = embedding_sizes
        self.n_codebooks = n_codebooks
        self.card = card

        self.emb = nn.ModuleList(
            [
                nn.Embedding(self.card + 3, self.embedding_sizes[codebook])
                for codebook in range(self.n_codebooks)
            ]
        )

        # +3 for pad, pattern tokens, and mask tokens

    def forward(self, indices):
        B, K, T = indices.shape

        embeddings = [self.emb[k](indices[:, k])
                      for k in range(K)]  # shape B,T,E
        if self.embedding_behaviour == "sum":
            input_ = sum(embeddings)
        else:
            input_ = torch.cat(embeddings, dim=-1)

        return input_


class Encoder(nn.Module):
    """ "Default transformer encoder. Default behaviour is according to encodecMAE (or similar):

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
        embedding_behaviour="concat",
        position_encoder="sinusoidal",
        sequence_len=2048,
        layers=6,
        n_heads=8,
        p=0.5,
        batched_mask=False,
        mask_special_token=1025,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.n_codebooks = n_codebooks
        self.embedding_size = embedding_size
        self.card = card
        self.sequence_len = sequence_len
        self.embedding_behaviour = embedding_behaviour
        self.mask_special_token = mask_special_token

        if self.embedding_behaviour == "concat":
            self.d_model = sum(self.embedding_size)
        else:
            self.d_model = self.embedding_size[0]

        if position_encoder == "sinusoidal":
            self.position_encoder = PositionalEncoding(
                self.d_model, max_len=self.sequence_len
            )
        else:
            self.position_encoder = LearnedPositionalEncoding(
                self.d_model, mex_len=self.sequence_len
            )

        self.emb = Embed(
            embedding_behaviour=self.embedding_behaviour,
            embedding_sizes=self.embedding_size,
            card=self.card,
            n_codebooks=self.n_codebooks,
        )
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.embedding_size[codebook], self.card)
                for codebook in range(self.n_codebooks)
            ]
        )

        self.n_heads = n_heads
        self.layers = layers

        self.transformer = None

        self.norm_in = LayerNorm(self.d_model)

        self.class_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.mask_token = nn.Parameter(torch.randn(self.d_model))
        self.encoder_mask_emb = nn.Parameter(
            torch.FloatTensor(self.d_model).uniform_())
        self.mask_p = p
        self.batched_mask = batched_mask


        self.first_run = True

    def forward(self, codes, padding_mask=None, mask_before=False, mask=True):
        # indices is of shape B,n_q,T
        B, K, T = codes.shape

        # masking before, i.e all timesteps are sent to be encoded but allows for structured masking algos.
        if mask_before and mask:
            (
                masked_idx,
                retained_idx,
                retained_padding_mask,
                codes,
                codes_mask,
            ) = self.mask_before(padding_mask=padding_mask, codes=codes)

        original_embeddings = self.emb(codes)  # B,T,d_model
        original_embeddings = self.position_encoder(original_embeddings)

        if not mask_before and mask:
            (
                input_,
                masked_idx,
                retained_idx,
                retained_padding_mask,
                codes_mask,
            ) = self.mask_after(
                x=original_embeddings, padding_mask=padding_mask, codes=codes
            )

        if mask_before and mask:
            input_ = original_embeddings

        if not mask:
            codes_mask = torch.zeros_like(codes).to(codes.device)
            retained_padding_mask = padding_mask
            input_ = original_embeddings

        # shape B,T,d_model
        class_token = self.class_token.expand(
            B, 1, self.d_model
        )  # Expand class token for batch
        retained_padding_mask = torch.cat(
            [
                torch.zeros(B, 1, device=retained_padding_mask.device),
                retained_padding_mask,
            ],
            dim=1,
        )
        padding_mask = torch.cat(
            [torch.zeros(B, 1, device=retained_padding_mask.device),
             padding_mask],
            dim=1,
        )

        input_ = torch.cat([class_token, input_], dim=1)
        input_ = self.norm_in(input_)
        output_ = self.transformer(
            input_, src_key_padding_mask=retained_padding_mask)
        # shape B,T,d_model


        if self.first_run:
            print("shape coming out of encoder: ============")
            print(output_.shape)

        # unmasking happens here
        if self.first_run:
            print("========= retained idx shape ============")
            print(torch.tensor(retained_idx).shape)

        unmasked_output = self.unmask(
            embeddings=output_,
            original_embeddings=original_embeddings,
            masked_idx=masked_idx,
            retained_idx=retained_idx,
            retained_padding_mask=retained_padding_mask,
        )

        if self.first_run:
            print("========= All outputs for the encoder========")
            print("-------- masked output with class token --------")
            print(output_.shape)
            print(
                "-------- unmasked output with class token and original embeddingd without class token --------"
            )
            print(unmasked_output.shape)
            print(original_embeddings.shape)
            print("-------- codes_mask.shape ---------")
            print(codes_mask.shape)
            print("------ padding_mask.shape ----------")
            print(padding_mask.shape)

        self.first_run = False

        return output_, unmasked_output, codes_mask, padding_mask
    

    def unmask(
        self,
        embeddings,
        original_embeddings,
        masked_idx,
        retained_idx,
        retained_padding_mask,
    ):
        class_token = embeddings[:, 0, :].unsqueeze(1)
        without_class_token = embeddings[:, 1:, :]
        all_masked = torch.empty(
            original_embeddings.shape,
            device=original_embeddings.device,
            dtype=original_embeddings.dtype,
        )

        if self.first_run:
            print("=========== Masked without embeddings shape ========")
            print(all_masked.shape)

        for i, (cur_feat, ridx, midx) in enumerate(
            zip(without_class_token, retained_idx, masked_idx)
        ):
            all_masked[i, ridx] = cur_feat
            all_masked[i, midx] = self.mask_token

        if self.first_run:
            print("========= all_masked.shape ==========")
            print(all_masked.shape)
            print("========= around masked index ========")
            print(all_masked[0, retained_idx[1][1] -
                1: retained_idx[1][1] + 1, :3])

            print(all_masked[1, retained_idx[0][1] -
                1: retained_idx[0][1] + 1, :3])
            print("class token =============")
            print(class_token.shape)

        all_masked = torch.cat([class_token, all_masked], dim=1)

        return all_masked
    
    def mask_before(self, padding_mask, codes):
        """creates a mask for the input. the input here is codes of shape B, K, T."""

        B, K, T = codes.shape

        if self.first_run:
            print(f"masking before with masking proba : {self.mask_p}")

        # used to compute loss over masked tokens. because this is before, mask should already be computed
        codes_mask = torch.zeros_like(codes).to(codes.device)

        # multiple ways to mask here:
        # Randomly (easiest, first to implement)
        # with chunks
        # column-wise (same thing)
        # full codebook (probably a hard regularizer, only restrain to one codebook)
        retained_idx = [list(range(T)) for b in range(B)]
        masked_idx = []

        # random masking

        codes_mask = torch.empty(codes.shape).uniform_() > self.mask_p
        codes[codes_mask] = self.mask_special_token

        if self.first_run:
            print("============== codes_masking ==============")
            print(codes_mask.shape)
            print(f"{codes_mask.sum()} tokens were masked with random masking")

        retained_padding_mask = padding_mask

        # do some checking here for whole masked columns -> change retained_idx, masked_idx, and retained_padding_mask

        self.first_run = False
        return masked_idx, retained_idx, retained_padding_mask, codes, codes_mask
        # All masking modules will return:
        # the list of masked indices, the list of unsmaked indices, the retained padding mask, the retained features, the masked code matrix (boolean)

    def mask_after(self, x, padding_mask, codes):
        """creates a mask for the input. note that the same amount of tokens must be masked in each batch (because of matrices). so either:
        - mask a precise amount of tokens
        - create a batch mask (easier)
        """
        # sample binomial probability

        B, T, d_model = x.shape
        num_retained_tokens = int((1 - self.mask_p) * T)
        num_retained_tokens = max(1, num_retained_tokens)

        if self.first_run:
            print(f"masking proba : {self.mask_p}")
            print(f"tokens to mask : {T-num_retained_tokens}")

        # used to compute loss over masked tokens. because this is after, mask by columns
        codes_mask = torch.zeros_like(codes).to(codes.device)

        retained_idx = []
        masked_idx = []

        for i in range(B):
            idx = list(range(T))
            random.shuffle(idx)
            cur_retained_idx = idx[:num_retained_tokens]
            retained_idx.append(cur_retained_idx)
            cur_masked_idx = idx[num_retained_tokens:]
            masked_idx.append(cur_masked_idx)
            x[i, cur_masked_idx] = self.encoder_mask_emb
            codes_mask[i, :, cur_masked_idx] = 1

        if self.batched_mask:
            x = x[:, retained_idx[0]]
            retained_padding_mask = padding_mask[:, retained_idx[0]]
        else:
            new_x = []
            retained_padding_mask = []
            for i in range(B):
                new_x.append(x[i, retained_idx[i]])
                retained_padding_mask.append(padding_mask[i, retained_idx[i]])
            x = torch.stack(new_x, dim=0)
            retained_padding_mask = torch.stack(retained_padding_mask, dim=0)

        if self.first_run:
            print("============== codes_masking ==============")
            print(codes_mask)
            print(codes_mask.shape)

            print("============== new x shape ================")
            print(x.shape)

        self.first_run = False
        return x, masked_idx, retained_idx, retained_padding_mask, codes_mask
        # All masking modules will return:
        # the list of masked indices, the list of unsmaked indices, the retained padding mask, the retained features, the masked code matrix (boolean)


class LinearEncoder(Encoder):
    """ "Does not work yet because of paddding mask implementation"""

    def __init__(
        self,
        n_codebooks=4,
        embedding_size=[512, 256, 128, 64],
        card=1024,
        embedding_behaviour="concat",
        position_encoder="sinusoidal",
        sequence_len=2048,
        layers=6,
        n_heads=8,
        p=0.5,
        batched_mask=False,
        mask_special_token=1025,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            n_codebooks,
            embedding_size,
            card,
            embedding_behaviour,
            position_encoder,
            sequence_len,
            layers,
            n_heads,
            p,
            batched_mask,
            mask_special_token,
            *args,
            **kwargs,
        )
        self.norm = LayerNorm(self.d_model)
        self.transformer = Linformer(
            self.d_model, self.sequence_len, self.layers)


class VanillaEncoder(Encoder):
    def __init__(
        self,
        n_codebooks=4,
        embedding_size=[512, 256, 128, 64],
        card=1024,
        embedding_behaviour="concat",
        position_encoder="sinusoidal",
        sequence_len=2048,
        layers=6,
        n_heads=8,
        p=0.5,
        batched_mask=False,
        mask_special_token=1025,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            n_codebooks,
            embedding_size,
            card,
            embedding_behaviour,
            position_encoder,
            sequence_len,
            layers,
            n_heads,
            p,
            batched_mask,
            mask_special_token,
            *args,
            **kwargs,
        )
        self.norm = LayerNorm(self.d_model)
        self.transformer_layer = TransformerEncoderLayer(
            self.d_model, self.n_heads, activation="gelu", batch_first=True
        )
        self.transformer = TransformerEncoder(
            self.transformer_layer, self.layers, norm=self.norm
        )
