# MuMRVQ
Multiscale Contrastive Learning for Music with RVQ

Using RVQ and Multitask for representation learning:

Ideas include
  - MLM and Contrastive
  - IntraModal contrastive and Intermodal contrastive
  - RFSQ
  - Parallel cross-attention with modality dropout
    

TODO:

- Masking in module: structured loss before and unstructured loss after
- first training runs with reconstruction loss only on clotho, FMA
- Clear up ideas for dual loss, per-codebook loss, local vs global contrastive loss
- implement augmentations
- implement contrastive learning dataset
- Linear attention in decoder and encoder
- wandb logging and config saving + checkpointing
- clean config file with all necessary items.

