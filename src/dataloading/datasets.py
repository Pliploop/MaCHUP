import os
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from encodec.utils import convert_audio
import torch
import numpy as np


class CustomAudioDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_sample_rate = 24000, target_length = 20):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.target_length = target_length
        self.target_n_samples = target_sample_rate*target_length

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)

        waveform, sample_rate = torchaudio.load(file_path)

        if self.transform:
            waveform = self.transform(waveform)


        encodec_audio = convert_audio(waveform,sample_rate,self.target_sample_rate,1)

        empty = torch.zeros(self.target_n_samples)

        if encodec_audio.shape[-1] > self.target_n_samples:
            start = np.random.randint(0,self.target_n_samples - encodec_audio.shape[-1])
            empty = encodec_audio[:,start:]
            original_len = self.target_n_samples
        else:
            empty[:,:encodec_audio.shape[-1]] = encodec_audio
            original_len = encodec_audio.shape[-1]


        return {
            "wav" : waveform,
            "original_len" : original_len
        }

class CustomAudioDataModule(pl.LightningDataModule):
    def __init__(self, train_data_dir = '', val_data_dir = '', batch_size=64, num_workers = 0, target_sample_rate=24000, target_length = 20):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.batch_size = batch_size
        self.train_transforms = None
        self.val_transforms = None
        self.num_workers = num_workers
        self.target_sample_rate = target_sample_rate
        self.target_length = target_length

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CustomAudioDataset(data_dir=self.train_data_dir, transform=self.train_transforms, target_sample_rate = self.target_sample_rate, target_length = self.target_length )
            self.val_dataset = CustomAudioDataset(data_dir=self.val_data_dir, transform=self.val_transforms, target_sample_rate = self.target_sample_rate, target_length = self.target_length)

        ## maybe include fine-tuning datasets and evaluation datasets here

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)



