import os
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from encodec.utils import convert_audio
import torch
import numpy as np
import soundfile as sf
from src.dataloading.augmentations import *
import pandas as pd


class GTZANFinetuneDataset(Dataset):
    def __init__(self, data_dir, file_list=None, augmentations=None, transform=True, target_sample_rate=24000, target_length=20, train=True, n_augmentations=1, extension="wav", sanity_check_n = None):
        self.data_dir = data_dir
        self.extension = extension
        self.augmentations = augmentations
        self.transform = transform
        
        # annotation files for the GTZAN dataset
        self.annotations = pd.read_csv(os.path.join("data/gtzan_annotations.csv"))
        print(self.annotations.head())
        
        # add a column to annotations with the class index
        self.annotations["class_idx"] = self.annotations["genre"].astype('category').cat.codes
        
        
        if sanity_check_n is not None:
            self.transform = False  ## turn off augmentations for sanity checks
            self.annotations = self.annotations.iloc[:sanity_check_n]
            
        
        self.target_sample_rate = target_sample_rate
        self.target_length = target_length
        self.n_augmentations = n_augmentations
        self.target_n_samples_one = target_sample_rate*target_length
        self.target_n_samples = target_sample_rate*target_length*n_augmentations
        # e.g if the target length is 5s and the target number of augmentations is 4 then the target length of audio to split is 20s
        # note that for n_augmentations = 1, it defaults back to a classic dataloading case
        self.train = train
        

    def __len__(self):
        return len(self.annotations)



    def __getitem__(self, idx):
        file_name = self.annotations.iloc[idx]["filename"]
        file_folder = self.annotations.iloc[idx]["genre"]
        file_path = os.path.join(self.data_dir, file_folder, file_name)
        # get the label
        label = self.annotations.iloc[idx]["class_idx"]
        


        info = sf.info(file_path)
        sample_rate = info.samplerate
        n_frames = info.frames

        if n_frames < self.target_n_samples/self.target_sample_rate * sample_rate:
            return (self[idx+1])
        new_target_n_samples = int(
            self.target_n_samples/self.target_sample_rate * sample_rate)
        start_idx = np.random.randint(low=0, high=sf.info(
            file_path).frames - new_target_n_samples)
        waveform, sample_rate = sf.read(
            file_path, start=start_idx, stop=start_idx + new_target_n_samples, dtype='float32', always_2d=True)

        waveform = torch.Tensor(waveform.transpose())
        encodec_audio = convert_audio(
            
            waveform, sample_rate, self.target_sample_rate, 1)
        
        if self.augmentations is not None and self.transform and self.train:
            encodec_audio = self.augmentations(encodec_audio)
            
            
        

        return {
            "wav": encodec_audio,
            "label": int(label),
            "original_lens": self.target_n_samples
        }


class FineTuneDataModule(pl.LightningDataModule):
    def __init__(self, train_data_dir=None, val_data_dir=None, batch_size=64, num_workers=0, target_sample_rate=24000, target_length=20, validation_split=None, n_augmentations=1, transform=True, sanity_check_n=None, task = 'GTZAN'):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        if val_data_dir is None and validation_split is not None:
            self.val_data_dir = self.train_data_dir

        self.validation_split = validation_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_augmentations = n_augmentations
        self.transform = transform
        self.sanity_check_n = sanity_check_n

        self.target_sample_rate = target_sample_rate
        self.target_length = target_length

        self.train_transforms = ComposeManySplit(
            [
                RandomApply([PolarityInversion()], p=0.5),
                RandomApply(
                    [Noise(min_snr=0.001, max_snr=0.005)],
                    p=0.5,
                ),
                RandomApply([Gain()], p=0.5),
                RandomApply(
                    [HighLowPass(sample_rate=self.target_sample_rate)], p=0.5
                ),
            ]
        )

        self.val_transforms = None
        self.task = task

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.task == 'GTZAN':
                self.train_dataset = GTZANFinetuneDataset(data_dir=self.train_data_dir, augmentations=self.train_transforms, target_sample_rate=self.target_sample_rate,
                                                        target_length=self.target_length, n_augmentations=self.n_augmentations, transform=self.transform, sanity_check_n=self.sanity_check_n)
            # self.val_dataset = CustomAudioDataset(data_dir = self.val_data_dir, transform=self.val_transforms, target_sample_rate = self.target_sample_rate, target_length = self.target_length, n_augmentations=2)
            if self.validation_split is not None:
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    self.train_dataset, [1-self.validation_split, self.validation_split])
                self.val_dataset.train = False
                self.val_dataset.transform = False

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
