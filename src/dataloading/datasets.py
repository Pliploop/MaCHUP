import os
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from encodec.utils import convert_audio
import torch
import numpy as np
import soundfile as sf
from torch_audiomentations import *


class CustomAudioDataset(Dataset):
    def __init__(self, data_dir, file_list=None, augmentations=None, transform=True, target_sample_rate=24000, target_length=20, train=True, n_augmentations=1, extension="wav", sanity_check_n = None):
        self.data_dir = data_dir
        self.extension = extension
        self.augmentations = augmentations
        self.transform = transform

        self.file_list = self.get_file_list(
            data_dir=data_dir, extenstion=extension)
        
        if sanity_check_n is not None:
            self.transform = False  ## turn off augmentations for sanity checks
            self.file_list = self.file_list[:sanity_check_n]
            
        
        self.target_sample_rate = target_sample_rate
        self.target_length = target_length
        self.n_augmentations = n_augmentations
        self.target_n_samples_one = target_sample_rate*target_length
        self.target_n_samples = target_sample_rate*target_length*n_augmentations
        # e.g if the target length is 5s and the target number of augmentations is 4 then the target length of audio to split is 20s
        # note that for n_augmentations = 1, it defaults back to a classic dataloading case
        self.train = train
        

    def __len__(self):
        return len(self.file_list)

    def get_file_list(self, data_dir, extenstion):

        file_paths = []

        # Walk through the directory and get relative paths of files with the specified extension
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(extenstion):
                    relative_path = os.path.relpath(
                        os.path.join(root, file), data_dir)
                    file_paths.append(relative_path)

        return file_paths

    def __getitem__(self, idx):
        
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)

        # waveform, sample_rate = torchaudio.load(file_path)
        try:
            info = sf.info(file_path)
            sample_rate = info.samplerate
            n_frames = info.frames
            
            
            new_target_n_samples = int(
                self.target_n_samples/self.target_sample_rate * sample_rate)

            if n_frames <= new_target_n_samples:
                return (self[idx+1])
            start_idx = np.random.randint(low=0, high=n_frames - new_target_n_samples)
            waveform, sample_rate = sf.read(
                file_path, start=start_idx, stop=start_idx + new_target_n_samples, dtype='float32', always_2d=True)
        except:
            return (self[idx+1])

        waveform = torch.Tensor(waveform.transpose())
        encodec_audio = convert_audio(
            waveform, sample_rate, self.target_sample_rate, 1)

        # Do the split here
        waveform = torch.cat(torch.split(
            encodec_audio, self.target_n_samples_one, dim=1)).unsqueeze(1)
        
        if self.augmentations is not None and self.transform and self.train:
            waveform = self.augmentations(waveform)
            
        ## end up with a [x_batch, 1, T] tensor
        
            

        return {
            "wav": waveform,
            "original_lens": self.target_n_samples
        }


class CustomAudioDataModule(pl.LightningDataModule):
    def __init__(self, train_data_dir=None, val_data_dir=None, batch_size=64, num_workers=0, target_sample_rate=24000, target_length=20, validation_split=None, n_augmentations=1, transform=True, sanity_check_n=None):
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

        self.train_transforms = Compose(
                [
                    Gain(
                        min_gain_in_db=-15.0,
                        max_gain_in_db=5.0,
                        p=0.4,
                        sample_rate=24000
                    ),
                    PolarityInversion(p=0.6, sample_rate=24000),
                    AddColoredNoise(p=0.3, sample_rate=24000),
                    PitchShift(p=0.3, sample_rate = 24000),
                    OneOf([
                        BandPassFilter(p=0.3, sample_rate = 24000),
                        BandStopFilter(p=0.3, sample_rate = 24000),
                        HighPassFilter(p=0.3, sample_rate = 24000),
                        LowPassFilter(p=0.3, sample_rate = 24000),
                    ])
                ]
            )

        self.val_transforms = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CustomAudioDataset(data_dir=self.train_data_dir, augmentations=self.train_transforms, target_sample_rate=self.target_sample_rate,
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
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)
