
import torch
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




class GTZANTrainDataset(Dataset):
    def __init__(self, data_dir, annotations, augmentations=None, transform=True, target_sample_rate=24000, target_length=20, train=True, n_augmentations=1, extension="wav", sanity_check_n = None, *args, **kwargs):
        self.data_dir = data_dir
        self.extension = extension
        self.augmentations = augmentations
        self.transform = transform
        
        self.mode = 'train'
        self.annotations = annotations
        print(self.annotations.head())
        
        
        if sanity_check_n is not None:
            self.transform = False  ## turn off augmentations for sanity checks
            self.annotations = self.annotations.iloc[:sanity_check_n]
            
        
        self.target_sample_rate = target_sample_rate
        self.target_length = target_length
        self.n_augmentations = n_augmentations
        self.target_n_samples_one = target_sample_rate*target_length
        self.target_n_samples = target_sample_rate*target_length*n_augmentations
        self.train = train


    def __len__(self):
        return len(self.annotations)
    

    def __getitem__(self, idx):
        
        file_name = self.annotations.iloc[idx]["filename"]
        file_folder = self.annotations.iloc[idx]["genre"]
        file_path = os.path.join(self.data_dir, file_folder, file_name)
        label = self.annotations.iloc[idx]["class_idx"]
        


        info = sf.info(file_path)
        sample_rate = info.samplerate
        n_frames = info.frames

        if n_frames < self.target_n_samples/self.target_sample_rate * sample_rate:
            return (self[idx+1])
        new_target_n_samples = int(
            self.target_n_samples/self.target_sample_rate * sample_rate)
        start_idx = np.random.randint(low=0, high=n_frames - new_target_n_samples)
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
        
        
        
        
class GTZANTestDataset(GTZANTrainDataset):
    
    def __init__(self, data_dir, annotations, augmentations=None, transform=True, target_sample_rate=24000, target_length=20, train=True, n_augmentations=1, extension="wav", sanity_check_n=None, *args, **kwargs):
        super().__init__(data_dir, annotations, augmentations, transform, target_sample_rate, target_length, train, n_augmentations, extension, sanity_check_n, *args, **kwargs)

    def __getitem__(self, idx):
        
        file_name = self.annotations.iloc[idx]["filename"]
        file_folder = self.annotations.iloc[idx]["genre"]
        file_path = os.path.join(self.data_dir, file_folder, file_name)
        label = self.annotations.iloc[idx]["class_idx"]
        


        info = sf.info(file_path)
        sample_rate = info.samplerate
        n_frames = info.frames

        waveform, sample_rate = sf.read(
            file_path, dtype='float32', always_2d=True)

        waveform = torch.Tensor(waveform.transpose())
        encodec_audio = convert_audio(
            
            waveform, sample_rate, self.target_sample_rate, 1)
        
        encodec_audio = torch.cat(torch.split(encodec_audio, self.target_n_samples_one, dim=1)[:-1],dim=0)
        
        return {
            "wav": encodec_audio,
            "label": int(label),
            "original_lens": self.target_n_samples
        }
        
