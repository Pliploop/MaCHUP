from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from src.dataloading.augmentations import *
import pandas as pd
from src.dataloading import DatasetRouter


class FineTuneDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, annotations_path='', batch_size=64, num_workers=0, target_sample_rate=24000, target_length=20, validation_split=0, test_split=0, n_augmentations=1, transform=True, sanity_check_n=None, task='GTZAN', extension="wav"):
        super().__init__()

        self.data_dir = data_dir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_augmentations = n_augmentations
        self.transform = transform
        self.sanity_check_n = sanity_check_n

        self.target_sample_rate = target_sample_rate
        self.target_length = target_length
        self.extension = extension

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
        
        print(task)

        self.dataset_router = DatasetRouter(task=task, annotations_path=annotations_path, data_dir=self.data_dir, augmentations=self.train_transforms, target_sample_rate=self.target_sample_rate,
                                            target_length=self.target_length, n_augmentations=self.n_augmentations, transform=self.transform, sanity_check_n=self.sanity_check_n, validation_split=validation_split, test_split=test_split, extension=self.extension)

    def setup(self, stage=None):

            self.train_dataset = self.dataset_router.get_train_dataset()
            self.val_dataset = self.dataset_router.get_val_dataset()
            self.test_dataset = self.dataset_router.get_test_dataset()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)
