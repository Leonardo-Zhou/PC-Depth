from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler

import datasets.custom_transforms as custom_transforms
from config import get_training_size
from datasets.train_folders import *
# from datasets.validation_folders import ValidationSet


class VideosDataModule(LightningDataModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.training_size = get_training_size(hparams.dataset_name)

        # data loader
        # cancel custom_transforms.Normalize()
        self.train_transform = custom_transforms.Compose([
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.RandomScaleCrop(),
            custom_transforms.RescaleTo(self.training_size),
            custom_transforms.ArrayToTensor(),]
        )
        self.valid_transform = custom_transforms.Compose([
            custom_transforms.RescaleTo(self.training_size),
            custom_transforms.ArrayToTensor(),]
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        print("model version: ", self.hparams.hparams.model_version)
        if self.hparams.hparams.model_version == "SC-Depth":
            self.train_dataset = TrainFolder(
                self.hparams.hparams.dataset_dir,
                train=True,
                transform=self.train_transform,
                sequence_length=self.hparams.hparams.sequence_length,
                skip_frames=self.hparams.hparams.skip_frames,
                dataset=self.hparams.hparams.dataset_name,
                do_color_aug=self.hparams.hparams.do_color_aug,
                split=self.hparams.hparams.split
            )

            self.val_dataset = TrainFolder(
                self.hparams.hparams.dataset_dir,
                train=False,
                transform=self.valid_transform,
                sequence_length=self.hparams.hparams.sequence_length,
                skip_frames=self.hparams.hparams.skip_frames,
                dataset=self.hparams.hparams.dataset_name,
            )
        elif self.hparams.hparams.model_version == "PC-Depth":
            self.train_dataset = TrainFolderIA(
                self.hparams.hparams.dataset_dir,
                train=True,
                transform=self.train_transform,
                sequence_length=self.hparams.hparams.sequence_length,
                skip_frames=self.hparams.hparams.skip_frames,
                dataset=self.hparams.hparams.dataset_name,
                do_color_aug=self.hparams.hparams.do_color_aug,
                split=self.hparams.hparams.split
            )

            self.val_dataset = TrainFolderIA(
                self.hparams.hparams.dataset_dir,
                train=False,
                transform=self.valid_transform,
                sequence_length=self.hparams.hparams.sequence_length,
                skip_frames=self.hparams.hparams.skip_frames,
                dataset=self.hparams.hparams.dataset_name
            )

        print('{} samples found for training'.format(len(self.train_dataset)))
        print('{} samples found for validation'.format(len(self.val_dataset)))


    def train_dataloader(self):
        sampler = RandomSampler(self.train_dataset,
                                replacement=True,
                                # num_samples=self.hparams.hparams.batch_size * self.hparams.hparams.epoch_size
                                )
        return DataLoader(self.train_dataset,
                          sampler=sampler,
                          num_workers=8,
                          batch_size=self.hparams.hparams.batch_size,
                          pin_memory=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=8,
                          batch_size=self.hparams.hparams.batch_size,
                          pin_memory=True,
                          drop_last=True)
