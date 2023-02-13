from torchvision import transforms
import torchvision
import pytorch_lightning as pl
import random
from imgaug import augmenters as iaa
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import math
from PIL import Image
from PIL import ImageOps
import traceback
import torch
from datamodule import train_transforms, test_transforms, filter_files

class ImagesDataset(Dataset):
    def __init__(self, path, files, transform):
        self.path = path
        self.files = files
        self.transform = transform

    def __getitem__(self, index):
        full_path = os.path.join(self.path, self.files[index])
        image = Image.open(full_path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        image = self.transform(np.array(image))
        return image
    
    def __len__(self):
        return len(self.files)    

class ImagesModule(pl.LightningDataModule):
    def __init__(self, data_folder: str, batch_size: int, augment_training: bool = True, val_split: float = 0.05):
        super().__init__()
        self.num_workers = 16
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.augment_training = augment_training
        self.val_split = val_split
        self.init_data()
        
    def init_data(self):
        files = filter_files(self.data_folder)
        images = []
        for file in files:
            full_path = os.path.join(self.data_folder, file)
            images_path = os.path.join(full_path, "images")
            image_names = os.listdir(images_path)
            for image_name in image_names:
                images.append(os.path.join(file, "images", image_name))
        random.shuffle(images)
        val_split = math.ceil(len(images) * self.val_split)
        self.files_train = images[val_split:]
        self.files_val = images[:val_split]
        
    def prepare_data(self):
        pass

    def setup(self, stage):
        pass
        
    def train_dataloader(self):
        transforms = train_transforms
        if not self.augment_training:
            transforms = test_transforms
        dataset = ImagesDataset(self.data_folder, self.files_train, transform = transforms)
        return DataLoader(dataset, num_workers = self.num_workers, batch_size = self.batch_size, shuffle=True)

    def val_dataloader(self):
        dataset = ImagesDataset(self.data_folder, self.files_val, transform = test_transforms)
        return DataLoader(dataset, num_workers = self.num_workers, batch_size = self.batch_size)

    def teardown(self, stage):
        # clean up after fit or test
        # called on every process in DDP
        pass

