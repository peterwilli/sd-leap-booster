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
from safetensors import safe_open

class FakeDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        return torch.zeros(4, 3, 128, 128).uniform_(0,1), torch.zeros(509248).uniform_(0,1)
    
    def __len__(self):
        return 100

class FakeWeightsModule(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 16
        self.data_train = FakeDataset()
        self.data_val = FakeDataset()
        
    def prepare_data(self):
        pass

    def setup(self, stage):
        pass
        
    def train_dataloader(self):
        return DataLoader(self.data_train, num_workers = self.num_workers, batch_size = self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, num_workers = self.num_workers, batch_size = self.batch_size)

    def teardown(self, stage):
        # clean up after fit or test
        # called on every process in DDP
        pass

test_transforms = transforms.Compose(
    [
        iaa.Resize({"shorter-side": (32, 64), "longer-side": "keep-aspect-ratio"}).augment_image,
        iaa.CropToFixedSize(width=32, height=32).augment_image,
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

train_transforms = transforms.Compose(
    [
        iaa.Resize({"shorter-side": (32, 64), "longer-side": "keep-aspect-ratio"}).augment_image,
        iaa.CropToFixedSize(width=32, height=32).augment_image,
        iaa.Sometimes(0.8, iaa.Sequential([
            iaa.flip.Fliplr(p=0.5),
            iaa.flip.Flipud(p=0.5),
            iaa.Sometimes(
                0.5,
                iaa.Sequential([
                    iaa.ShearX((-20, 20)),
                    iaa.ShearY((-20, 20))
                ])
            ),
            iaa.GaussianBlur(sigma=(0.0, 0.05)),
            iaa.MultiplyBrightness(mul=(0.65, 1.35)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        ], random_order=True)).augment_image,
        np.copy,
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

class ImageWeightDataset(Dataset):
    def __init__(self, path, files, transform):
        self.path = path
        self.files = files
        self.transform = transform
        self.num_images = 4
        self.sorted_keys = None

    def __getitem__(self, index):
        full_path = os.path.join(self.path, self.files[index])
        try:
            images_path = os.path.join(full_path, "images")
            image_names = os.listdir(images_path)
            # random.shuffle(image_names)
            image_names.sort()
            # image_names = image_names[:random.randint(1, self.num_images)]
            image_names = image_names[:self.num_images]
            images = None
            for image_name in image_names:
                image = Image.open(os.path.join(images_path, image_name)).convert("RGB")
                image = ImageOps.exif_transpose(image)
                image = self.transform(np.array(image)).unsqueeze(0)
                if images is None:
                    images = image
                else:
                    images = torch.cat((images, image), 0)
                
            # Pad with images to match full space
            current_img_len = images.shape[0]
            if current_img_len < self.num_images:
                for i in range(self.num_images - current_img_len):
                    image = images[i % current_img_len, ...].unsqueeze(0)
                    images = torch.cat((images, image), 0)

            model_path = os.path.join(full_path, "models")
            with safe_open(os.path.join(model_path, "step_1000.safetensors"), framework="pt") as f:
                tensor = None
                if self.sorted_keys is None:
                    keys = list(f.keys())
                    # Avoiding undefined behaviour: Making sure we always use keys in alphabethical order!
                    keys.sort()
                    self.sorted_keys = keys
                for k in self.sorted_keys:
                    if tensor is None:
                        tensor = f.get_tensor(k).flatten()
                    else:
                        tensor = torch.cat((tensor, f.get_tensor(k).flatten()), 0)
            return images, tensor
        except:
            print(f"Error with {full_path}!")
            traceback.print_exception(*sys.exc_info())  
    
    def __len__(self):
        return len(self.files)    

def inspect_record(full_path) -> bool:
    images_path = os.path.join(full_path, "images")
    if not os.path.exists(images_path):
        return False
    images_count = len(os.listdir(images_path))
    if images_count == 0:
        return False
    model_path = os.path.join(full_path, "models", "step_1000.safetensors")
    if not os.path.exists(model_path):
        return False
    return True

def filter_files(path):
    files = os.listdir(path)
    result = []
    for file in files:
        full_path = os.path.join(path, file)
        if inspect_record(full_path):
            result.append(file)
    print(f"filter_files: {len(result)} approved - {len(files) - len(result)} rejected")
    return result

class ImageWeightsModule(pl.LightningDataModule):
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
        random.shuffle(files)
        val_split = math.ceil(len(files) * self.val_split)
        self.files_train = files[val_split:]
        self.files_val = files[:val_split]
        
    def prepare_data(self):
        pass

    def setup(self, stage):
        pass
        
    def train_dataloader(self):
        transforms = train_transforms
        if not self.augment_training:
            transforms = test_transforms
        dataset = ImageWeightDataset(self.data_folder, self.files_train, transform = transforms)
        return DataLoader(dataset, num_workers = self.num_workers, batch_size = self.batch_size, shuffle=True)

    def val_dataloader(self):
        dataset = ImageWeightDataset(self.data_folder, self.files_val, transform = test_transforms)
        return DataLoader(dataset, num_workers = self.num_workers, batch_size = self.batch_size)

    def teardown(self, stage):
        # clean up after fit or test
        # called on every process in DDP
        pass

