from torchvision import transforms
import torchvision
import pytorch_lightning as pl
import random
from imgaug import augmenters as iaa
from torch.utils.data import Dataset, DataLoader
from time import time
import numpy as np
from tqdm import tqdm
import os
import sys
import math
from PIL import Image
from PIL import ImageOps
import traceback
import torch
from safetensors.torch import safe_open as open_safetensors
from sklearn.decomposition import PCA

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
        iaa.Resize({"shorter-side": 32, "longer-side": 32}).augment_image,
        iaa.CropToFixedSize(width=32, height=32).augment_image,
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

train_transforms = transforms.Compose(
    [
        iaa.Resize({"shorter-side": (32, 32), "longer-side": (32, 32)}).augment_image,
        iaa.CropToFixedSize(width=32, height=32).augment_image,
        iaa.Sometimes(1, iaa.Sequential([
            iaa.flip.Fliplr(p=0.5),
            iaa.Sometimes(
                0.5,
                iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-20, 10),
                    shear=(-8, 8),
                    cval=(0, 255),
                    mode=['constant', 'edge', 'reflect']
                )
            ),
            iaa.GaussianBlur(sigma=(0.0, 1)),
            iaa.MultiplySaturation(mul=(0.5, 1.5)),
            iaa.MultiplyBrightness(mul=(0.65, 1.35)),
            iaa.LinearContrast((0.75, 1.5)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        ], random_order=True)).augment_image,
        np.copy,
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

@np.printoptions(suppress=True)
def test_pca(pca, x):
    t0 = time()
    z = pca.transform(x)
    print("transform done in %0.3fs" % (time() - t0))
    print(f"out = {z.shape}")
    t0 = time()
    x_hat = pca.inverse_transform(z)
    print("inverse_transform done in %0.3fs" % (time() - t0))
    print(f"loss = {abs(x_hat - x).mean()}")

class ImageWeightDataset(Dataset):
    def __init__(self, path, files, transform):
        self.path = path
        self.files = files
        self.transform = transform
        self.num_images = 4
        self.sorted_keys = None
        self.randomize = False
        self.get_pca = True

    def __getitem__(self, index):
        full_path = os.path.join(self.path, self.files[index])
        try:
            images_path = os.path.join(full_path, "images_generated")
            image_names = os.listdir(images_path)
            if self.randomize:
                random.shuffle(image_names)
                image_names = image_names[:random.randint(1, self.num_images)]
            else:
                image_names.sort()
                image_names = [image_names[0]]
            
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

            tensor = None
            cls_num = 0
            if self.get_pca:
                model_file_path = os.path.join(full_path, "models", "pca_embed.safetensors")
                with open_safetensors(model_file_path, framework="pt") as f:
                    tensor = f.get_tensor('pca_embed')
                    cls_num = f.get_tensor('cls_num')
            else:
                model_file_path = os.path.join(full_path, "models", "step_1000.safetensors")
                with open_safetensors(model_file_path, framework="pt") as f:
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
            # print(cls_num, self.files[index])
            return images, tensor, cls_num
        except:
            print(f"Error with {full_path}!")
            traceback.print_exception(*sys.exc_info())  
    
    def __len__(self):
        return len(self.files)    

def inspect_record(full_path) -> bool:
    images_path = os.path.join(full_path, "images_generated")
    if not os.path.exists(images_path):
        return False
    images_count = len(os.listdir(images_path))
    if images_count == 0:
        return False
    model_path = os.path.join(full_path, "models", "pca_embed.safetensors")
    if not os.path.exists(model_path):
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
        self.num_workers = 1
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.augment_training = augment_training
        self.val_split = val_split
        self.init_data()
        
    def init_data(self):
        files = filter_files(self.data_folder)
        # files = ["vol"]
        self.total_records = len(files)
        random.shuffle(files)
        # files = files[:50]
        val_split = math.ceil(len(files) * self.val_split)
        self.files_train = files[val_split:]
        self.files_val = files[:val_split]
        if len(self.files_train) == 0:
            self.files_train = self.files_val
            print("WARNING! Using val for training too, because the training set is empty!")
        
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