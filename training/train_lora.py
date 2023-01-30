import argparse
import math
import torch
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
import torchmetrics
import os
from tqdm import tqdm
from imgaug import augmenters as iaa
import numpy as np
import random
from functools import partial
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from PIL import ImageOps
from pytorch_lightning.callbacks import LearningRateMonitor
import traceback
import sys
from get_extrema import get_extrema
from leap_sd import LM
from safetensors import safe_open

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--logging", type=str, default="tensorboard")
    parser.add_argument("--latent_dim_size", type=int, default=509248)
    parser.add_argument("--min_weight", type=int, default=None)
    parser.add_argument("--max_weight", type=int, default=None)
    parser.add_argument("--latent_dim_buffer_size", type=int, default=1024 * 4)
    parser.add_argument("--dropout_p", type=float, default=0.01)
    file_path = os.path.abspath(os.path.dirname(__file__))
    parser.add_argument("--dataset_path", type=str, default=os.path.join(file_path, "lora_dataset_creator/lora_dataset"))
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args(args)

def get_datamodule(path: str, batch_size: int, augment: bool):
    test_transforms = transforms.Compose(
        [
            iaa.Resize({"shorter-side": (128, 256), "longer-side": "keep-aspect-ratio"}).augment_image,
            iaa.CropToFixedSize(width=128, height=128).augment_image,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    if augment:
        train_transforms = transforms.Compose(
            [
                iaa.Resize({"shorter-side": (128, 256), "longer-side": "keep-aspect-ratio"}).augment_image,
                iaa.CropToFixedSize(width=128, height=128).augment_image,
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
    else:
        train_transforms = test_transforms

    class ImageWeightDataset(Dataset):
        def __init__(self, path, transform):
            self.path = path
            self.files = os.listdir(self.path)
            self.transform = transform
            self.num_images = 4

        def __getitem__(self, index):
            full_path = os.path.join(self.path, self.files[index])
            try:
                images_path = os.path.join(full_path, "images")
                image_names = os.listdir(images_path)
                random.shuffle(image_names)
                image_names = image_names[:random.randint(1, self.num_images)]
                image_names_len = len(image_names)
                if image_names_len < self.num_images:
                    for i in range(self.num_images - image_names_len):
                        image_names.append(image_names[i % image_names_len])

                images = None
                for image_name in image_names:
                    image = Image.open(os.path.join(images_path, image_name)).convert("RGB")
                    image = ImageOps.exif_transpose(image)
                    image = self.transform(np.array(image)).unsqueeze(0)
                    if images is None:
                        images = image
                    else:
                        images = torch.cat((images, image), 0)

                model_path = os.path.join(full_path, "models")
                with safe_open(os.path.join(model_path, "step_1000.safetensors"), framework="pt") as f:
                    tensor = None
                    keys = list(f.keys())
                    # Avoiding undefined behaviour: Making sure we always use keys in alphabethical order!
                    keys.sort()
                    for k in keys:
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

    class ImageWeights(pl.LightningDataModule):
        def __init__(self, data_folder: str, batch_size: int):
            super().__init__()
            self.num_workers = 16
            self.data_folder = data_folder
            self.batch_size = batch_size
            self.overfit = False
            self.num_samples = len(os.listdir(os.path.join(self.data_folder, "train")))
            if self.overfit:
                self.num_samples = 250
            
        def prepare_data(self):
            pass

        def setup(self, stage):
            pass
            
        def train_dataloader(self):
            dataset = ImageWeightDataset(os.path.join(self.data_folder, "train"), transform = train_transforms)
            if self.overfit:
                file_list = dataset.files[:1]
                print("Overfit! Using only:", file_list)
                dataset.files = file_list * 250
            return DataLoader(dataset, num_workers = self.num_workers, batch_size = self.batch_size, shuffle=True)

        def val_dataloader(self):
            return DataLoader(ImageWeightDataset(os.path.join(self.data_folder, "val"), transform = test_transforms), num_workers = self.num_workers, batch_size = self.batch_size)

        def test_dataloader(self):
            return DataLoader(ImageWeightDataset(os.path.join(self.data_folder, "test"), transform = test_transforms), num_workers = self.num_workers, batch_size = self.batch_size)

        def teardown(self, stage):
            # clean up after fit or test
            # called on every process in DDP
            pass
    
    dm = ImageWeights(path, batch_size = batch_size)
    
    return dm

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    pl.seed_everything(1)
    args = parse_args()
    
    # Add some dm attributes to args Namespace
    args.input_shape = (3, 128, 128)

    # compute total number of steps
    batch_size = args.batch_size * args.gpus if args.gpus > 0 else args.batch_size
    
    if args.max_weight is None or args.max_weight is None:
        print("Getting extrema")
        dm = get_datamodule(batch_size = batch_size, path = args.dataset_path, augment = False)
        min_weight, max_weight = get_extrema(dm.train_dataloader())
        print(f"Extrema of entire training set: {min_weight} <> {max_weight}")
        args.min_weight = min_weight
        args.max_weight = max_weight

    dm = get_datamodule(batch_size = batch_size, path = args.dataset_path, augment = True)    
    args.steps = dm.num_samples // batch_size * args.max_epochs
    
    # Init Lightning Module
    lm = LM(**vars(args))
    lm.train()

    # Init callbacks
    if args.logging != "none":
        lr_monitor = LearningRateMonitor(logging_interval='step')
        args.callbacks = [lr_monitor]
        if args.logging == "wandb":
            from pytorch_lightning.loggers import WandbLogger
            args.logger = WandbLogger(project="LEAP_Lora")
    else:
        args.checkpoint_callback = False
        args.logger = False
    
    # Set up Trainer
    trainer = pl.Trainer.from_argparse_args(args)
    
    # Train!
    trainer.fit(lm, dm)
