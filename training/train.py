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

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--logging", type=str, default="tensorboard")
    parser.add_argument("--latent_dim_size", type=int, default=1024)
    parser.add_argument("--dropout_p", type=float, default=0.01)
    file_path = os.path.abspath(os.path.dirname(__file__))
    parser.add_argument("--dataset_path", type=str, default=os.path.join(file_path, "dataset_creator/sd_extracted"))
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args(args)

def get_datamodule(path: str, batch_size: int):
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
    test_transforms = transforms.Compose(
        [
            iaa.Resize({"shorter-side": (128, 256), "longer-side": "keep-aspect-ratio"}).augment_image,
            iaa.CropToFixedSize(width=128, height=128).augment_image,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    class ImageWeightDataset(Dataset):
        def __init__(self, path, transform):
            self.path = path
            self.files = os.listdir(self.path)
            self.transform = transform
            self.num_images = 4

        def __getitem__(self, index):
            full_path = os.path.join(self.path, self.files[index])
            try:
                images_path = os.path.join(full_path, "concept_images")
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

                loaded_learned_embeds = torch.load(os.path.join(full_path, "learned_embeds.bin"), map_location="cpu")
                embed_model = loaded_learned_embeds[list(loaded_learned_embeds.keys())[0]].detach()
                embed_model = embed_model.to(torch.float32)
                return images, embed_model
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
    args.image_size = 128
    args.patch_size = 32
    args.input_shape = (3, 128, 128)

    # compute total number of steps
    batch_size = args.batch_size * args.gpus if args.gpus > 0 else args.batch_size
    min_weight, max_weight = get_extrema(os.path.join(args.dataset_path, "train"))
    print(f"Extrema of entire training set: {min_weight} <> {max_weight}")
    args.min_weight = min_weight
    args.max_weight = max_weight

    dm = get_datamodule(batch_size = batch_size, path = args.dataset_path)
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
            args.logger = WandbLogger(project="LEAP")
    else:
        args.checkpoint_callback = False
        args.logger = False
    
    # Set up Trainer
    trainer = pl.Trainer.from_argparse_args(args)
    
    # Train!
    trainer.fit(lm, dm)
