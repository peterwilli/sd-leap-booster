from raw_images_datamodule import ImagesModule
from datamodule import ImageWeightsModule
import pytorch_lightning as pl
import torch
from leap_sd import Autoencoder
import argparse
import os
import torchvision
from pytorch_lightning.callbacks import LearningRateMonitor
from callbacks import GenerateCallback

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--linear_warmup_ratio", type=float, default=0.05)
    parser.add_argument("--base_channel_size", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=128)
    file_path = os.path.abspath(os.path.dirname(__file__))
    parser.add_argument("--dataset_path", type=str, default=os.path.join(file_path, "lora_dataset_creator/lora_dataset"))
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args(args)

def main():
    torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(1)
    
    args = parse_args()
    # dm = ImagesModule(args.dataset_path, args.batch_size, augment_training=False)
    dm = ImageWeightsModule(args.dataset_path, 10, augment_training=False)
    ae = Autoencoder(**vars(args))
    ae.train()

    lr_monitor = LearningRateMonitor(logging_interval='step')
    args.callbacks = [
        lr_monitor,
        GenerateCallback(dm.val_dataloader(), every_n_epochs=1)
    ]

    # Set up Trainer
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(ae, dm)

if __name__ == "__main__":
    main()