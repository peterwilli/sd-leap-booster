import os

import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
from .model_components import EmbedNormalizer, EmbedDenormalizer
from .utils import linear_warmup_cosine_decay
from hflayers import HopfieldLayer

class Encoder(nn.Module):
    def __init__(self, size_in: int, size_out: int, hidden_size: int, num_heads: int, quantity: int, scaling: float, dropout: float):
        super().__init__()
        self.encoder = HopfieldLayer(
            input_size=size_in,
            output_size=size_out,
            hidden_size=hidden_size,
            num_heads=num_heads,
            quantity=quantity,
            scaling=scaling,
            dropout=dropout,
            pattern_projection_as_connected = True
        )

    def forward(self, x):
        x_hat = self.encoder(x)
        return x_hat

class Decoder(nn.Module):
    def __init__(self, size_in: int, size_out: int, hidden_size: int, num_heads: int, quantity: int, scaling: float, dropout: float):
        super().__init__()
        self.decoder = HopfieldLayer(
            input_size=size_in,
            output_size=size_out,
            hidden_size=hidden_size,
            num_heads=num_heads,
            quantity=quantity,
            scaling=scaling,
            dropout=dropout,
            pattern_projection_as_connected = True
        )
        self.output = nn.Tanh()

    def forward(self, x):
        x_hat = self.decoder(x)
        # x_hat = self.output(x_hat)
        return x_hat

class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        size_in: int, 
        latent_dim: int, 
        hidden_size: int,
        num_heads: int,
        quantity: int, 
        dropout: float,
        scaling: float,
        linear_warmup_ratio: float,
        mapping,
        extrema,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=kwargs.keys())
        self.linear_warmup_ratio = linear_warmup_ratio

        self.encoder = Encoder(size_in, latent_dim, hidden_size, num_heads, quantity, scaling, dropout)
        self.decoder = Decoder(latent_dim, size_in, hidden_size * 8, num_heads, quantity, scaling, dropout)

        self.embed_normalizer = EmbedNormalizer(mapping = mapping, extrema = extrema)
        self.embed_denormalizer = EmbedDenormalizer(mapping = mapping, extrema = extrema)

        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, size_in)

    def forward(self, x, log = False):
        """The forward function takes in an image and returns the reconstructed image."""
        x = x.unsqueeze(1)
        z = self.encoder(x)
        if log:
            print("z", z)
        x_hat = self.decoder(z)
        x_hat = x_hat.squeeze(1)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
        _, x = batch
        x = self.embed_normalizer(x)
        x_hat = self.forward(x)
        loss = F.l1_loss(x, x_hat, reduction='none')
        loss = loss.sum(dim=0).mean(dim=0)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.linear_warmup_ratio * steps)
        scheduler = {
            "scheduler": linear_warmup_cosine_decay(optimizer, warmup_steps, steps),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def _diff_log(self, batch):
        _, x = batch
        x_normalized = self.embed_normalizer(x)
        x_hat = self.forward(x)
        x_hat = self.embed_denormalizer(x_hat)
        print(f'_diff_log (predicted) ({x.shape}):')
        print(x_hat[0])
        print('vs (original):')
        print(x[0])

        x = torch.zeros_like(x).uniform_(0, 1)
        x_hat = self.forward(x, log = True)
        x_hat = self.embed_denormalizer(x_hat)
        print(f'noise (predicted) ({x.shape}):')
        print(x_hat[0])

    def validation_step(self, batch, batch_idx):
        self._diff_log(batch)
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)