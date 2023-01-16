from .utils import linear_warmup_cosine_decay
import pytorch_lightning as pl
import torch
import torchvision
import random
from itertools import chain
from torch import nn, einsum
import torch.nn.functional as F

class LM(pl.LightningModule):
    def __init__(
        self,
        steps,
        input_shape,
        min_weight = 0,
        max_weight = 0,
        learning_rate=1e-4,
        weight_decay=0.0001,
        dropout_p=0.0,
        linear_warmup_ratio=0.01,
        latent_dim_size=1024,
        **_
    ):
        super().__init__()
        self.save_hyperparameters()
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.latent_dim_size = latent_dim_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps = steps
        self.linear_warmup_ratio = linear_warmup_ratio
        self.criterion = torch.nn.L1Loss()
        self.init_model(input_shape, dropout_p)
    
    def init_model(self, input_shape, dropout_p):
        feature_layers = [
            nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(p=dropout_p)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(p=dropout_p)
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(p=dropout_p)
            )
        ]
        self.features = nn.Sequential(*feature_layers)
        n_sizes = self._get_conv_output(input_shape)
        output_layers = [
            nn.Linear(n_sizes, self.latent_dim_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.latent_dim_size, self.latent_dim_size)
        ]
        self.output = nn.Sequential(*output_layers)
        self.forget_leveler = nn.Linear(n_sizes, 1)

    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self.features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # will be used during inference
    def forward(self, x):
        images_len = x.shape[1]
        xf = None
        for i in range(images_len):
            image_selection = x[:, i, ...]
            if xf is None:
                xf = self.features(image_selection)
            else:
                xf += self.features(image_selection)
        xf = xf / images_len
        xf = xf.view(xf.size(0), -1)
        xfo = self.forget_leveler(xf)
        xf[xf < xfo] = 0
        xf = self.output(xf)
        xf = self.denormalize_embed(xf)
        return xf

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)
        warmup_steps = int(self.linear_warmup_ratio * self.steps)
        scheduler = {
            "scheduler": linear_warmup_cosine_decay(optimizer, warmup_steps, self.steps),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def denormalize_embed(self, embed):
        embed = embed * (abs(self.min_weight) + self.max_weight)
        embed = embed - abs(self.min_weight)
        return embed

    def shot(self, batch, name, image_logging = False):
        image_grid, target = batch
        pred = self.forward(image_grid)
        loss = self.criterion(pred, target)
        self.log(f"{name}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shot(batch, "train", image_logging = True)

    def validation_step(self, batch, batch_idx):
        return self.shot(batch, "val")