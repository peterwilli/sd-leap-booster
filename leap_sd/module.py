import pytorch_lightning as pl
import torch
import torchvision
import random
from itertools import chain
from diffusers import AutoencoderKL
from torch import nn, einsum
import torch.nn.functional as F

class LM(pl.LightningModule):
    def __init__(
        self,
        steps,
        input_shape,
        learning_rate=1e-4,
        weight_decay=0.0001,
        dropout_p=0.0,
        linear_warmup_ratio=0.01,
        **_,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim_size = 1024
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps = steps
        self.linear_warmup_ratio = linear_warmup_ratio
        self.criterion = torch.nn.MSELoss()
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
            nn.Linear(n_sizes, self.latent_dim_size * 1),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.latent_dim_size * 1, self.latent_dim_size)
        ]
        self.output = nn.Sequential(*output_layers)

    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self.features(input)
        print("output_feat", output_feat.shape)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        print("Conv length:", n_size)
        return n_size
        
    # will be used during inference
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # x = torch.cat((x, current_grad), dim = 1)
        x = self.output(x)
        return x

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