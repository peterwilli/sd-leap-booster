from .utils import linear_warmup_cosine_decay
import pytorch_lightning as pl
import torch
import torchvision
import random
import math
from itertools import chain
from torch import nn, einsum
import torch.nn.functional as F

class LEAPBlock(nn.Module):
    def __init__(self, act_fn, subsample_count: int, c_out: int, stride: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.AvgPool2d(subsample_count),
            nn.Conv2d(
                3, c_out, kernel_size=3, padding=1, stride=stride
            ),
            nn.Flatten()
        )
        self.act_fn = act_fn()

    def forward(self, image):
        z = self.net(image)
        out = self.act_fn(z)
        return out

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
        latent_dim_buffer_size = 1024,
        n_latent_dim_layers = 5,
        **_
    ):
        super().__init__()
        self.save_hyperparameters()
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.latent_dim_size = latent_dim_size
        self.latent_dim_buffer_size = latent_dim_buffer_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps = steps
        self.linear_warmup_ratio = linear_warmup_ratio
        self.n_latent_dim_layers = n_latent_dim_layers
        self.criterion = torch.nn.L1Loss()
        self.resnet_act_fn = nn.LeakyReLU
        self.init_model(input_shape, dropout_p)

    def init_leapblocks(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def init_model(self, input_shape, dropout_p):
        self.leap_block_1 = LEAPBlock(self.resnet_act_fn, 1, 64, 8)
        self.leap_block_2 = LEAPBlock(self.resnet_act_fn, 4, 64, 4)
        self.leap_block_3 = LEAPBlock(self.resnet_act_fn, 8, 64, 1)
        features_size = self._get_conv_output(input_shape)
        self.features_size = features_size
        print("features_size", features_size)
        output_layers = [
        ]
        for i in range(self.n_latent_dim_layers):
            if i == 0:
                output_layers += [
                    nn.Linear(features_size, self.latent_dim_buffer_size)
                ]
            else:
                output_layers += [
                    nn.Linear(self.latent_dim_buffer_size, self.latent_dim_buffer_size)
                ]
            if i < (self.n_latent_dim_layers - 1):
                output_layers += [
                    nn.LeakyReLU(),
                    nn.Dropout(p=dropout_p)
                ]
        self.output = nn.Sequential(*output_layers)
        self.forget_leveler = nn.Sequential(
            nn.Linear(features_size, 10),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(10, 5),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(5, 1),
        )
        self.init_leapblocks()

    def features(self, input):
        return torch.cat((self.leap_block_1(input), self.leap_block_2(input), self.leap_block_3(input)), dim=1)

    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self.features(input)
        return output_feat.shape[1]

    @staticmethod
    def unmap_flat_tensor(flat_tensor, mapping):
        keys = list(mapping.keys())
        keys.sort()
        result = {}
        items_done = 0
        for k in keys:
            mapping_obj = mapping[k]
            flat_slice = flat_tensor[items_done:items_done + mapping_obj['len']]
            result[k] = flat_slice.view(mapping_obj['shape'])
            items_done += mapping_obj['len']
        return result
        
    @staticmethod
    def map_tensors_flat(f):
        keys = list(f.keys())
        keys.sort()
        mapping = {}
        for k in keys:
            tensor = f.get_tensor(k)
            mapping[k] = {
                'len': len(tensor.flatten()),
                'shape': list(tensor.shape)
            }
        return mapping

    @staticmethod
    def _zero_pad(x, max_len: int):
        if len(x) < max_len:
            to_add = torch.zeros(max_len - len(x), device=x.device, dtype=x.dtype)
            x = torch.cat((x, to_add))
        return x

    def _unroll_buffer(self, x):
        result = None
        amount_to_unroll = math.ceil(self.latent_dim_size / self.latent_dim_buffer_size)
        linspace = torch.linspace(0, 2 * math.pi, self.features_size * amount_to_unroll, device=self.device)
        for idx in range(amount_to_unroll):
            positional_encoding = torch.sin(linspace[self.features_size * idx:self.features_size * (idx + 1)])
            positional_encoding += abs(torch.min(positional_encoding))
            positional_encoding /= torch.max(positional_encoding)
            positional_encoding = positional_encoding.expand(x.shape[0], -1)
            chunk = self.output(x + positional_encoding)
            if result is None: 
                result = chunk
            else:
                result = torch.cat((result, chunk), dim=1)
        return result[:, :self.latent_dim_size]

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
        xfo = self.forget_leveler(xf)
        xf[xf < xfo] = 0
        xf = self._unroll_buffer(xf)
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