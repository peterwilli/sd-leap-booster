from .utils import linear_warmup_cosine_decay
import pytorch_lightning as pl
import torch
import torchvision
import random
import math
from itertools import chain
from torch import nn, einsum
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, c_in, c_red: dict, c_out: dict, act_fn):
        """
        Inputs:
            c_in - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
            c_out - Dictionary with keys "1x1", "3x3", "5x5", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1), nn.BatchNorm2d(c_out["1x1"]), act_fn()
        )

        # 3x3 convolution branch
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
            nn.BatchNorm2d(c_red["3x3"]),
            act_fn(),
            nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out["3x3"]),
            act_fn(),
        )

        # 5x5 convolution branch
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
            nn.BatchNorm2d(c_red["5x5"]),
            act_fn(),
            nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out["5x5"]),
            act_fn(),
        )

        # Max-pool branch
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(c_in, c_out["max"], kernel_size=1),
            nn.BatchNorm2d(c_out["max"]),
            act_fn(),
        )

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
        return x_out

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
        self.criterion = torch.nn.L1Loss()
        self.resnet_act_fn = nn.LeakyReLU
        self.init_model(input_shape, dropout_p)
    
    def init_model(self, input_shape, dropout_p):
        self.input_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), self.resnet_act_fn()
        )
        features_layer = []
        last_c_chan = None
        for i in range(5):
            def sieve(obj):
                for k in obj.keys():
                    obj[k] = math.ceil(obj[k] / max(i, 1))
                return obj
            def total_channels(obj):
                result = 0
                for k in obj.keys():
                    result += obj[k]
                return result
            c_out = sieve({"1x1": 16, "3x3": 32, "5x5": 8, "max": 8})
            c_chan = total_channels(c_out)
            if last_c_chan is None:
                last_c_chan = c_chan
            features_layer.append(InceptionBlock(
                last_c_chan,
                c_red={"3x3": 32, "5x5": 16},
                c_out=c_out,
                act_fn=self.resnet_act_fn 
            ))
            features_layer.append(nn.MaxPool2d(3, stride=2, padding=1))
            last_c_chan = c_chan
        self.features = nn.Sequential(*features_layer)
        features_size = self._get_conv_output(input_shape)
        self.features_size = features_size
        output_layers = [
            nn.Linear(features_size, self.latent_dim_buffer_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.latent_dim_buffer_size, self.latent_dim_buffer_size)
        ]
        self.output = nn.Sequential(*output_layers)
        self.forget_leveler = nn.Sequential(
            nn.MaxPool1d(2, stride=2, ceil_mode=True),
            nn.Linear(math.ceil(features_size / 2), 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 1),
        )

    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        input = self.input_net(input)
        output_feat = self.features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

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
        linspace = torch.linspace(0, 2 * math.pi, self.latent_dim_size, device=self.device)
        linspace = LM._zero_pad(linspace, self.features_size * amount_to_unroll)
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
            input = self.input_net(image_selection)
            if xf is None:
                xf = self.features(input)
                print("xf", xf.shape)
            else:
                xf += self.features(input)
        xf = xf / images_len
        xf = xf.view(xf.size(0), -1)
        print("xf", xf.shape)
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