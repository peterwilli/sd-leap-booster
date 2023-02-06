from .utils import linear_warmup_cosine_decay
from .model_components import EmbedNormalizer, EmbedDenormalizer
from .model_components import LEAPBlock, LEAPBuffer
import pytorch_lightning as pl
import torch
import torchvision
import random
import math
from itertools import chain
from torch import nn, einsum
import torch.nn.functional as F

class LM(pl.LightningModule):
    def __init__(
        self,
        steps,
        input_shape,
        mapping,
        extrema,
        compress = 8,
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
        self.compress = compress
        self.mapping = mapping
        self.output_len = 0
        for key in self.mapping.keys():
            self.output_len += self.mapping[key]['len']
        self.extrema = extrema
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

        self.embed_normalizer = EmbedNormalizer(mapping = mapping, extrema = extrema)
        self.embed_denormalizer = EmbedDenormalizer(mapping = mapping, extrema = extrema)

    def init_leapblocks(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _create_output_layer(self, output_size: int, dropout_p: int):
        output_layers = [
            LEAPBuffer(1024, output_size, 1024, 10)
        ]
        return nn.Sequential(*output_layers)
    
    def init_inn(self, dropout_p):
        keys = list(self.mapping.keys())
        keys.sort()
        for key in keys:
            inn_name = f"inn_{key}"
            mapping = self.mapping[key]
            size = mapping['len']
            print(f"Setting {inn_name} with len {size}")
            output_layer = self._create_output_layer(size, dropout_p)
            setattr(self, inn_name, output_layer)
    
    def init_model(self, input_shape, dropout_p):
        self.leap_block_1 = LEAPBlock(act_fn=None, subsample_count=1, c_out=64, stride=8)
        self.leap_block_2 = LEAPBlock(act_fn=None, subsample_count=4, c_out=64, stride=4)
        self.leap_block_3 = LEAPBlock(act_fn=None, subsample_count=8, c_out=64, stride=2)
        features_size = self._get_conv_output(input_shape)
        self.features_down = nn.Sequential(
            nn.Linear(features_size, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 1024)
        )
        self.features_size = features_size
        print("features_size", features_size)
        self.init_inn(dropout_p)
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
        xfd = self.features_down(xf)
        keys = list(self.mapping.keys())
        keys.sort()
        result = torch.zeros(x.shape[0], self.output_len, device=x.device)
        len_done = 0
        for key in keys:
            inn_name = f"inn_{key}"
            inn_model = getattr(self, inn_name)
            inn_output = inn_model(xfd)
            result[:, len_done:len_done + inn_output.shape[1]] = inn_output
            len_done += inn_output.shape[1]
        if not self.training:
            result = self.embed_denormalizer(result)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5),
            "monitor": "train_loss",
            "interval": "epoch"
        }
        return [optimizer], [scheduler]

    def shot(self, batch, name, image_logging = False):
        image_grid, target = batch
        target = self.embed_normalizer(target)
        pred = self.forward(image_grid)
        loss = self.criterion(pred, target)
        self.log(f"{name}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shot(batch, "train", image_logging = True)

    def validation_step(self, batch, batch_idx):
        return self.shot(batch, "val")