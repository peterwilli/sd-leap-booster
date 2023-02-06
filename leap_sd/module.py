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
from hflayers import HopfieldLayer

class LM(pl.LightningModule):
    def __init__(
        self,
        steps,
        input_shape,
        mapping,
        extrema,
        total_data_records,
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
        self.total_data_records = total_data_records
        self.output_len = 0
        for key in mapping.keys():
            self.output_len += mapping[key]['len']
        self.latent_dim_size = latent_dim_size
        self.latent_dim_buffer_size = latent_dim_buffer_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps = steps
        self.linear_warmup_ratio = linear_warmup_ratio
        self.patch_size = 4
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

    def img_to_patch(self, x, patch_size, flatten_channels=True):
        """
        Inputs:
            x - Tensor representing the image of shape [B, C, H, W]
            patch_size - Number of pixels per dimension of the patches (integer)
            flatten_channels - If True, the patches will be returned in a flattened format
                            as a feature vector instead of a image grid.
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
        x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
        if flatten_channels:
            x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
        return x
        
    def init_model(self, input_shape, dropout_p):
        features_size = self._get_conv_output(input_shape)
        self.lookup = HopfieldLayer(
            input_size=features_size,
            output_size=509248,
            hidden_size=2,
            num_heads=2,
            quantity=self.total_data_records,
            scaling=3.0,
            dropout=dropout_p,
            lookup_weights_as_separated=True,
            lookup_targets_as_trainable=True,
            normalize_stored_pattern_affine=True,
            normalize_pattern_projection_affine=True
        )
        self.features_size = features_size
        print("features_size", features_size)
        self.init_leapblocks()

    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.img_to_patch(input, self.patch_size)
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

    # will be used during inference
    def forward(self, x):
        images_len = x.shape[1]
        result = None
        for i in range(images_len):
            image_selection = x[:, i, ...]
            xf = self.img_to_patch(image_selection, self.patch_size)
            xf = xf.view(x.shape[0], -1).unsqueeze(1)
            result = self.lookup(xf).squeeze(1)
            if result is None:
                result = result
            else:
                result += result
        result = result / images_len
        if not self.training:
            result = self.embed_denormalizer(result)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5, factor=0.95),
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