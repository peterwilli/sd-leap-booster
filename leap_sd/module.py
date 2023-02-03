from .utils import linear_warmup_cosine_decay
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
        bank_fn,
        learning_rate=1e-4,
        weight_decay=0.0001,
        dropout_p=0.0,
        linear_warmup_ratio=0.01,
        **_
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps = steps
        self.linear_warmup_ratio = linear_warmup_ratio
        self.bank_fn = bank_fn
        self.criterion = torch.nn.L1Loss()
        self.init_model(input_shape, dropout_p)

    def _create_output_layer(self, output_size: int, dropout_p: int):
        output_layers = [
            LEAPBuffer(1024, output_size, 1024, 10)
        ]
        return nn.Sequential(*output_layers)
    
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
        features_size = self._get_conv_output(input_shape)
        self.net_in = nn.Sequential(
            nn.Linear(features_size, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 4),
            # nn.ReLU()
        )
        self.features_size = features_size
        print("features_size", features_size)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        '''Takes in a module and initializes all linear layers with weight
           values taken from a normal distribution.'''

        if isinstance(module, nn.Linear):
            y = module.in_features
            module.weight.data.normal_(0.0, 1 / np.sqrt(1024))
            module.bias.data.fill_(0)

    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

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
        curve_params = self.net_in(xf)
        if self.training:
            to_add = 1 * self.trainer.optimizers[0].param_groups[0]['lr']
            noise = torch.zeros_like(curve_params).uniform_(to_add * -1, to_add)
            curve_params = curve_params + noise
        bank_data = self.bank_fn(curve_params.cpu()).to(x.device)
        return bank_data

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5, factor = 0.9),
            "monitor": "train_loss",
            "interval": "epoch"
        }
        return [optimizer], [scheduler]

    def denormalize_embed(self, embed):
        keys = list(self.mapping.keys())
        keys.sort()
        len_done = 0
        for key in keys:
            obj = self.mapping[key]
            obj_extrema = self.extrema[key]
            mapping_len = obj['len']
            model_slice = embed[:, len_done:len_done + mapping_len]
            max_weight = obj_extrema['max']
            min_weight = obj_extrema['min']
            model_slice *= (abs(min_weight) + abs(max_weight))
            model_slice -= abs(min_weight)
            len_done += mapping_len
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