from .utils import linear_warmup_cosine_decay
from .model_components import EmbedNormalizer, EmbedDenormalizer, LeapAvgPool1D
from .model_components import LEAPBlock, LEAPBuffer
from .autoencoder import Encoder, Decoder
import pytorch_lightning as pl
import torch
import torchvision
import random
import math
from itertools import chain
from torch import nn, einsum
import torch.nn.functional as F
from hflayers import HopfieldLayer
import numpy as np

class LM(pl.LightningModule):
    def __init__(
        self,
        input_shape,
        # mapping,
        # extrema,
        encoder,
        num_heads: int,
        hidden_size: int,
        num_cnn_layers: int,
        optimizer_name: str,
        scheduler_name: str,
        mapping,
        total_records: int,
        pca,
        learning_rate=1e-4,
        weight_decay=0.0001,
        dropout_cnn=0.0,
        dropout_hopfield=0.0,
        hopfield_scaling=4.0,
        linear_warmup_ratio=0.01,
        avg_val_loss_history = 5,
        sgd_momentum = 0.9,
        reduce_lr_on_plateau_factor = 0.9,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=list(kwargs.keys()) + ["pca"])
        self.model = VisionTransformer(**model_kwargs)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.sgd_momentum = sgd_momentum
        self.reduce_lr_on_plateau_factor = reduce_lr_on_plateau_factor
        self.optimizer_name = optimizer_name
        self.mapping = mapping
        self.scheduler_name = scheduler_name
        self.pca = pca
        self.linear_warmup_ratio = linear_warmup_ratio
        self.encoder = encoder
        self.total_records = total_records
        self.criterion_embed = torch.nn.MSELoss()
        self.init_model(input_shape, num_cnn_layers, dropout_cnn, dropout_hopfield, hidden_size, num_heads, hopfield_scaling)
        # self.embed_normalizer = EmbedNormalizer(mapping = mapping, extrema = extrema)
        # self.embed_denormalizer = EmbedDenormalizer(mapping = mapping, extrema = extrema)
        self.avg_val_loss_history = avg_val_loss_history
        self.val_loss_history = []

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self.features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
    
    def init_feature_layers(self, num_cnn_layers, dropout_cnn):
        feature_layers = []
        last_channel = 3
        for i in range(num_cnn_layers):
            new_channel = 16 * (i + 1)
            features = [
                nn.Conv2d(last_channel, new_channel, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ]
            if i < num_cnn_layers - 1:
                features.append(nn.Dropout(p=dropout_cnn))
            feature_layers.append(nn.Sequential(*features))
            last_channel = new_channel
        return nn.Sequential(*feature_layers)
        
    def init_model(self, input_shape, num_cnn_layers, dropout_cnn, dropout_hopfield, hidden_size, num_heads, hopfield_scaling):
        # self.lookup = HopfieldLayer(
        #     input_size=128 * 4,
        #     output_size=100,
        #     hidden_size=hidden_size,
        #     num_heads=num_heads,
        #     quantity=self.total_records,
        #     scaling=hopfield_scaling,
        #     dropout=dropout_hopfield
        # )
        # self.features = self.init_feature_layers(num_cnn_layers, dropout_cnn)
        # self.feature_size = self._get_conv_output(input_shape)
        self.feature_size = 128
        print(f"feature_size: {self.feature_size}")
        num_dimensions = self.pca['pca'].n_components
        feature_translator = []
        total_layers = 10
        hidden_size = 2048
        feature_translator.append(nn.Linear(self.feature_size, hidden_size))
        for i in range(total_layers):
            feature_translator += [
                nn.Dropout(p=dropout_hopfield),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ]
        feature_translator += [
            nn.Linear(hidden_size, num_dimensions),
            nn.Sigmoid()
        ]
        print("feature_translator", feature_translator)
        self.feature_translator = nn.Sequential(*feature_translator)

    def post_process(self, flat_tensor):
        flat_tensor = flat_tensor.unsqueeze(0).numpy()
        flat_tensor = self.pca['scaler'].inverse_transform(flat_tensor)
        flat_tensor = self.pca['pca'].inverse_transform(flat_tensor)
        flat_tensor = torch.tensor(flat_tensor).squeeze(0)
        keys = list(self.mapping.keys())
        keys.sort()
        result = {}
        items_done = 0
        for k in keys:
            mapping_obj = self.mapping[k]
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

    def forward(self, x):
        grid_batch = None
        for i in range(x.shape[0]):
            grid = torchvision.utils.make_grid(x[i, ...], nrow=2, padding=0).unsqueeze(0)
            if grid_batch is None:
                grid_batch = grid
            else:
                grid_batch = torch.cat((grid_batch, grid), dim=0)

        result = self.model(grid_batch)
        return result

    def configure_optimizers(self):
        optimizer = None
        if self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.sgd_momentum, weight_decay=self.weight_decay)
        elif self.optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.weight_decay)
        steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.linear_warmup_ratio * steps)
        scheduler = None
        if self.scheduler_name == "linear_warmup_cosine_decay":
            scheduler = {
                "scheduler": linear_warmup_cosine_decay(optimizer, warmup_steps, steps),
                "interval": "step",
            }
        elif self.scheduler_name == "reduce_lr_on_plateau":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience = 5, factor = self.reduce_lr_on_plateau_factor),
                "interval": "epoch",
                "monitor": "avg_val_loss",
                "strict": True
            }
        # size_steps = math.floor(steps / 10)
        # scheduler = {
        #     "scheduler": torch.optim.lr_scheduler.CyclicLR(optimizer, cycle_momentum = self.optimizer_name == "SGD", base_lr=self.learning_rate * 0.1, max_lr=self.learning_rate, step_size_up=size_steps, step_size_down=size_steps),
        #     "interval": "step",
        # }
        # scheduler["scheduler"]._scale_fn_custom = scheduler["scheduler"]._scale_fn_ref()
        # scheduler["scheduler"]._scale_fn_ref = None
        return [optimizer], [scheduler]

    def shot(self, batch, name):
        image_grid, target = batch
        target = torch.tensor(self.pca['scaler'].transform(target.cpu().numpy()), dtype=torch.float32).to(target.device)
        embed_pred = self.forward(image_grid)
        print("embed_pred", embed_pred[0, ...])
        print("target", target[0, ...])
        loss_embed = self.criterion_embed(embed_pred, target)
        self.log(f"{name}_loss_embed", loss_embed)
        return loss_embed

    def training_step(self, batch, batch_idx):
        return self.shot(batch, "train")

    def validation_step(self, batch, batch_idx):
        val_loss = self.shot(batch, "val")

        self.val_loss_history.append(val_loss.item())
        if len(self.val_loss_history) > self.avg_val_loss_history:
            self.val_loss_history.pop(0)

        avg_val_loss = sum(self.val_loss_history) / len(self.val_loss_history)
        self.log("avg_val_loss", avg_val_loss, prog_bar=True)