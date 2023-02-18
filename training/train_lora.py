import argparse
import math
import torch
import pytorch_lightning as pl
import torchmetrics
import os
from time import time
import wandb
from tqdm import tqdm
import numpy as np
import random
from functools import partial
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import traceback
import pickle
import sys
from dataset_utils import get_extrema
from datamodule import ImageWeightsModule, FakeWeightsModule
from leap_sd import LM, Autoencoder
from leap_sd.model_components import EmbedNormalizer, EmbedDenormalizer
from callbacks import InputMonitor, OutputMonitor, GenerateFromLoraCallback, PCASaveCallback
import optuna

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--logging", type=str, default="tensorboard")
    parser.add_argument("--min_weight", type=int, default=None)
    parser.add_argument("--max_weight", type=int, default=None)
    parser.add_argument("--num_cnn_layers", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=5)
    parser.add_argument("--num_heads", type=int, default=5)
    parser.add_argument("--dropout_hopfield", type=float, default=0.5)
    parser.add_argument("--dropout_cnn", type=float, default=0.01)
    parser.add_argument("--hopfield_scaling", type=float, default=8.0)
    parser.add_argument("--optimizer_name", type=str, default="AdamW")
    parser.add_argument("--scheduler_name", type=str, default="linear_warmup_cosine_decay")
    parser.add_argument("--autoencoder_path", type=str, default="./autoencoder.ckpt")
    parser.add_argument("--sgd_momentum", type=float, default=0.99)
    parser.add_argument("--swa_lr", type=float, default=0.0)
    parser.add_argument("--swa_epoch_start", type=float, default=0.5)
    parser.add_argument("--gen_every_n_epochs", type=int, default=25)
    parser.add_argument("--annealing_strategy", type=str, default="cos")
    parser.add_argument("--reduce_lr_on_plateau_factor", type=float, default=0.90)
    parser.add_argument("--linear_warmup_ratio", type=float, default=0.05)
    parser.add_argument("--hyperparam_search", action="store_true")
    file_path = os.path.abspath(os.path.dirname(__file__))
    parser.add_argument("--dataset_path", type=str, default=os.path.join(file_path, "lora_dataset_creator/lora_dataset"))
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args(args)

def compress_mapping(mapping):
    result = {
    }
    keys = list(mapping.keys())
    keys.sort()
    def add_len(key, len):
        if not key in result:
            result[key] = {
                'len': 0
            }
        result[key]["len"] += len
    add_len("<s1>", 1024)
    for k in keys:
        obj = mapping[k]
        if k.startswith("text_encoder:"):
            add_len("text_encoder", obj["len"])
        if k.startswith("unet:"):
            add_len("unet", obj["len"])
    return result

@torch.no_grad()
def set_lookup_weights(hopfield, encoder, loader):
    # Z = None
    # for x, _ in loader:
    #     z = None
    #     for i in range(x.shape[1]):
    #         encoded = encoder(x[:, i, ...])
    #         if z is None:
    #             z = encoded
    #         else:
    #             z += encoded
    #     z = z / x.shape[1]
    #     if Z is None:
    #         Z = z
    #     else:
    #         Z = torch.cat((Z, z), dim=0)
    
    # Z = None
    # for x, _ in loader:
    #     z = encoder(x[:, 0, ...])
    #     if Z is None:
    #         Z = z
    #     else:
    #         Z = torch.cat((Z, z), dim=0)

    Z = None
    for x, _ in loader:
        z_inner = None
        for i in range(x.shape[1]):
            encoded = encoder(x[:, i, ...])
            if z_inner is None:
                z_inner = encoded
            else:
                z_inner = torch.cat((z_inner, encoded), dim=1)
        if Z is None:
            Z = z_inner
        else:
            Z = torch.cat((Z, z_inner), dim=0)
    
    Z = Z.unsqueeze(0)
    print("set_lookup_weights > X", Z.shape)
    hopfield.lookup_weights[:] = Z

def self_test(loader, mapping, extrema):
    print("Doing self-test...")
    normalizer = EmbedNormalizer(mapping = mapping, extrema = extrema)
    denormalizer = EmbedDenormalizer(mapping = mapping, extrema = extrema)

    for x, y in loader:
        y_normalized = normalizer(y)
        y_min = torch.min(y_normalized)
        y_max = torch.max(y_normalized)
        assert y_min >= 0 and y_max <= 1, "Not between 0 and 1!"
        y_denormalized = denormalizer(y_normalized)
        assert abs(y - y_denormalized).mean() < 0.01, "(De)Normalize NOT working!!"
        break
    print("All systems go!")

@torch.no_grad()
def init_extrema(args, dm):
    print("Getting extrema")
    extrema = get_extrema(dm.train_dataloader(), args.mapping)
    print(f"Extrema of entire training set: {extrema}")
    return extrema

def get_data(loader):
    X = None
    Y = None
    for x, y in tqdm(loader, desc='Loading all data'):
        if X is None:
            X = x
        X = torch.cat((X, x), dim=0)
        if Y is None:
            Y = y
        Y = torch.cat((Y, y), dim=0)
    return X, np.array(Y)

def train(args, do_self_test = True, project_name = "LEAP_Lora"):
    torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision('medium')

    pl.seed_everything(1)
    
    args.input_shape = (3, 128, 128)
    dm = ImageWeightsModule(args.dataset_path, 10, augment_training=True)
    # dm = FakeWeightsModule(10)
    # compute total number of steps
    batch_size = args.batch_size * args.gpus if args.gpus > 0 else args.batch_size
    args.total_records = dm.total_records
    mapping = {'<s1>': {'len': 1024, 'shape': [1024]}, 'text_encoder:0:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:0:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:10:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:10:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:11:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:11:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:12:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:12:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:13:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:13:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:14:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:14:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:15:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:15:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:16:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:16:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:17:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:17:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:18:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:18:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:19:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:19:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:1:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:1:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:20:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:20:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:21:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:21:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:22:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:22:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:23:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:23:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:24:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:24:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:25:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:25:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:26:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:26:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:27:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:27:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:28:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:28:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:29:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:29:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:2:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:2:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:30:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:30:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:31:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:31:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:32:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:32:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:33:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:33:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:34:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:34:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:35:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:35:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:36:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:36:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:37:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:37:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:38:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:38:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:39:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:39:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:3:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:3:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:40:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:40:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:41:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:41:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:42:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:42:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:43:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:43:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:44:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:44:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:45:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:45:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:46:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:46:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:47:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:47:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:48:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:48:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:49:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:49:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:4:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:4:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:50:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:50:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:51:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:51:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:52:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:52:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:53:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:53:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:54:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:54:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:55:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:55:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:56:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:56:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:57:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:57:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:58:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:58:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:59:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:59:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:5:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:5:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:60:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:60:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:61:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:61:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:62:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:62:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:63:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:63:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:64:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:64:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:65:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:65:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:66:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:66:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:67:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:67:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:68:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:68:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:69:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:69:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:6:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:6:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:70:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:70:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:71:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:71:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:72:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:72:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:73:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:73:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:74:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:74:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:75:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:75:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:76:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:76:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:77:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:77:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:78:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:78:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:79:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:79:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:7:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:7:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:80:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:80:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:81:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:81:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:82:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:82:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:83:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:83:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:84:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:84:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:85:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:85:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:86:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:86:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:87:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:87:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:88:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:88:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:89:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:89:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:8:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:8:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:90:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:90:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:91:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:91:up': {'len': 1024, 'shape': [1024, 1]}, 'text_encoder:9:down': {'len': 1024, 'shape': [1, 1024]}, 'text_encoder:9:up': {'len': 1024, 'shape': [1024, 1]}, 'unet:0:down': {'len': 320, 'shape': [1, 320]}, 'unet:0:up': {'len': 320, 'shape': [320, 1]}, 'unet:100:down': {'len': 640, 'shape': [1, 640]}, 'unet:100:up': {'len': 640, 'shape': [640, 1]}, 'unet:101:down': {'len': 640, 'shape': [1, 640]}, 'unet:101:up': {'len': 640, 'shape': [640, 1]}, 'unet:102:down': {'len': 640, 'shape': [1, 640]}, 'unet:102:up': {'len': 640, 'shape': [640, 1]}, 'unet:103:down': {'len': 640, 'shape': [1, 640]}, 'unet:103:up': {'len': 5120, 'shape': [5120, 1]}, 'unet:104:down': {'len': 640, 'shape': [1, 640]}, 'unet:104:up': {'len': 640, 'shape': [640, 1]}, 'unet:105:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:105:up': {'len': 640, 'shape': [640, 1]}, 'unet:106:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:106:up': {'len': 640, 'shape': [640, 1]}, 'unet:107:down': {'len': 640, 'shape': [1, 640]}, 'unet:107:up': {'len': 640, 'shape': [640, 1]}, 'unet:108:down': {'len': 320, 'shape': [1, 320]}, 'unet:108:up': {'len': 320, 'shape': [320, 1]}, 'unet:109:down': {'len': 320, 'shape': [1, 320]}, 'unet:109:up': {'len': 320, 'shape': [320, 1]}, 'unet:10:down': {'len': 320, 'shape': [1, 320]}, 'unet:10:up': {'len': 320, 'shape': [320, 1]}, 'unet:110:down': {'len': 320, 'shape': [1, 320]}, 'unet:110:up': {'len': 320, 'shape': [320, 1]}, 'unet:111:down': {'len': 320, 'shape': [1, 320]}, 'unet:111:up': {'len': 320, 'shape': [320, 1]}, 'unet:112:down': {'len': 320, 'shape': [1, 320]}, 'unet:112:up': {'len': 2560, 'shape': [2560, 1]}, 'unet:113:down': {'len': 320, 'shape': [1, 320]}, 'unet:113:up': {'len': 320, 'shape': [320, 1]}, 'unet:114:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:114:up': {'len': 320, 'shape': [320, 1]}, 'unet:115:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:115:up': {'len': 320, 'shape': [320, 1]}, 'unet:116:down': {'len': 320, 'shape': [1, 320]}, 'unet:116:up': {'len': 320, 'shape': [320, 1]}, 'unet:117:down': {'len': 320, 'shape': [1, 320]}, 'unet:117:up': {'len': 320, 'shape': [320, 1]}, 'unet:118:down': {'len': 320, 'shape': [1, 320]}, 'unet:118:up': {'len': 320, 'shape': [320, 1]}, 'unet:119:down': {'len': 320, 'shape': [1, 320]}, 'unet:119:up': {'len': 320, 'shape': [320, 1]}, 'unet:11:down': {'len': 320, 'shape': [1, 320]}, 'unet:11:up': {'len': 320, 'shape': [320, 1]}, 'unet:120:down': {'len': 320, 'shape': [1, 320]}, 'unet:120:up': {'len': 320, 'shape': [320, 1]}, 'unet:121:down': {'len': 320, 'shape': [1, 320]}, 'unet:121:up': {'len': 2560, 'shape': [2560, 1]}, 'unet:122:down': {'len': 320, 'shape': [1, 320]}, 'unet:122:up': {'len': 320, 'shape': [320, 1]}, 'unet:123:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:123:up': {'len': 320, 'shape': [320, 1]}, 'unet:124:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:124:up': {'len': 320, 'shape': [320, 1]}, 'unet:125:down': {'len': 320, 'shape': [1, 320]}, 'unet:125:up': {'len': 320, 'shape': [320, 1]}, 'unet:126:down': {'len': 320, 'shape': [1, 320]}, 'unet:126:up': {'len': 320, 'shape': [320, 1]}, 'unet:127:down': {'len': 320, 'shape': [1, 320]}, 'unet:127:up': {'len': 320, 'shape': [320, 1]}, 'unet:128:down': {'len': 320, 'shape': [1, 320]}, 'unet:128:up': {'len': 320, 'shape': [320, 1]}, 'unet:129:down': {'len': 320, 'shape': [1, 320]}, 'unet:129:up': {'len': 320, 'shape': [320, 1]}, 'unet:12:down': {'len': 320, 'shape': [1, 320]}, 'unet:12:up': {'len': 320, 'shape': [320, 1]}, 'unet:130:down': {'len': 320, 'shape': [1, 320]}, 'unet:130:up': {'len': 2560, 'shape': [2560, 1]}, 'unet:131:down': {'len': 320, 'shape': [1, 320]}, 'unet:131:up': {'len': 320, 'shape': [320, 1]}, 'unet:132:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:132:up': {'len': 320, 'shape': [320, 1]}, 'unet:133:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:133:up': {'len': 320, 'shape': [320, 1]}, 'unet:134:down': {'len': 320, 'shape': [1, 320]}, 'unet:134:up': {'len': 320, 'shape': [320, 1]}, 'unet:135:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:135:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:136:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:136:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:137:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:137:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:138:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:138:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:139:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:139:up': {'len': 10240, 'shape': [10240, 1]}, 'unet:13:down': {'len': 320, 'shape': [1, 320]}, 'unet:13:up': {'len': 2560, 'shape': [2560, 1]}, 'unet:140:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:140:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:141:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:141:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:142:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:142:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:143:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:143:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:14:down': {'len': 320, 'shape': [1, 320]}, 'unet:14:up': {'len': 320, 'shape': [320, 1]}, 'unet:15:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:15:up': {'len': 320, 'shape': [320, 1]}, 'unet:16:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:16:up': {'len': 320, 'shape': [320, 1]}, 'unet:17:down': {'len': 320, 'shape': [1, 320]}, 'unet:17:up': {'len': 320, 'shape': [320, 1]}, 'unet:18:down': {'len': 640, 'shape': [1, 640]}, 'unet:18:up': {'len': 640, 'shape': [640, 1]}, 'unet:19:down': {'len': 640, 'shape': [1, 640]}, 'unet:19:up': {'len': 640, 'shape': [640, 1]}, 'unet:1:down': {'len': 320, 'shape': [1, 320]}, 'unet:1:up': {'len': 320, 'shape': [320, 1]}, 'unet:20:down': {'len': 640, 'shape': [1, 640]}, 'unet:20:up': {'len': 640, 'shape': [640, 1]}, 'unet:21:down': {'len': 640, 'shape': [1, 640]}, 'unet:21:up': {'len': 640, 'shape': [640, 1]}, 'unet:22:down': {'len': 640, 'shape': [1, 640]}, 'unet:22:up': {'len': 5120, 'shape': [5120, 1]}, 'unet:23:down': {'len': 640, 'shape': [1, 640]}, 'unet:23:up': {'len': 640, 'shape': [640, 1]}, 'unet:24:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:24:up': {'len': 640, 'shape': [640, 1]}, 'unet:25:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:25:up': {'len': 640, 'shape': [640, 1]}, 'unet:26:down': {'len': 640, 'shape': [1, 640]}, 'unet:26:up': {'len': 640, 'shape': [640, 1]}, 'unet:27:down': {'len': 640, 'shape': [1, 640]}, 'unet:27:up': {'len': 640, 'shape': [640, 1]}, 'unet:28:down': {'len': 640, 'shape': [1, 640]}, 'unet:28:up': {'len': 640, 'shape': [640, 1]}, 'unet:29:down': {'len': 640, 'shape': [1, 640]}, 'unet:29:up': {'len': 640, 'shape': [640, 1]}, 'unet:2:down': {'len': 320, 'shape': [1, 320]}, 'unet:2:up': {'len': 320, 'shape': [320, 1]}, 'unet:30:down': {'len': 640, 'shape': [1, 640]}, 'unet:30:up': {'len': 640, 'shape': [640, 1]}, 'unet:31:down': {'len': 640, 'shape': [1, 640]}, 'unet:31:up': {'len': 5120, 'shape': [5120, 1]}, 'unet:32:down': {'len': 640, 'shape': [1, 640]}, 'unet:32:up': {'len': 640, 'shape': [640, 1]}, 'unet:33:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:33:up': {'len': 640, 'shape': [640, 1]}, 'unet:34:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:34:up': {'len': 640, 'shape': [640, 1]}, 'unet:35:down': {'len': 640, 'shape': [1, 640]}, 'unet:35:up': {'len': 640, 'shape': [640, 1]}, 'unet:36:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:36:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:37:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:37:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:38:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:38:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:39:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:39:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:3:down': {'len': 320, 'shape': [1, 320]}, 'unet:3:up': {'len': 320, 'shape': [320, 1]}, 'unet:40:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:40:up': {'len': 10240, 'shape': [10240, 1]}, 'unet:41:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:41:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:42:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:42:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:43:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:43:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:44:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:44:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:45:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:45:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:46:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:46:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:47:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:47:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:48:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:48:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:49:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:49:up': {'len': 10240, 'shape': [10240, 1]}, 'unet:4:down': {'len': 320, 'shape': [1, 320]}, 'unet:4:up': {'len': 2560, 'shape': [2560, 1]}, 'unet:50:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:50:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:51:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:51:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:52:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:52:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:53:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:53:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:54:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:54:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:55:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:55:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:56:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:56:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:57:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:57:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:58:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:58:up': {'len': 10240, 'shape': [10240, 1]}, 'unet:59:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:59:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:5:down': {'len': 320, 'shape': [1, 320]}, 'unet:5:up': {'len': 320, 'shape': [320, 1]}, 'unet:60:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:60:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:61:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:61:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:62:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:62:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:63:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:63:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:64:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:64:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:65:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:65:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:66:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:66:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:67:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:67:up': {'len': 10240, 'shape': [10240, 1]}, 'unet:68:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:68:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:69:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:69:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:6:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:6:up': {'len': 320, 'shape': [320, 1]}, 'unet:70:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:70:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:71:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:71:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:72:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:72:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:73:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:73:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:74:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:74:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:75:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:75:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:76:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:76:up': {'len': 10240, 'shape': [10240, 1]}, 'unet:77:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:77:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:78:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:78:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:79:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:79:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:7:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:7:up': {'len': 320, 'shape': [320, 1]}, 'unet:80:down': {'len': 1280, 'shape': [1, 1280]}, 'unet:80:up': {'len': 1280, 'shape': [1280, 1]}, 'unet:81:down': {'len': 640, 'shape': [1, 640]}, 'unet:81:up': {'len': 640, 'shape': [640, 1]}, 'unet:82:down': {'len': 640, 'shape': [1, 640]}, 'unet:82:up': {'len': 640, 'shape': [640, 1]}, 'unet:83:down': {'len': 640, 'shape': [1, 640]}, 'unet:83:up': {'len': 640, 'shape': [640, 1]}, 'unet:84:down': {'len': 640, 'shape': [1, 640]}, 'unet:84:up': {'len': 640, 'shape': [640, 1]}, 'unet:85:down': {'len': 640, 'shape': [1, 640]}, 'unet:85:up': {'len': 5120, 'shape': [5120, 1]}, 'unet:86:down': {'len': 640, 'shape': [1, 640]}, 'unet:86:up': {'len': 640, 'shape': [640, 1]}, 'unet:87:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:87:up': {'len': 640, 'shape': [640, 1]}, 'unet:88:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:88:up': {'len': 640, 'shape': [640, 1]}, 'unet:89:down': {'len': 640, 'shape': [1, 640]}, 'unet:89:up': {'len': 640, 'shape': [640, 1]}, 'unet:8:down': {'len': 320, 'shape': [1, 320]}, 'unet:8:up': {'len': 320, 'shape': [320, 1]}, 'unet:90:down': {'len': 640, 'shape': [1, 640]}, 'unet:90:up': {'len': 640, 'shape': [640, 1]}, 'unet:91:down': {'len': 640, 'shape': [1, 640]}, 'unet:91:up': {'len': 640, 'shape': [640, 1]}, 'unet:92:down': {'len': 640, 'shape': [1, 640]}, 'unet:92:up': {'len': 640, 'shape': [640, 1]}, 'unet:93:down': {'len': 640, 'shape': [1, 640]}, 'unet:93:up': {'len': 640, 'shape': [640, 1]}, 'unet:94:down': {'len': 640, 'shape': [1, 640]}, 'unet:94:up': {'len': 5120, 'shape': [5120, 1]}, 'unet:95:down': {'len': 640, 'shape': [1, 640]}, 'unet:95:up': {'len': 640, 'shape': [640, 1]}, 'unet:96:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:96:up': {'len': 640, 'shape': [640, 1]}, 'unet:97:down': {'len': 1024, 'shape': [1, 1024]}, 'unet:97:up': {'len': 640, 'shape': [640, 1]}, 'unet:98:down': {'len': 640, 'shape': [1, 640]}, 'unet:98:up': {'len': 640, 'shape': [640, 1]}, 'unet:99:down': {'len': 640, 'shape': [1, 640]}, 'unet:99:up': {'len': 640, 'shape': [640, 1]}, 'unet:9:down': {'len': 320, 'shape': [1, 320]}, 'unet:9:up': {'len': 320, 'shape': [320, 1]}}
    args.mapping = mapping
    
    with open('pca.bin', 'rb') as f:
        args.pca = pickle.load(f)

    ae = Autoencoder.load_from_checkpoint(args.autoencoder_path)
    ae.freeze()
    args.encoder = ae.encoder
    
    if do_self_test:
        self_test(dm.train_dataloader(), mapping, args.extrema)
        
    if args.swa_lr > 0:
        swa_calback = StochasticWeightAveraging(args.swa_lr, swa_epoch_start=args.swa_epoch_start, annealing_strategy=args.annealing_strategy)
        args.callbacks += [swa_calback]
        print("Using SWA LR Callback:", swa_calback)

    # Init Lightning Module
    lm = LM(**vars(args))
    lm.train()
    # set_lookup_weights(lm.lookup, args.encoder, all_data_loader)

    # Init callbacks
    if args.logging != "none":
        lr_monitor = LearningRateMonitor(logging_interval='step')
        args.callbacks += [lr_monitor]
        if args.logging == "wandb":
            from pytorch_lightning.loggers import WandbLogger
            args.logger = WandbLogger(project=project_name)
    else:
        args.checkpoint_callback = False
        args.logger = False
    
    # Set up Trainer
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(lm, dm)
    return trainer

def objective(trial: optuna.trial.Trial, args) -> float:
    stopper = EarlyStopping(monitor="avg_val_loss", mode="min", patience = 15)
    args.callbacks = [stopper]
    args.num_cnn_layers = trial.suggest_int("num_cnn_layers", 1, 5)
    args.num_heads = trial.suggest_int("num_heads", 1, 15)
    args.hidden_size = trial.suggest_int("hidden_size", 1, 15)
    args.dropout_cnn = trial.suggest_float("dropout_cnn", 0.0, 0.5)
    args.dropout_hopfield = trial.suggest_float("dropout_hopfield", 0.0, 0.5)
    args.hopfield_scaling = trial.suggest_float("hopfield_scaling", 0.0, 8.0)
    args.linear_warmup_ratio = trial.suggest_float("linear_warmup_ratio", 0.0, 0.5)
    args.weight_decay = trial.suggest_float("weight_decay", 0.0, 1e-3)
    args.optimizer_name = "AdamW" #trial.suggest_categorical("optimizer", ["SGD", "AdamW"])
    
    if trial.suggest_int("should_do_swa_lr", 0, 1) == 1:
        args.swa_lr = trial.suggest_float("swa_lr", 0.0, 1e-4)
        args.annealing_strategy = trial.suggest_categorical("annealing_strategy", ["cos", "linear"])
        args.swa_epoch_start = trial.suggest_float("swa_epoch_start", 0.0, 0.8)
    else:
        args.swa_lr = 0.0

    if args.optimizer_name == "SGD":
        args.sgd_momentum = trial.suggest_float("sgd_momentum", 0.0, 0.99)
    args.scheduler_name = trial.suggest_categorical("scheduler", ["linear_warmup_cosine_decay", "reduce_lr_on_plateau"])
    if args.scheduler_name == "reduce_lr_on_plateau":
        args.reduce_lr_on_plateau_factor = trial.suggest_float("reduce_lr_on_plateau_factor", 0.1, 0.99)
    args.learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3)
    trainer = train(args, do_self_test=False, project_name="LEAP_Lora_HyperparamOpt")
    wandb.finish()
    return trainer.callback_metrics["avg_val_loss"].item()

def hyperparam_search(args):
    study = optuna.create_study(
        storage="sqlite:///optuna.sqlite3",  # Specify the storage URL here.
        study_name="leap_lora_training",
        load_if_exists=True
    )
    func = lambda trial: objective(trial, args)
    study.optimize(func, n_trials=1000)
    print(study.best_params)

def main():
    args = parse_args()
    if args.hyperparam_search:
        print("Doing hyperparam search!")
        hyperparam_search(args)
    else:
        args.callbacks = [
            GenerateFromLoraCallback("training/test_images/vol", every_n_epochs=args.gen_every_n_epochs),
            PCASaveCallback()
        ]
        train(args, do_self_test=False)

if __name__ == "__main__":
    main()