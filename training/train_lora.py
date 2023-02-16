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
import sys
from dataset_utils import get_extrema
from datamodule import ImageWeightsModule, FakeWeightsModule
from leap_sd import LM, Autoencoder
from leap_sd.model_components import EmbedNormalizer, EmbedDenormalizer
from callbacks import InputMonitor, OutputMonitor, GenerateFromLoraCallback
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
    
    args.input_shape = (3, 32, 32)
    dm = ImageWeightsModule(args.dataset_path, 10, augment_training=False)
    # dm = FakeWeightsModule(10)
    # compute total number of steps
    batch_size = args.batch_size * args.gpus if args.gpus > 0 else args.batch_size
    #all_data_loader = ImageWeightsModule(args.dataset_path, 1, augment_training=False, val_split=0).train_dataloader()
    args.total_records = 336
    args.pca = dm.pca
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
            GenerateFromLoraCallback("training/test_images/vol", every_n_epochs=args.gen_every_n_epochs)
        ]
        train(args, do_self_test=False)

if __name__ == "__main__":
    main()