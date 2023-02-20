import argparse
import os
from leap_sd import LM, Autoencoder
from tqdm import tqdm
import torch
from safetensors.torch import save_file as save_safetensors
from safetensors.torch import safe_open as open_safetensors
import numpy as np
from time import time
import math
import random
import pytorch_lightning as pl
from sklearn.decomposition import PCA
import pickle
from sklearn.preprocessing import MinMaxScaler

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    file_path = os.path.abspath(os.path.dirname(__file__))
    parser.add_argument("--pca_output_path", type=str, default=os.path.join(file_path, "pca.bin"))
    parser.add_argument("--dataset_path", type=str, default=os.path.join(file_path, "lora_dataset_creator/lora_dataset"))
    parser.add_argument("--n_components", type=int, default=200)
    parser.add_argument("--val_split", type=float, default=0.05)
    return parser.parse_args(args)

@np.printoptions(suppress=True)
def test_pca(pca, x):
    t0 = time()
    z = pca.transform(x)
    print("transform done in %0.3fs" % (time() - t0))
    print(f"out = {z.shape}")
    t0 = time()
    x_hat = pca.inverse_transform(z)
    print("inverse_transform done in %0.3fs" % (time() - t0))
    print(f"loss = {abs(x_hat - x).mean()}")

def init_pca(path, pca_output_path, n_components, val_split):
    model_files = os.listdir(path)
    random.shuffle(model_files)
    X = None
    sorted_keys = None
    for model_file in tqdm(model_files, desc="Loading all data for PCA"):
        model_path = os.path.join(path, model_file, "models")
        model_file_path = os.path.join(model_path, "step_1000.safetensors")
        if os.path.exists(model_file_path):
            with open_safetensors(model_file_path, framework="pt") as f:
                tensor = None
                if sorted_keys is None:
                    keys = list(f.keys())
                    # Avoiding undefined behaviour: Making sure we always use keys in alphabethical order!
                    keys.sort()
                    sorted_keys = keys
                for k in sorted_keys:
                    if tensor is None:
                        tensor = f.get_tensor(k).flatten()
                    else:
                        tensor = torch.cat((tensor, f.get_tensor(k).flatten()), 0)
                tensor = tensor.unsqueeze(0)
                if X is None:
                    X = tensor
                else:
                    X = torch.cat((X, tensor), dim=0)

    val_split = math.ceil(X.shape[0] * val_split)
    X_val = X[:val_split, :].numpy()
    X_train = X[val_split:, :].numpy()
    t0 = time()
    print(f"in = {X_train.shape}")
    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True, random_state=69).fit(X_train)
    print("done in %0.3fs" % (time() - t0))
    print("Train loss:")
    test_pca(pca, X_train)
    print("Val loss:")
    test_pca(pca, X_val)

    X_transformed = pca.transform(X)
    
    scaler = MinMaxScaler()
    scaler.fit(X_transformed)

    with open(pca_output_path, 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'pca': pca
        }, f)

    X_transformed = torch.tensor(X_transformed)
    for i in tqdm(range(X_transformed.shape[0]), desc="Saving PCA'ed models..."):
        model_file = model_files[i]
        model_path = os.path.join(path, model_file, "models")
        model_file_path = os.path.join(model_path, "pca_embed.safetensors")
        if os.path.exists(model_file_path):
            print(f"Deleting old {model_file_path}...")
            os.remove(model_file_path)
        save_safetensors({ 'pca_embed': X_transformed[i, :] }, model_file_path)


def main():
    pl.seed_everything(1)
    args = parse_args()
    init_pca(args.dataset_path, args.pca_output_path, args.n_components, args.val_split)
    
if __name__ == "__main__":
    main()