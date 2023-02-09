import os
import torch
import numpy as np

def get_mean_std(data_loader) -> (float, float):
    mean = 0.0
    std = 0.0
    for _, y in data_loader:
        batch_samples = y.shape[0]
        y = y.view(batch_samples, -1)
        mean += y.mean(1).sum(0)
        std += y.std(1).sum(0)

    mean /= len(data_loader.dataset)
    std /= len(data_loader.dataset)
    return mean, std

def get_min_weights(embed_model, current) -> float:
    min_model = torch.min(embed_model.flatten()).item()
    if current is None:
        return min_model
    return min(current, min_model)

def get_max_weights(embed_model, current) -> float:
    max_model = torch.max(embed_model.flatten()).item()
    if current is None:
        return max_model
    return max(current, max_model)
    
def get_extrema(data_loader, mapping):
    result = {}
    for _, embed_batch in data_loader:
        len_done = 0

        keys = list(mapping.keys())
        keys.sort()

        for key in keys:
            obj = mapping[key]
            mapping_len = obj['len']
            model_slice = embed_batch[:, len_done:len_done + mapping_len]
            if not key in result:
                result[key] = {
                    'min': None,
                    'max': None
                }
            result[key]['min'] = get_min_weights(model_slice, result[key]['min'])
            result[key]['max'] = get_max_weights(model_slice, result[key]['max'])
            len_done += mapping_len
    return result
