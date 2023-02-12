import os
import torch
import numpy as np

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for _, target in dataloader:
        channels_sum += torch.mean(data)
        channels_squared_sum += torch.mean(data**2)
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

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
