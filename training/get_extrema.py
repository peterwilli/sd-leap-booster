import os
import torch

def get_min_weights(embed_model, current) -> float:
    min_model = torch.min(embed_model).item()
    if current is None:
        return min_model
    return min(current, min_model)

def get_max_weights(embed_model, current) -> float:
    max_model = torch.max(embed_model).item()
    if current is None:
        return max_model
    return max(current, max_model)
    
def get_extrema(data_loader):
    min_weight = None
    max_weight = None

    for _, embed_batch in data_loader:
        for i in range(embed_batch.shape[0]):
            embed_model = embed_batch[i]
            min_weight = get_min_weights(embed_model, min_weight)
            max_weight = get_max_weights(embed_model, max_weight)
    return min_weight, max_weight
