import os
import torch

def get_min_weights(embed_model, current) -> float:
    print("embed_model.flatten()", embed_model.flatten().shape)
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
        for key in mapping.keys():
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
