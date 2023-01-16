import os
import torch

def get_min_weights(embed_model, current) -> float:
    min_model = min(embed_model).item()
    if current is None:
        return min_model
    return min(current, min_model)

def get_max_weights(embed_model, current) -> float:
    max_model = max(embed_model).item()
    if current is None:
        return max_model
    return max(current, max_model)
    
def get_extrema(folder_to_scan):
    min_weight = None
    max_weight = None

    for folder in os.listdir(folder_to_scan):
        folder_path = os.path.join(folder_to_scan, folder)
        if os.path.isdir(folder_path):
            embed_path = os.path.join(folder_path, "learned_embeds.bin")
            loaded_learned_embeds = torch.load(embed_path, map_location="cpu")
            embed_model = loaded_learned_embeds[list(loaded_learned_embeds.keys())[0]].detach()
            min_weight = get_min_weights(embed_model, min_weight)
            max_weight = get_max_weights(embed_model, max_weight)
    return min_weight, max_weight