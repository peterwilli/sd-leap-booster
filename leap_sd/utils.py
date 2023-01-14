import torch
import math
from einops import rearrange
import random

def linear_warmup_cosine_decay(optimizer, warmup_steps, total_steps):
    """
    Linear warmup for warmup_steps, with cosine annealing to 0 at total_steps
    """

    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, fn
    )
