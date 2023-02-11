
from torch import nn
import torch
import torch.nn.functional as F
import math
from hflayers import Hopfield, HopfieldPooling

class EmbedNormalizer(nn.Module):
    def __init__(self, mapping, extrema):
        super().__init__() 
        self.mapping = mapping
        self.extrema = extrema    

    def forward(self, embed):
        embed = embed.clone()
        keys = list(self.mapping.keys())
        keys.sort()
        len_done = 0
        for key in keys:
            obj = self.mapping[key]
            obj_extrema = self.extrema[key]
            mapping_len = obj['len']
            model_slice = embed[:, len_done:len_done + mapping_len]
            max_weight = obj_extrema['max']
            min_weight = obj_extrema['min']
            model_slice -= min_weight
            model_slice /= (max_weight - min_weight)
            len_done += mapping_len
        return embed

class EmbedDenormalizer(nn.Module):
    def __init__(self, mapping, extrema):
        super().__init__() 
        self.mapping = mapping
        self.extrema = extrema    

    def forward(self, embed):
        embed = embed.clone()
        keys = list(self.mapping.keys())
        keys.sort()
        len_done = 0
        for key in keys:
            obj = self.mapping[key]
            obj_extrema = self.extrema[key]
            mapping_len = obj['len']
            model_slice = embed[:, len_done:len_done + mapping_len]
            max_weight = obj_extrema['max']
            min_weight = obj_extrema['min']
            model_slice *= (max_weight - min_weight)
            model_slice += min_weight
            len_done += mapping_len
        return embed
        
class LEAPBuffer(nn.Module):
    def __init__(self, size_in: int, size_out: int, hidden_size: int, n_hidden: int, dropout_p: float = 0.01):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.dropout_p = dropout_p
        self.init_model()

    def init_model(self):
        # pooled_size = self.size_in // 2
        # self.hopfield_pooling = HopfieldPooling(
        #     input_size=self.size_in, hidden_size=32, output_size=pooled_size, num_heads=1
        # )
        self.net_in = nn.Linear(self.size_in, self.hidden_size)
        hidden_layers = []
        for i in range(self.n_hidden):
            hidden_layers += [
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p)
            ]
        self.net_hidden = nn.Sequential(*hidden_layers)
        self.net_out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )
            
    def unroll_buffer(self, x):
        amount_to_unroll = math.ceil(self.size_out / self.hidden_size)
        sinspace = torch.sin(torch.linspace(0, 2 * math.pi, self.hidden_size * amount_to_unroll, device=x.device))
        result = torch.zeros(x.shape[0], self.hidden_size * amount_to_unroll, device=x.device)
        for idx in range(amount_to_unroll):
            positional_encoding = sinspace[self.hidden_size * idx:self.hidden_size * (idx + 1)]
            # positional_encoding += abs(torch.min(positional_encoding))
            # positional_encoding /= torch.max(positional_encoding)
            positional_encoding = positional_encoding.expand(x.shape[0], -1)
            result[:, self.hidden_size * idx:self.hidden_size * (idx + 1)] = self.net_out(self.net_hidden(x + positional_encoding))
        return result[:, :self.size_out]

    def forward(self, x):
        # x = x.unsqueeze(1)
        # x = self.hopfield_pooling(x)
        # x = x.squeeze(1)
        x = self.net_in(x)
        return self.unroll_buffer(x)

class LEAPBlock(nn.Module):
    def __init__(self, act_fn, subsample_count: int, c_out: int, stride: int):
        super().__init__() 
        self.net = nn.Sequential(
            nn.AvgPool2d(subsample_count),
            nn.Conv2d(
                3, c_out, kernel_size=3, padding=1, stride=stride
            ),
            nn.Flatten()
        )
        self.act_fn = act_fn

    def forward(self, image):
        z = self.net(image)
        if self.act_fn is None:
            out = z
        else:
            out = self.act_fn(z)
        return out