
from torch import nn
import torch
import torch.nn.functional as F
import math

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

class LEAPBufferHiddenLayer(nn.Module):
    def __init__(self, act_fn, size: int, return_pos: bool):
        super().__init__() 
        self.net = nn.Sequential(
            nn.Linear(size, size),
            # nn.BatchNorm1d(size)
        )
        self.act_fn = act_fn
        self.return_pos = return_pos

    def rescale(self, x):
        x = x + abs(torch.min(x))
        x = x / abs(torch.max(x))
        return x

    def forward(self, x_pos):
        slice_pos = int(x_pos.shape[1] / 2)
        x = x_pos[:, 0:slice_pos]
        positional_encoding = x_pos[:, slice_pos:x_pos.shape[1]]
        z = self.net(x + positional_encoding)
        if self.act_fn is None:
            out = z
        else:
            out = self.act_fn(z)
        if self.return_pos:
            return torch.cat((out, positional_encoding), dim=1)
        return out

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
        self.net_in = nn.Linear(self.size_in, self.hidden_size)
        hidden_layers = []
        for i in range(self.n_hidden):
            return_pos = i < (self.n_hidden - 1)
            hidden_layers += [
                LEAPBufferHiddenLayer(nn.LeakyReLU(), self.hidden_size, return_pos),
                nn.Dropout(p=self.dropout_p)
            ]
        self.net_hidden = nn.Sequential(*hidden_layers)
        self.net_out = nn.Linear(self.hidden_size, self.hidden_size)
            
    def unroll_buffer(self, x):
        amount_to_unroll = math.ceil(self.size_out / self.hidden_size)
        sinspace = torch.sin(torch.linspace(0, 2 * math.pi, self.hidden_size * amount_to_unroll, device=x.device))
        result = torch.zeros(x.shape[0], self.hidden_size * amount_to_unroll, device=x.device)
        for idx in range(amount_to_unroll):
            positional_encoding = sinspace[self.hidden_size * idx:self.hidden_size * (idx + 1)]
            # positional_encoding += abs(torch.min(positional_encoding))
            # positional_encoding /= torch.max(positional_encoding)
            positional_encoding = positional_encoding.expand(x.shape[0], -1)
            x_pos = torch.cat((x, positional_encoding), dim=1)
            result[:, self.hidden_size * idx:self.hidden_size * (idx + 1)] = self.net_out(self.net_hidden(x_pos))
        return result[:, :self.size_out]

    def forward(self, x):
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