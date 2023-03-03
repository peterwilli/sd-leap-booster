
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

class LeapAvgPool1D(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.halved_size = math.ceil(input_size / 2)
        self.params = nn.Parameter(torch.rand(self.halved_size), requires_grad=True)
        self.act_fn = nn.Sigmoid()

    def forward(self, data):
        output = torch.zeros(data.shape[0], self.halved_size).to(data.device)
        for i in range(self.halved_size):
            x_start = data[:, i]
            x_end = data[:, min(data.shape[1] - 1, i + 1)]
            x_new = torch.lerp(x_start, x_end, (self.params[i]))
            output[:, i] = x_new
        return output

class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        dropout=0.0,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x

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