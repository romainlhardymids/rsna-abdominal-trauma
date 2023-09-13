import torch
import torch.nn as nn

from timm.layers import LayerNorm2d


class Attention(nn.Module):
    def __init__(self, time_dim, hidden_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.features_dim = 0
        self.weight = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(hidden_dim, 1))
        )
        self.bias = nn.Parameter(torch.zeros(time_dim)) if bias else None
        self.act = nn.Tanh()
        
    def forward(self, x, mask=None):
        eij = torch.mm(
            x.contiguous().view(-1, self.hidden_dim), 
            self.weight
        ).view(-1, self.time_dim)
        if self.bias is not None:
            eij = eij + self.bias
        eij = self.act(eij)
        a = torch.exp(eij)
        if mask is not None:
            a = a * mask
        a = a / torch.sum(a, 1, keepdim=True) + 1e-10
        weighted_input = x * torch.unsqueeze(a, dim=-1)
        return torch.sum(weighted_input, dim=1)


class SoftAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1)
        )
        self.sSE = nn.Conv2d(in_channels, 1, 1)
        self.head = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x, return_logits=False):
        logits = self.head(self.cSE(x) * self.sSE(x))
        weights = torch.sigmoid(logits)
        if return_logits:
            return (1.0 + weights) * x, logits
        else:
            return (1.0 + weights) * x


class Adapter(nn.Module):
    def __init__(self, in_channels, reduction=4, skip_connect=True):
        super().__init__()
        self.norm = LayerNorm2d(in_channels)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True)
        )
        self.skip_connect = skip_connect

    def forward(self, x):
        x_channel = self.channel(x) * x
        x_spatial = self.spatial(x_channel)
        if self.skip_connect:
            x = x + x_spatial
        else:
            x = x_spatial
        return self.norm(x)
