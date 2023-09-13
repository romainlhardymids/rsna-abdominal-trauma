import torch
import torch.nn as nn

from timm.layers import LayerNorm2d, SelectAdaptivePool2d


class MaskEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, r, **kwargs):
        super().__init__(**kwargs)
        channels = [in_channels, in_channels * r, in_channels * r * r, out_channels]
        self.encoder = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=(2, 2), stride=(2, 2)),
            LayerNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[1], channels[2], kernel_size=(2, 2), stride=(2, 2)),
            LayerNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[2], channels[3], kernel_size=(1, 1), stride=(1, 1))
        )
        self.global_pool = SelectAdaptivePool2d(pool_type="avg", flatten=nn.Flatten(start_dim=1, end_dim=-1))

    def forward(self, x):
        x = 2.0 * torch.softmax(x, dim=1) - 1.0
        x = self.encoder(x)
        return self.global_pool(x)
