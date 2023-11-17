import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation import modules as md


class DecoderBlock(nn.Module):
    """UNet decoder block."""
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        block_depth=1,
        separable=False,
        use_aspp=False,
        use_batchnorm=True,
        attention_type=None,
        activation="relu"
    ):
        super().__init__()
        self.attention = nn.ModuleList([
            md.Attention(attention_type, in_channels=in_channels + skip_channels),
            md.Attention(attention_type, in_channels=out_channels)
        ])
        self.aspp = md.ASPP(
            in_channels,
            in_channels,
            atrous_rates=[1, 2, 4],
            reduction=2,
            dropout=0.2,
            activation=activation
        ) if use_aspp else nn.Identity()
        module = md.SeparableConvBnAct if separable else md.ConvBnAct
        self.stem = module(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
            activation=activation
        )
        self.body = nn.Sequential(*[
            module(
                out_channels, 
                out_channels, 
                kernel_size=3, 
                padding=1, 
                use_batchnorm=use_batchnorm,
                activation=activation
            ) for _ in range(block_depth)
         ])

    def forward(self, x, skip=None, scale_factor=1):
        if scale_factor != 1:
            x = F.interpolate(x, scale_factor=scale_factor, mode="trilinear")
        x = self.aspp(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention[0](x)
        x = self.stem(x)
        x = self.body(x)
        x = self.attention[1](x)
        return x


class UnetDecoder(nn.Module):
    """UNet decoder module."""
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        scale_factors,
        num_blocks=5,
        block_depth=1,
        separable=False,
        use_aspp=False,
        use_batchnorm=True,
        attention_type=None,
        activation="relu"
    ):
        super().__init__()
        assert num_blocks >= len(encoder_channels) - 1
        assert num_blocks == len(decoder_channels)
        assert num_blocks == len(scale_factors)
        self.scale_factors = scale_factors
        encoder_channels = encoder_channels[1:][::-1]
        in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:])
        skip_channels += [0] * (len(in_channels) - len(skip_channels))
        out_channels = decoder_channels
        aspp_idx = len(in_channels) - 2
        blocks = []
        for i, (i_ch, s_ch, o_ch) in enumerate(zip(in_channels, skip_channels, out_channels)):
            blocks.append(
                DecoderBlock(
                    i_ch, 
                    s_ch, 
                    o_ch, 
                    block_depth,
                    separable=separable,
                    use_aspp=use_aspp if i == aspp_idx else False,
                    use_batchnorm=use_batchnorm, 
                    attention_type=attention_type,
                    activation=activation
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[1:][::-1]
        x = features[0]
        skips = features[1:]
        for i, (block, scale_factor) in enumerate(zip(self.blocks, self.scale_factors)):
            skip = skips[i] if i < len(skips) else None
            x = block(x, skip, scale_factor)
        return x