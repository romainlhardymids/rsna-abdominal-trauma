import sys
import torch
import torch.nn as nn

from . import nextvit
from timm import create_model
from segmentation.inflate import InflatedEfficientNet, InflatedConvNeXt, InflatedResNest


def create_encoder(encoder_params):
    """Initializes an encoder from a given configuration."""
    module = getattr(sys.modules[__name__], encoder_params["class"])
    name = encoder_params["encoder_name"]
    return module(name=name, **encoder_params["params"])


class BaseEncoder(nn.Module):
    """Base encoder class."""
    def __init__(self, out_channels, **kwargs):
        super().__init__()
        self.out_channels = out_channels

    def get_stages(self):
        return [nn.Identity()]

    def forward(self, x):
        stages = self.get_stages()
        features = []
        for stage in stages:
            x = stage(x)
            features.append(x)
        return features
    

class ConvNeXtEncoder2d(BaseEncoder):
    """ConvNeXt encoder class."""
    def __init__(
        self, 
        name,
        stage_idx,
        backbone_params={},
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = create_model(name, **backbone_params)
        assert len(stage_idx) <= len(self.encoder.stages)
        self.stage_idx = stage_idx
        self.depth = len(stage_idx) + 2

    def get_stages(self):
        return [nn.Identity(), self.encoder.stem] + \
            [self.encoder.stages[i : j] for i, j in zip([0] + self.stage_idx, self.stage_idx + [len(self.encoder.stages)])]

    def forward_head(self, x):
        x = self.encoder.norm_pre(x)
        x = self.encoder.forward_head(x)
        return x


class EfficientNetEncoder2d(BaseEncoder):
    """EfficientNet encoder class."""
    def __init__(
        self, 
        name,
        stage_idx,
        backbone_params={},
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = create_model(name, **backbone_params)
        assert len(stage_idx) <= len(self.encoder.blocks)
        self.stage_idx = stage_idx
        self.depth = len(stage_idx) + 2

    def get_stages(self):
        return [nn.Identity(), nn.Sequential(self.encoder.conv_stem, self.encoder.bn1)] + \
            [self.encoder.blocks[i : j] for i, j in zip([0] + self.stage_idx, self.stage_idx + [len(self.encoder.blocks)])]
    
    def forward_head(self, x):
        x = self.encoder.conv_head(x)
        x = self.encoder.bn2(x)
        x = self.encoder.forward_head(x)
        return x