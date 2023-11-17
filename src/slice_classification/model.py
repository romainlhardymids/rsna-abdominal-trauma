import sys
sys.path.append("/home/romainlhardy/kaggle/rsna-abdominal-trauma/")
sys.path.append("/home/romainlhardy/kaggle/rsna-abdominal-trauma/sam_med2d/segment_anything")

import sys
import torch
import torch.nn as nn

from layers import heads
from timm import create_model

def create_slice_classification_model(encoder_params):
    """Initializes a slice-level classifier from a given configuration"""
    module = getattr(sys.modules[__name__], encoder_params["class"])
    name = encoder_params["encoder_name"]
    return module(name=name, **encoder_params["params"])


class SliceClassificationModel(nn.Module):
    """Slice-level classifier."""
    def __init__(self, name, backbone_params, dropout):
        super().__init__()
        self.num_channels = backbone_params["in_chans"]
        self.encoder = create_model(name, num_classes=0, **backbone_params)
        feature_dim = self.feature_dim()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.head = heads.ClassificationHead(feature_dim, (2, 3, 4, 4, 4, 5))

    def feature_dim(self):
        x = torch.randn(2, self.num_channels, 256, 256)
        return self.encoder(x).shape[-1]

    def forward_features(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.drop(x)
        return self.head(x)