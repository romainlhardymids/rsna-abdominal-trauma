import torch.nn as nn

from timm import create_model


def create_segmentation_model(config):
    return SegmentationModel(**config)


class SegmentationModel(nn.Module):
    def __init__(self, encoder_params):
        super().__init__()
        self.model = create_model(**encoder_params)

    def forward(self, x):
        logits = self.model(x)
        return logits
