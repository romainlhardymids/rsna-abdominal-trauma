import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, feature_dim, head_classes, **kwargs):
        super().__init__(**kwargs)
        self.head = nn.ModuleList([
            nn.Linear(feature_dim, hc) for hc in head_classes
        ])

    def forward(self, x):
        return [h(x) for h in self.head]