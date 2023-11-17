# Conv3d with 'SAME' padding, modified from
# https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/conv2d_same.py
# https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/padding.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional, List


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    """Calculate symmetric padding for a convolution."""
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def get_same_padding(x, k, s, d):
    """Calculate asymmetric TF-like 'SAME' padding for a convolution"""
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


def is_static_pad(kernel_size, stride, dilation, **_):
    """Returns True if padding can be done statically, and False otherwise."""
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1, 1), value: float = 0):
    """Dynamically pads an input x using 'SAME' padding."""
    ih, iw, iz = x.size()[-3:]
    pad_h = get_same_padding(ih, k[0], s[0], d[0])
    pad_w = get_same_padding(iw, k[1], s[1], d[1])
    pad_z = get_same_padding(iz, k[2], s[2], d[2])
    if pad_h > 0 or pad_w > 0 or pad_z > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_z // 2, pad_z - pad_z // 2], value=value)
    return x


def get_padding_value(padding, kernel_size, **kwargs):
    """Returns the correct padding value based on the padding type."""
    dynamic = False
    if isinstance(padding, str):
        padding = padding.lower()
        if padding == "same":
            if is_static_pad(kernel_size, **kwargs):
                padding = get_padding(kernel_size, **kwargs)
            else:
                padding = 0
                dynamic = True
        elif padding == "valid":
            padding = 0
        else:
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


def conv3d_same(x, weight, bias, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1):
    """Functional implementation of a 3D convolution with 'SAME' padding."""
    x = pad_same(x, weight.shape[-3:], stride, dilation)
    return F.conv3d(x, weight, bias, stride, (0, 0, 0), dilation, groups)