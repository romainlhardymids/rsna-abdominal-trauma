import torch
import torch.nn as nn
import transformers

from layers import conv3d_same
from segmentation import decoder
from segmentation.encoder import create_encoder
from segmentation.modules import SegmentationHead
from segmentation_models_pytorch.base.initialization import initialize_decoder, initialize_head
from timm import create_model
from timm.models.layers import Conv2dSame


def create_segmentation_model(config):
    """Initializes a segmentation model from a given configuration."""
    config_ = config.copy()
    family = config_.pop("family")
    if family == "unet":
        return inflate_module(Unet(**config_))
    else:
        raise ValueError(f"Model family {family} is not supported.")
    

def inflate_module(module):
    """Copies the weights of a two-dimensional module to its three-dimensional equivalent."""
    module_ = module
    if isinstance(module, nn.BatchNorm2d):
        module_ = nn.BatchNorm3d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_.weight = module.weight
                module_.bias = module.bias
        module_.running_mean = module.running_mean
        module_.running_var = module.running_var
        module_.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_.qconfig = module.qconfig
    elif isinstance(module, Conv2dSame):
        module_ = conv3d_same.Conv3dSame(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
        )
        module_.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1, 1, 1, 1,module.kernel_size[0]))
    elif isinstance(module, torch.nn.Conv2d):
        module_ = torch.nn.Conv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode
        )
        module_.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1, 1, 1, 1,module.kernel_size[0]))
    elif isinstance(module, torch.nn.MaxPool2d):
        module_ = torch.nn.MaxPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        module_ = torch.nn.AvgPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
        )
    for name, child in module.named_children():
        module_.add_module(name, inflate_module(child))
    del module
    return module_


class Unet(nn.Module):
    """UNet segmentation model."""
    def __init__(
        self, 
        encoder_params,
        decoder_params, 
        num_classes=1
    ):
        super().__init__()
        self.encoder = create_encoder(encoder_params)
        self.decoder = decoder.UnetDecoder(self.encoder.out_channels, **decoder_params)
        self.head = SegmentationHead(
            decoder_params["decoder_channels"][-1], 
            num_classes, 
            kernel_size=3,
            padding=1, 
            upsampling=1
        )
        initialize_decoder(self.decoder)
        initialize_head(self.head)

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        logits = self.head(decoder_output)
        return logits
    

