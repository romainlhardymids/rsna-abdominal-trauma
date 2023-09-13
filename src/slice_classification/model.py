import sys
sys.path.append("/home/romainlhardy/kaggle/rsna-abdominal-trauma/")
sys.path.append("/home/romainlhardy/kaggle/rsna-abdominal-trauma/sam_med2d/segment_anything")

import copy
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from argparse import Namespace
from layers import heads, attention, modules
from sam_med2d.segment_anything import sam_model_registry
from scripts.segmentation.train import SegmentationModule
from segmentation.encoder import create_encoder
from timm import create_model
from timm.layers import SelectAdaptivePool2d


def create_slice_classification_model(encoder_params):
    module = getattr(sys.modules[__name__], encoder_params["class"])
    name = encoder_params["encoder_name"]
    return module(name=name, **encoder_params["params"])


# def create_slice_classification_model(config):
#     return SliceClassificationModel(**config)


class SliceClassificationModel(nn.Module):
    def __init__(self, name, backbone_params, dropout):
        super().__init__()
        self.encoder = create_model(name, num_classes=0, **backbone_params)
        feature_dim = self.feature_dim()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.head = heads.ClassificationHead(feature_dim, (2, 3, 4, 4, 4, 5))

    def feature_dim(self):
        x = torch.randn(2, 3, 256, 256)
        return self.encoder(x).shape[-1]

    def forward_features(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.drop(x)
        return self.head(x)
        


# class SliceClassificationModel(nn.Module):
#     def __init__(self, dropout, segmentation_checkpoint, encoder_params):
#         super().__init__()
#         self.seg_encoder = create_encoder(encoder_params)
#         if segmentation_checkpoint is not None:
#             state_dict = torch.load(segmentation_checkpoint)["state_dict"]
#             state_dict = {
#                 k.replace("model.encoder.encoder.", ""): v for k, v in state_dict.items()
#             }
#             self.seg_encoder.encoder.load_state_dict(state_dict, strict=False)
#         for param in self.seg_encoder.parameters():
#             param.requires_grad = False
#         self.seg_encoder.eval()
#         self.clf_encoder = create_encoder(encoder_params)
#         self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
#         feature_dim = self.feature_dim()
#         self.heads = nn.ModuleList([
#             nn.Linear(feature_dim, 2),
#             nn.Linear(feature_dim, 2),
#             nn.Linear(feature_dim, 3),
#             nn.Linear(feature_dim, 4),
#             nn.Linear(feature_dim, 4),
#             nn.Linear(feature_dim, 4),
#             nn.Linear(feature_dim, 5)
#         ])

#     def feature_dim(self):
#         x = torch.randn(2, 3, 256, 256)
#         return self.clf_encoder.encoder(x).shape[-1]

#     def forward_features(self, x):
#         seg_stages = self.seg_encoder.get_stages()
#         clf_stages = self.clf_encoder.get_stages()
#         x_seg, x_clf = x, x
#         for seg_stage, clf_stage in zip(seg_stages, clf_stages):
#             x_seg = seg_stage(x_seg)
#             x_clf = (clf_stage(x_clf) + x_seg) / 2.
#         features = self.clf_encoder.forward_head(x_clf)
#         return features

#     def forward(self, x):
#         features = self.forward_features(x)
#         features = self.drop(features)
#         return [head(features) for head in self.heads]


def load_segmentator(config_path, checkpoint_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    segmentator = SegmentationModule(config["model"])
    state_dict = torch.load(checkpoint_path)["state_dict"]
    segmentator.load_state_dict(state_dict)
    for param in segmentator.parameters():
        param.requires_grad = False
    segmentator.eval()
    return segmentator.model


class EfficientNetClassifier(nn.Module):
    def __init__(
        self, 
        name,
        backbone_params={},
        segmentator_params={},
        dropout=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        encoder = create_model(name, **backbone_params)
        self.conv_stem = encoder.conv_stem
        self.bn1 = encoder.bn1
        block_output_sizes, feature_dim = self.get_encoder_meta(encoder)
        blocks = []
        for block, output_size in zip(encoder.blocks, block_output_sizes):
            blocks.append(
                nn.Sequential(*[
                    block,
                    attention.Adapter(output_size, reduction=4, skip_connect=True)
                ])
            )
        self.blocks = nn.ModuleList(blocks)
        self.conv_head = encoder.conv_head
        self.bn2 = encoder.bn2
        self.global_pool = encoder.global_pool

        self.segmentator = load_segmentator(segmentator_params["config_path"], segmentator_params["checkpoint_path"])
        self.mask_encoder = modules.MaskEncoder(in_channels=6, out_channels=feature_dim, r=4)

        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.head = heads.ClassificationHead(feature_dim, (2, 2, 3, 4, 4, 4))

    def get_encoder_meta(self, encoder):
        x = torch.randn(2, 3, 224, 224)
        x = encoder.conv_stem(x)
        x = encoder.bn1(x) 
        sizes = []
        for block in encoder.blocks:
            x = block(x)
            sizes.append(x.shape[1])
        x = encoder.conv_head(x)
        x = encoder.bn2(x)
        feature_dim = x.shape[1]
        return sizes, feature_dim
    
    def forward_features(self, x):
        with torch.no_grad():
            m = self.segmentator(x)
        x = self.conv_stem(x)
        x = self.bn1(x) 
        for block in self.blocks:
            x = block(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.global_pool(x) + self.mask_encoder(m)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.drop(x)
        return self.head(x)


class SamMedClassifier(nn.Module):
    def __init__(
        self, 
        name,
        dropout=0.2,
        image_size=512,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert image_size % 256 == 0
        args = Namespace()
        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = "/home/romainlhardy/kaggle/rsna-abdominal-trauma/models/sam_med2d/sam-med2d_b.pth"
        self.encoder = sam_model_registry["vit_b"](args).image_encoder
        self.encoder.pos_embed = self.fix_pos_embed(self.encoder.pos_embed, image_size // 256)
        del self.encoder.neck
        self.global_pool = SelectAdaptivePool2d(pool_type="avg", flatten=nn.Flatten(start_dim=1, end_dim=-1))

        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.head = heads.ClassificationHead(768, (2, 3, 4, 4, 4, 5))

    def fix_pos_embed(self, pos_embed, r):
        _, h, w, c = pos_embed.shape
        return nn.Parameter(
            F.interpolate(pos_embed[:, None], (h * r, w * r, c), mode="trilinear")[:, 0]
        )

    def forward(self, x):
        x = self.encoder.patch_embed(x)
        if self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed
        for block in self.encoder.blocks:
            x = block(x)
        x = x.permute(0, 3, 1, 2)
        x = self.global_pool(x)
        x = self.drop(x)
        return self.head(x)


class EfficientNetMaskGuided(nn.Module):
    def __init__(
        self, 
        name,
        backbone_params={},
        segmentator_params={},
        dropout=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        encoder = create_model(name, **backbone_params)
        self.conv_stem = encoder.conv_stem
        self.bn1 = encoder.bn1
        block_output_sizes, feature_dim = self.get_encoder_meta(encoder)
        main_blocks = []
        for block, output_size in zip(encoder.blocks[:-2], block_output_sizes[:-2]):
            main_blocks.append(
                nn.Sequential(*[
                    block,
                    attention.SoftAttention(output_size, reduction=4)
                ])
            )
        main_blocks.append(encoder.blocks[-2])
        self.main_blocks = nn.ModuleList(main_blocks)
        self.mask_attention_layers = nn.ModuleList([
            attention.SoftAttention(block_output_sizes[-2], reduction=4) for _ in range(5)
        ])
        self.final_blocks = nn.ModuleList([
            nn.Sequential(*[
                copy.deepcopy(encoder.blocks[-1]),
                attention.SoftAttention(block_output_sizes[-1], reduction=4),
                copy.deepcopy(encoder.conv_head),
                copy.deepcopy(encoder.bn2),
                copy.deepcopy(encoder.global_pool),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            ]) for _ in range(5)
        ])

        self.segmentator = load_segmentator(segmentator_params["config_path"], segmentator_params["checkpoint_path"])
        self.mask_encoder = modules.MaskEncoder(in_channels=6, out_channels=feature_dim, r=4)

        self.head = nn.ModuleList([
            heads.ClassificationHead(feature_dim, (2, 2, 3, 4, 4, 4)),
            heads.ClassificationHead(feature_dim, (2, 2, 3)),
            heads.ClassificationHead(feature_dim, (2, 2, 4)),
            heads.ClassificationHead(feature_dim, (2, 2, 4)),
            heads.ClassificationHead(feature_dim, (2, 2, 4))
        ])

    def get_encoder_meta(self, encoder):
        x = torch.randn(2, 3, 224, 224)
        x = encoder.conv_stem(x)
        x = encoder.bn1(x) 
        sizes = []
        for block in encoder.blocks:
            x = block(x)
            sizes.append(x.shape[1])
        x = encoder.conv_head(x)
        x = encoder.bn2(x)
        feature_dim = x.shape[1]
        return sizes, feature_dim
    
    def forward(self, x):
        with torch.no_grad():
            segmentation = torch.sigmoid(self.segmentator(x))

        x = self.conv_stem(x)
        x = self.bn1(x) 
        for block in self.main_blocks:
            x = block(x)
        features = []
        attention_weights = []
        for layer in self.mask_attention_layers:
            f, w = layer(x, return_logits=True)
            features.append(f)
            attention_weights.append(w)
        logits = []
        for f, final_block, h in zip(features, self.final_blocks, self.head):
            logits.append(h(final_block(f)))
        logits = [
            torch.max(torch.stack([logits[i][0] for i in range(5)], dim=1), dim=1)[0],
            torch.max(torch.stack([logits[i][1] for i in range(5)], dim=1), dim=1)[0],
            torch.max(torch.stack([logits[0][2], logits[1][2]], dim=1), dim=1)[0],
            torch.max(torch.stack([logits[0][3], logits[2][2]], dim=1), dim=1)[0],
            torch.max(torch.stack([logits[0][4], logits[3][2]], dim=1), dim=1)[0],
            torch.max(torch.stack([logits[0][5], logits[4][2]], dim=1), dim=1)[0]
        ]
        return logits, segmentation, attention_weights