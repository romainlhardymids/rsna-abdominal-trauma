import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import attention, heads


def create_scan_classification_model(config):
    return ScanClassificationModel(**config)


# class ScanClassificationModel(torch.nn.Module):
#     def __init__(
#         self, 
#         time_dim, 
#         feature_dim, 
#         hidden_dim, 
#         dropout=0.2,
#         bidirectional=True
#     ):
#         super().__init__()
#         self.lstm = nn.GRU(
#             feature_dim, 
#             hidden_dim, 
#             bidirectional=bidirectional,
#             batch_first=True
#         )
#         scale_factor = 2 if bidirectional else 1
#         self.attention = attention.Attention(time_dim, hidden_dim * scale_factor)
#         self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
#         self.head = heads.ClassificationHead(2 * scale_factor * hidden_dim, (2, 2, 2, 3, 3, 3))

#     def forward(self, x, masks):
#         n, c, t, f = x.shape
#         x = x.view(-1, t, f)
#         masks = masks.view(-1, t)
#         x, _ = self.lstm(x)
#         max_pool, _ = torch.max(x, dim=1)
#         att_pool = self.attention(x, masks)
#         x = torch.cat([max_pool, att_pool], dim=1)
#         x = x.view(n, c, -1).max(dim=1)[0]
#         x = self.drop(x)
#         return self.head(x)


class ScanClassificationModel(torch.nn.Module):
    def __init__(
        self, 
        time_dim, 
        feature_dim, 
        hidden_dim, 
        dropout=0.2,
        bidirectional=True
    ):
        super().__init__()
        self.lstm = nn.GRU(
            feature_dim, 
            hidden_dim, 
            bidirectional=bidirectional,
            batch_first=True
        )
        scale_factor = 2 if bidirectional else 1
        self.attention = attention.Attention(time_dim, hidden_dim * scale_factor)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.seg_head = heads.ClassificationHead(scale_factor * hidden_dim, [5])
        self.clf_head = heads.ClassificationHead(scale_factor * hidden_dim * 2, [2, 2, 3, 3, 3])

    def forward(self, x, mask):
        x, _ = self.lstm(x)
        x = self.drop(x)
        logits_list = self.seg_head(x * torch.unsqueeze(mask, dim=-1))
        max_pool, _ = torch.max(x, dim=1)
        att_pool = self.attention(x, mask)
        cat = torch.cat([max_pool, att_pool], dim=1)
        logits_list = self.clf_head(cat) + logits_list
        return logits_list