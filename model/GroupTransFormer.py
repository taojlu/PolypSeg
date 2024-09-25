
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from einops import rearrange
from functools import partial

class ConvStem(nn.Module):
    """ Image to Patch Embedding """

    def __init__(self, in_dim=3, embedding_dims=64):
        super().__init__()
        mid_dim = embedding_dims // 2

        self.proj1 = nn.Conv2d(in_dim, mid_dim, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.SyncBatchNorm(mid_dim)
        self.act1 = nn.Hardswish()

        self.proj2 = nn.Conv2d(mid_dim, embedding_dims, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.SyncBatchNorm(embedding_dims)
        self.act2 = nn.Hardswish()

    def forward(self, x):
        x = self.act1(self.norm1(self.proj1(x)))
        x = self.act2(self.norm2(self.proj2(x)))
        return x

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.pointwise_conv(self.conv1(x))
        return x

class PatchEmbedLayer(nn.Module):
    def __init__(self, patch_size=16, in_dim=3, embedding_dims=768, is_first_layer=False):
        super().__init__()
        if is_first_layer:
            patch_size = 1
            in_dim = embedding_dims

        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = SeparableConv2d(in_dim, embedding_dims, 3, patch_size, 1)
        self.norm = nn.SyncBatchNorm(embedding_dims)
        self.act = nn.Hardswish()

    def forward(self, x):
        _, _, H, W = x.shape
        out_H, out_W = H // self.patch_size[0], W // self.patch_size[1]
        x = self.act(self.norm(self.proj(x)))
        x = x.flatten(2).transpose(1, 2)
        return x, (out_H, out_W)


class GroupFormer(nn.Module):
    def __init__(
            self,
            patch_size=4,  # seem useless
            in_dim=3,  # seem useless
            num_stages=1,
            num_classes=1000,
            embedding_dims=[80, 160, 320, 320],
            serial_depths=[2, 4, 12, 4],
            num_heads=8,
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            return_interm_layers=True,
            out_features=None,  # seem useless
            pretrained=False  # seem useless
    ):
        super().__init__()
        self.num_stages = num_stages

        self.conv_stem = ConvStem(in_dim=in_dim, embedding_dims=embedding_dims[0])
        self.patch_embed_layers = nn.ModuleList([
            PatchEmbedLayer(
                patch_size=2,
                in_dim=embedding_dims[i - 1],
                embedding_dims=embedding_dims[i],
                is_first_layer=True if i == 0 else False,
            ) for i in range(self.num_stages)
        ])



    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.conv_stem(x)

        # i = 1
        for i in range(self.num_stages):
            x_patch, (H, W) = self.patch_embed_layers[i](x)
            print('x_patch: ', x_patch.shape)

        return x_patch




if __name__ == "__main__":
    device = 'cuda'
    model = GroupFormer().to(device)
    model.eval()
    inputs = torch.randn(1, 3, 352, 352).to(device)
    out = model(inputs)
    print(out.shape)



