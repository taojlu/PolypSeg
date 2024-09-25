import pdb
import math
from model_comprise.polyp_pvt.pvtv2 import pvt_v2_b2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torch.utils.checkpoint as checkpoint

'''
代码来源于paper:
https://github.com/DengPingFan/Polyp-PVT/tree/main/lib
https://github.com/OSUPCVLab/SegFormer3D/blob/main/architectures/segformer3d.py
'''
ALIGN_CORNERS = True


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



class SelfAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int = 768,
            num_heads: int = 8,
            sr_ratio: int = 2,
            qkv_bias: bool = False,
            attn_dropout: float = 0.0,
            proj_dropout: float = 0.0,
    ):
        """
        embed_dim : hidden size of the PatchEmbedded input
        num_heads: number of attention heads
        sr_ratio: the rate at which to down sample the sequence length of the embedded patch
        qkv_bias: whether or not the linear projection has bias
        attn_dropout: the dropout rate of the attention component
        proj_dropout: the dropout rate of the final linear projection
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim should be divisible by number of heads!"

        self.num_heads = num_heads
        self.attention_head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.key_value = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                embed_dim, embed_dim, kernel_size=(sr_ratio, sr_ratio), stride=(sr_ratio, sr_ratio)
            )
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.query(x).reshape(B, N, self.num_heads, self.attention_head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # Assuming height = width for simplicity
            h = w = int(math.sqrt(N))
            x_ = x.permute(0, 2, 1).reshape(B, C, h, w)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = self.key_value(x_).reshape(B, -1, 2, self.num_heads, self.attention_head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.key_value(x).reshape(B, -1, 2, self.num_heads, self.attention_head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attention_score = (q @ k.transpose(-2, -1)) / math.sqrt(self.num_heads)
        attention_prob = attention_score.softmax(dim=-1)
        attention_prob = self.attn_dropout(attention_prob)
        out = (attention_prob @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        # 使用2D深度可分离卷积，维持原有参数设置
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        # x 的形状预期为 (B, C, H, W)
        x = self.dwconv(x)  # 应用深度可分离卷积
        x = self.bn(x)  # 应用批量归一化
        # pdb.set_trace()
        x = x.flatten(2).transpose(1, 2)
        return x


class _MLP(nn.Module):
    def __init__(self, in_feature, mlp_ratio=2, dropout=0.0):
        super().__init__()
        out_feature = in_feature  # 修改此处以匹配DWConv的通道数要求
        self.fc1 = nn.Linear(in_feature, out_feature)
        self.dwconv = DWConv(dim=out_feature)  # 此处dim应与out_feature相等
        self.fc2 = nn.Linear(out_feature, in_feature)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # pdb.set_trace()
        x = self.fc1(x)
        B, N, C = x.shape
        H = int(math.sqrt(N))
        W = int(math.sqrt(N))
        x = x.reshape(B, C, H, W)  # 确保这里的reshape正确
        x = self.dwconv(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int = 768,
            mlp_ratio: int = 2,
            num_heads: int = 8,
            sr_ratio: int = 2,
            qkv_bias: bool = False,
            attn_dropout: float = 0.0,
            proj_dropout: float = 0.0,
    ):
        """
        embed_dim : hidden size of the PatchEmbedded input
        mlp_ratio: at which rate increasse the projection dim of the embedded patch in the _MLP component
        num_heads: number of attention heads
        sr_ratio: the rate at which to down sample the sequence length of the embedded patch
        qkv_bias: whether or not the linear projection has bias
        attn_dropout: the dropout rate of the attention component
        proj_dropout: the dropout rate of the final linear projection
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = _MLP(in_feature=embed_dim, mlp_ratio=mlp_ratio, dropout=0.0)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, upsample_scale=2):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # 修改 expand 层，使其输出通道为输入的 1/2
        self.expand = nn.Linear(dim, dim // 2, bias=False)
        self.norm = norm_layer(dim // 2)  # 更新 norm_layer 参数以适应新的通道数
        self.upsample_scale = upsample_scale

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        x = self.norm(x)

        # Reshape for spatial dimensions and apply upsample
        x = x.permute(0, 2, 1).view(-1, self.dim // 2, H, W)  # Reshape to [B, C, H, W]
        x = F.interpolate(x, scale_factor=self.upsample_scale, mode='nearest')  # Perform upsampling

        return x


class PvtFormer(nn.Module):
    def __init__(self, channel=32,
                 in_channels: int = 4,
                 sr_ratios: list = [4, 2, 1, 1],
                 embed_dims: list = [32, 64, 160, 256],
                 patch_kernel_size: list = [7, 3, 3, 3],
                 patch_stride: list = [4, 2, 2, 2],
                 patch_padding: list = [3, 1, 1, 1],
                 mlp_ratios: list = [4, 4, 4, 4],
                 num_heads: list = [1, 2, 5, 8],
                 depths: list = [2, 2, 2, 2],
                 decoder_head_embedding_dim: int = 256,
                 num_classes: int = 3,
                 decoder_dropout: float = 0.0,

                 ):
        super(PvtFormer, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = '/root/nfs/gaobin/wt/PolypSeg_2/model_comprise/polyp_pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.de_block4_up = PatchExpand((11, 11), 512)

        self.de_block3_up = PatchExpand((22, 22), 576)

        self.de_block2_up = PatchExpand((44, 44), 416)

        self.norm4 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(576)
        self.norm2 = nn.LayerNorm(416)
        self.norm1 = nn.LayerNorm(272)

        self.decoder_block4 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=512,
                    num_heads=8,
                    mlp_ratio=2,
                    sr_ratio=2,
                    qkv_bias=True,
                )
                for _ in range(depths[0])
            ]
        )

        self.decoder_block3 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=576,
                    num_heads=4,
                    mlp_ratio=2,
                    sr_ratio=2,
                    qkv_bias=True,
                )
                for _ in range(depths[0])
            ]
        )

        self.decoder_block2 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=416,
                    num_heads=4,
                    mlp_ratio=2,
                    sr_ratio=2,
                    qkv_bias=True,
                )
                for _ in range(depths[0])
            ]
        )

        self.decoder_block1 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=272,
                    num_heads=4,
                    mlp_ratio=2,
                    sr_ratio=2,
                    qkv_bias=True,
                )
                for _ in range(depths[0])
            ]
        )
        self.head = BasicConv2d(272, 32, 1)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        B0, C0, H0, W0 = x.shape
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]  # ([1, 64, 88, 88])
        x2 = pvt[1]  # ([1, 128, 44, 44])
        x3 = pvt[2]  # ([1, 320, 22, 22])
        x4 = pvt[3]  # ([1, 512, 11, 11])

        # decoder block 4
        x = x4.flatten(2).transpose(1, 2)  # ([1, 121, 512])
        # B, N, C = x.shape

        for i, blk in enumerate(self.decoder_block4):
            # x = blk(x)  # ([1, 121, 512])
            x = checkpoint.checkpoint(blk, x)  # 使用梯度检查点
        # pdb.set_trace()
        x = self.norm4(x)

        # decoder block 3
        x4_up = self.de_block4_up(x)  # ([1, 256, 22, 22])
        x = torch.cat((x4_up, x3), dim=1)
        x = x.flatten(2).transpose(1, 2)  # ([1, 484, 512])
        for i, blk in enumerate(self.decoder_block3):
            # x = blk(x)  # ([1, 484, 512])
            x = checkpoint.checkpoint(blk, x)  # 使用梯度检查点
        x = self.norm3(x)

        # decoder block 2
        x3_up = self.de_block3_up(x)  # [1, 288, 44, 44]
        x = torch.cat((x3_up, x2), dim=1)
        x = x.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.decoder_block2):
            # x = blk(x)
            x = checkpoint.checkpoint(blk, x)  # 使用梯度检查点
        x = self.norm2(x)  # ([1, 1936, 416])
        # pdb.set_trace()
        # decoder block 1
        x2_up = self.de_block2_up(x)  # ([1, 208, 88, 88])
        x = torch.cat((x2_up, x1), dim=1)  # ([1, 272, 88, 88])
        x = x.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.decoder_block1):
            # x = blk(x)  # ([1, 7744, 272])
            x = checkpoint.checkpoint(blk, x)  # 使用梯度检查点
        x = self.norm1(x)
        d1_out = x.permute(0, 2, 1).view(B0, 272, 88, 88)

        d1_out = F.interpolate(d1_out, size=(H0, W0),
                               mode='bilinear', align_corners=ALIGN_CORNERS)

        # # pdb.set_trace()
        head = self.head(d1_out)
        out = self.out(head)

        return out


if __name__ == '__main__':
    device = torch.device('cuda:1')
    model = PvtFormer().to(device)
    input_tensor = torch.randn(1, 3, 352, 352).to(device)
    print("输入尺寸：", input_tensor.shape)

    prediction = model(input_tensor)
    print(prediction.size())
    summary(model, input_size=(1, 3, 352, 352), device=device)
