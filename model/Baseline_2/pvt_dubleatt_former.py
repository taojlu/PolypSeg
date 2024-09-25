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

class DSSAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, sr_ratio, qkv_bias=False, attn_dropout=0.0, proj_dropout=0.0, rank=32):
        super(DSSAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.scale = math.sqrt(embed_dim // num_heads)

        # Apply low-rank approximation to reduce parameter count
        self.query = nn.Sequential(
            nn.Linear(embed_dim, rank, bias=qkv_bias),
            nn.Linear(rank, embed_dim, bias=qkv_bias)
        )
        self.key_value = nn.Sequential(
            nn.Linear(embed_dim, rank * 2, bias=qkv_bias),
            nn.Linear(rank * 2, 2 * embed_dim, bias=qkv_bias)
        )
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, rank, bias=qkv_bias),
            nn.Linear(rank, embed_dim, bias=qkv_bias)
        )
        self.proj_dropout = nn.Dropout(proj_dropout)

        # Dynamic focus layer with depthwise separable convolution
        self.dynamic_focus = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, groups=embed_dim),  # depthwise convolution
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)  # pointwise convolution
        )

    def forward(self, x):
        B, N, C = x.shape
        q = self.query(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.key_value(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        weighted_v = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N, C)

        # Apply dynamic focus layer
        weighted_v = weighted_v.permute(0, 2, 1).view(B, C, int(math.sqrt(N)), int(math.sqrt(N)))
        dynamic_attn = self.dynamic_focus(weighted_v)
        dynamic_attn = dynamic_attn.flatten(2).transpose(1, 2)

        x = dynamic_attn + x
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x
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
        self.attention = DSSAttention(
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


class MemoryAttention(nn.Module):
    def __init__(self, channel_size, d_model, dim_feedforward, dropout, num_heads=1):
        super(MemoryAttention, self).__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.channel_size = channel_size
        self.num_heads = num_heads

        # Self-attention using multi-head attention for spatial data
        self.self_attn = nn.MultiheadAttention(embed_dim=channel_size, num_heads=num_heads, batch_first=True)

        # Positional Encoding that respects spatial dimensions
        self.position_encoding = nn.Parameter(torch.randn(1, channel_size, 1, 1), requires_grad=True)

        # Feedforward layers
        self.conv1 = nn.Conv2d(channel_size, dim_feedforward, kernel_size=1)
        self.conv2 = nn.Conv2d(dim_feedforward, channel_size, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

        # Normalization and Activation
        self.norm1 = nn.BatchNorm2d(channel_size)
        self.norm2 = nn.BatchNorm2d(channel_size)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, tgt):
        # Adding positional encoding
        B, C, H, W = tgt.size()
        tgt += self.position_encoding

        # Reshape for multi-head attention
        tgt_reshaped = tgt.flatten(2).permute(0, 2, 1)  # [Batch, Height*Width, Channels]

        # Self-Attention processing
        attn_output, _ = self.self_attn(tgt_reshaped, tgt_reshaped, tgt_reshaped)
        attn_output = attn_output.permute(0, 2, 1).view(B, C, H, W)  # [Batch, Channels, Height, Width]

        # Combine with the original input
        tgt = attn_output + tgt

        # Feedforward processing
        tgt = self.conv2(self.dropout(self.activation(self.conv1(self.norm1(tgt))))) + self.norm2(tgt)

        return tgt


def split_image(image):
    """
    将输入图像划分为四个子图
    :param image: 输入图像，尺寸为 (1, 1, 256, 256)
    :return: 四个子图，尺寸分别为 (1, 1, 128, 128)
    """
    b, c, h, w = image.size()
    patches = []

    patch_size = h // 2  # 子图尺寸
    for i in range(2):
        for j in range(2):
            patch = image[:, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            patches.append(patch)

    return patches


class ChannelAttention(nn.Module):
    """
    CBAM混合注意力机制的通道注意力
    """
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x

class xLSTMPooling(nn.Module):
    def __init__(self, input_channels, hidden_size):
        super(xLSTMPooling, self).__init__()
        self.lstm = nn.LSTM(input_size=input_channels,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)

        # 用于生成注意力权重的卷积层
        self.attention_conv = nn.Conv2d(hidden_size * 2, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入维度 [B, C, H, W], 我们处理宽度W作为序列
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # [B*H, W, C]
        x, _ = self.lstm(x)  # [B*H, W, 2*hidden_size]
        x = x.view(b, h, w, -1).permute(0, 3, 1, 2)  # [B, 2*hidden_size, H, W]
        x = self.attention_conv(x)  # [B, 1, H, W]
        x = self.sigmoid(x)
        return x

def combine_patches(patches):
    """
    将四个子图合并成原图
    :param patches: 四个子图，尺寸分别为 (1, 1, 128, 128)
    :return: 合并后的图像，尺寸为 (1, 1, 256, 256)
    """
    top = torch.cat((patches[0], patches[1]), dim=3)
    bottom = torch.cat((patches[2], patches[3]), dim=3)
    combined_image = torch.cat((top, bottom), dim=2)

    return combined_image

class LocalSpatialAttention(nn.Module):
    """
    使用跨维度交互的局部空间注意力机制
    """
    def __init__(self, input_channels, num_splits=2, lstm_hidden_size=20):
        super(LocalSpatialAttention, self).__init__()
        self.num_splits = num_splits
        self.lstm_pooling = xLSTMPooling(input_channels, lstm_hidden_size)

    def forward(self, x):
        # 分割输入特征图
        patches = split_image(x)
        attention_maps = []

        for patch in patches:
            # 使用 LSTM 池化来处理每个 patch
            attention = self.lstm_pooling(patch)
            attention_maps.append(attention)

        # 重新合成完整的注意力图
            # 将局部分支的输出合并并上采样
        local_output_combined = combine_patches(attention_maps)
        # 将注意力图应用于输入特征图
        return local_output_combined * x

class CBAM(nn.Module):
    """
    CBAM混合注意力机制
    """
    def __init__(self, in_channels, out_channels, ratio=16):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(in_channels, ratio=ratio)
        self.spatialattention = LocalSpatialAttention(input_channels=in_channels, num_splits=4)
        self.reduce_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.channelattention(x)
        x = self.spatialattention(x)
        x = self.reduce_conv(x)  # 降维
        x = self.bn(x)  # 添加批量归一化
        return x


class DuattFormer(nn.Module):
    def __init__(self, channel=32,
                 in_channels: int = 4,
                 sr_ratios: list = [4, 2, 1, 1],
                 embed_dims: list = [32, 64, 160, 256],
                 patch_kernel_size: list = [7, 3, 3, 3],
                 patch_stride: list = [4, 2, 2, 2],
                 patch_padding: list = [3, 1, 1, 1],
                 mlp_ratios: list = [4, 4, 4, 4],
                 num_heads: list = [1, 2, 5, 8],
                 depths: list = [1, 1, 1, 1],
                 decoder_head_embedding_dim: int = 256,
                 num_classes: int = 3,
                 decoder_dropout: float = 0.0,

                 ):
        super(DuattFormer, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = '/root/nfs/gaobin/wt/PolypSeg_2/model_comprise/polyp_pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # bridge block
        self.bridge_attention = MemoryAttention(channel_size=512, d_model=512, dim_feedforward=1024, dropout=0.1)

        self.de_block4_up = PatchExpand((11, 11), 512)

        self.de_block3_up = PatchExpand((22, 22), 512)

        self.de_block2_up = PatchExpand((44, 44), 512)


        self.norm4 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm1 = nn.LayerNorm(512)

        self.Att3 = CBAM(320, 256)
        self.Att2 = CBAM(128, 256)
        self.Att1 = CBAM(64, 256)

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
                    embed_dim=512,
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
                    embed_dim=512,
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
                    embed_dim=512,
                    num_heads=4,
                    mlp_ratio=2,
                    sr_ratio=2,
                    qkv_bias=True,
                )
                for _ in range(depths[0])
            ]
        )
        self.head = BasicConv2d(512, 32, 1)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        B0, C0, H0, W0 = x.shape
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]  # ([1, 64, 88, 88])
        x2 = pvt[1]  # ([1, 128, 44, 44])
        x3 = pvt[2]  # ([1, 320, 22, 22])
        x4 = pvt[3]  # ([1, 512, 11, 11])

        bridge = self.bridge_attention(x4)  # ([1, 512, 11, 11])

        # decoder block 4
        x = bridge.flatten(2).transpose(1, 2)  # ([1, 121, 512])
        # B, N, C = x.shape

        for i, blk in enumerate(self.decoder_block4):
            # x = blk(x)  # ([1, 121, 512])
            x = checkpoint.checkpoint(blk, x)  # 使用梯度检查点
        # pdb.set_trace()
        x = self.norm4(x)

        # decoder block 3
        x4_up = self.de_block4_up(x)  # ([1, 256, 22, 22])
        attention_3 = self.Att3(x3)  # ([1, 256, 22, 22])

        d3 = torch.cat((x4_up, attention_3), dim=1)  # ([1, 512, 22, 22])
        x = d3.flatten(2).transpose(1, 2)  # ([1, 484, 512])
        for i, blk in enumerate(self.decoder_block3):
            # x = blk(x)  # ([1, 484, 512])
            x = checkpoint.checkpoint(blk, x)  # 使用梯度检查点
        x = self.norm3(x)

        # decoder block 2
        x3_up = self.de_block3_up(x)  # [1, 288, 44, 44]
        attention_2 = self.Att2(x2)  # ([1, 256, 44, 44])
        d2 = torch.cat((x3_up, attention_2), dim=1)  # ([1, 512, 44, 44])
        x = d2.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.decoder_block2):
            # x = blk(x)
            x = checkpoint.checkpoint(blk, x)  # 使用梯度检查点
        x = self.norm2(x)  # ([1, 1936, 416])
        # pdb.set_trace()
        # decoder block 1
        x2_up = self.de_block2_up(x)  # ([1, 208, 88, 88])
        attention_1 = self.Att1(x1)  # ([1, 256, 88, 88])
        d1 = torch.cat((x2_up, attention_1), dim=1)
        x = d1.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.decoder_block1):
            # x = blk(x)  # ([1, 7744, 272])
            x = checkpoint.checkpoint(blk, x)  # 使用梯度检查点
        x = self.norm1(x)
        d1_out = x.permute(0, 2, 1).view(B0, 512, 88, 88)

        d1_out = F.interpolate(d1_out, size=(H0, W0),
                               mode='bilinear', align_corners=ALIGN_CORNERS)

        # # pdb.set_trace()
        head = self.head(d1_out)
        out = self.out(head)

        return out


if __name__ == '__main__':
    device = torch.device('cuda:1')
    model = DuattFormer().to(device)
    input_tensor = torch.randn(1, 3, 352, 352).to(device)
    print("输入尺寸：", input_tensor.shape)

    prediction = model(input_tensor)
    print(prediction.size())
    summary(model, input_size=(1, 3, 352, 352), device=device)