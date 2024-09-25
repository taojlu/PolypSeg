import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
from timm.models.registry import register_model

# Patch Embedding 模块

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
        # x = self.bn(x)
        return x
class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.permute(0, 2, 3, 1)  # (B, H/P, W/P, embed_dim)
        if self.norm:
            x = self.norm(x)
        return x

# Patch Merging 模块
class PatchMerging2D(nn.Module):
    def __init__(self, input_dim, output_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reduction = nn.Linear(4 * input_dim, output_dim, bias=False)
        self.norm = norm_layer(4 * input_dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape

        # 如果 H 或 W 为奇数，需要进行填充
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B, H/2, W/2, C
        x1 = x[:, 1::2, 0::2, :]  # B, H/2, W/2, C
        x2 = x[:, 0::2, 1::2, :]  # B, H/2, W/2, C
        x3 = x[:, 1::2, 1::2, :]  # B, H/2, W/2, C

        x = torch.cat([x0, x1, x2, x3], dim=-1)  # B, H/2, W/2, 4*C
        x = self.norm(x)
        x = self.reduction(x)  # B, H/2, W/2, output_dim

        return x

# Selective Scan 2D (Mamba核心模块)
class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dropout=0., bias=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, groups=self.d_inner, bias=True,
                                kernel_size=d_conv, padding=(d_conv - 1) // 2)
        self.act = nn.SiLU()

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    def forward(self, x):
        B, H, W, C = x.shape

        # 线性投影
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (B, H, W, d_inner)

        x = x.permute(0, 3, 1, 2).contiguous()  # (B, d_inner, H, W)
        x = self.act(self.conv2d(x))  # 卷积处理

        # 处理完成后重新调整输出形状
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, d_inner)
        x = self.out_proj(x)  # (B, H, W, d_model)

        if self.dropout is not None:
            x = self.dropout(x)

        return x

# Pyramid Vision Mamba 网络实现
class PyramidVisionMamba(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 depths=[3, 4, 6, 3], dims=[64, 128, 256, 512],
                 d_state=16, drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.dims = dims

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=dims[0], norm_layer=norm_layer)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        self.patch_merging_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = nn.ModuleList()
            # 每个阶段的块数
            depth = depths[i_layer]
            # 输入和输出维度
            dim = dims[i_layer]

            for i_block in range(depth):
                drop_path = dpr[sum(depths[:i_layer]) + i_block]
                block = SS2D(
                    d_model=dim,
                    d_state=d_state,
                    dropout=drop_rate,
                    bias=False,
                )
                layer.append(block)

            self.layers.append(layer)

            # 如果不是最后一层，添加 Patch Merging 层
            if i_layer < self.num_layers - 1:
                merging = PatchMerging2D(input_dim=dims[i_layer], output_dim=dims[i_layer + 1], norm_layer=norm_layer)
                self.patch_merging_layers.append(merging)

        self.conv = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.head = BasicConv2d(512, 32, 1)
        self.out = nn.Conv2d(32, 1, 1)


    def forward(self, x):
        x = self.patch_embed(x)

        for i_layer in range(self.num_layers):
            # 当前阶段的块
            layer = self.layers[i_layer]
            for block in layer:
                x = block(x)

            # 如果不是最后一层，执行 Patch Merging
            if i_layer < self.num_layers - 1:
                x = self.patch_merging_layers[i_layer](x)

        # 全局平均池化并分类
        # x = x.mean(dim=[1, 2])  # (B, C)
        # x = nn.Linear(self.dims[-1], self.num_classes)(x)
        x = x.permute(0,3,1,2).contiguous()


        head = self.head(x)
        out = self.out(head)
        x = F.interpolate(out, size=(352, 352),
                           mode='bilinear', align_corners=True)
        return x

# 注册模型
@register_model
class pvm_v2_b0(PyramidVisionMamba):
    def __init__(self, **kwargs):
        super(pvm_v2_b0, self).__init__(
            patch_size=4, dims=[32, 64, 160, 256], depths=[2, 2, 2, 2], d_state=16,
            drop_rate=0.0, drop_path_rate=0.1, num_classes=1000, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

@register_model
class pvm_v2_b1(PyramidVisionMamba):
    def __init__(self, **kwargs):
        super(pvm_v2_b1, self).__init__(
            patch_size=4, dims=[64, 128, 320, 512], depths=[2, 2, 2, 2], d_state=32,
            drop_rate=0.0, drop_path_rate=0.1, num_classes=1000, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

# 测试代码
if __name__ == "__main__":
    # 定义输入的图像大小和模型参数
    input_size = (1, 3, 352, 352)  # 批次为1，3通道RGB图像，224x224分辨率

    # 创建测试输入
    x = torch.randn(input_size)

    # 加载最简单配置的 Pyramid Vision Mamba 模型 (pvm_v2_b0)
    model = pvm_v2_b1()  # 使用默认的参数初始化模型

    # 将模型置于评估模式（不需要训练时的 dropout 等操作）
    model.eval()

    # 前向传播测试，确保模型能够处理输入并产生输出
    with torch.no_grad():
        output = model(x)
    print("输出形状:", output.shape)
