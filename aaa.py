import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchinfo import summary

class DSSAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, sr_ratio, qkv_bias=False, attn_dropout=0.0, proj_dropout=0.0,
                 block_size=16, rank=32):
        super(DSSAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.block_size = block_size
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

    def generate_sparse_mask(self, N, num_heads, block_size):
        mask = torch.zeros(num_heads, N, N)
        for i in range(num_heads):
            for j in range(0, N, block_size):
                end = min(j + block_size, N)
                mask[i, j:end, j:end] = torch.tril(torch.ones(end - j, end - j))
        return mask

    def forward(self, x):
        B, N, C = x.shape
        q = self.query(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.key_value(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))

        sparse_attn_mask = self.generate_sparse_mask(N, self.num_heads, self.block_size).to(x.device)
        attn = F.softmax(attn, dim=-1) * sparse_attn_mask
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


# 假设的输入：批次大小=1，序列长度=121，特征维度=512
input_tensor = torch.rand(1, 121, 512)

# 定义模型参数
embed_dim = 512  # 嵌入维度
num_heads = 8  # 注意力头数
sr_ratio = 1  # 下采样率，这里假设不进行下采样
block_size = 16  # 稀疏注意力块的大小

# 初始化注意力模块
attention_module = DSSAttention(embed_dim=embed_dim, num_heads=num_heads, sr_ratio=sr_ratio,
                                         block_size=block_size)

# 前向传播
output = attention_module(input_tensor)

# 打印输出
print("输出形状:", output.shape)

summary(attention_module, input_size=(1, 121, 512))
