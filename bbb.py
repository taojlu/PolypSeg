import pdb

from torchinfo import summary

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryGRUModule(nn.Module):
    def __init__(self, channel_size, d_model, memory_size, num_heads=1, dropout=0.1):
        super(MemoryGRUModule, self).__init__()
        self.channel_size = channel_size
        self.d_model = d_model
        self.memory_size = memory_size

        # GRU for processing spatial features
        self.gru = nn.GRU(input_size=channel_size, hidden_size=d_model, batch_first=True)
        self.gru_to_channel = nn.Linear(d_model, channel_size)

        # External memory
        self.memory = nn.Parameter(torch.randn(memory_size, d_model), requires_grad=True)
        self.memory_read = nn.Linear(d_model, channel_size)

        # Multi-head Attention
        self.self_attn = nn.MultiheadAttention(embed_dim=channel_size, num_heads=num_heads, batch_first=True)

        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm([11, 11])

    def forward(self, x):
        B, C, H, W = x.size()
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)

        # GRU pass
        gru_out, _ = self.gru(x_flat)
        gru_out = self.gru_to_channel(gru_out)
        gru_out = gru_out.permute(0, 2, 1).view(B, C, H, W)

        # Read from memory
        memory_output = self.memory_read(self.memory).view(self.memory_size, C)
        memory_output = memory_output[:H*W, :].unsqueeze(0).repeat(B, 1, 1)
        memory_output = memory_output.view(B, H, W, C).permute(0, 3, 1, 2)

        # Combine GRU output and memory read
        combined = gru_out + memory_output

        # Self-attention over combined features
        combined_flat = combined.view(B, C, H * W).permute(0, 2, 1)
        attn_output, _ = self.self_attn(combined_flat, combined_flat, combined_flat)
        attn_output = attn_output.permute(0, 2, 1).view(B, C, H, W)
        # pdb.set_trace()
        # Apply normalization and dropout
        attn_output = self.norm(attn_output)
        attn_output = self.dropout(attn_output)

        # Residual connection
        out = attn_output + x

        return out



test_input = torch.randn(2, 512, 11, 11)
memory_gru_module = MemoryGRUModule(channel_size=512, d_model=512, memory_size=256, num_heads=4, dropout=0.1)
output = memory_gru_module(test_input)

# 输出结果
print("Output shape:", output.shape)
print("Output type:", type(output))
summary(memory_gru_module, test_input=(2, 512, 11, 11))
