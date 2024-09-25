import torch
import torch.nn as nn
import torch.nn.functional as F

class IntegratedConvBnActBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation, levels=3):
        super(IntegratedConvBnActBlock, self).__init__()
        self.levels = levels
        # 主路径
        self.main_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            activation(),
        )
        # 多路径与多尺度
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch // levels, 3, padding=1, stride=2**i, bias=True),
                nn.BatchNorm2d(out_ch // levels),
                activation(),
                nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=True)
            ) for i in range(levels)
        ])
        # 附加的1x1卷积以处理跨路径特征整合
        self.integration_conv = nn.Sequential(
            nn.Conv2d(out_ch + (out_ch // levels) * levels, out_ch, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            activation(),
        )

    def forward(self, x):
        main = self.main_path(x)
        features = [main] + [path(x) for path in self.paths]
        integrated = torch.cat(features, dim=1)
        result = self.integration_conv(integrated)
        return result

# 实例化模块，输入通道为3，输出通道为8，使用ReLU激活函数
model = IntegratedConvBnActBlock(in_ch=3, out_ch=8, activation=nn.ReLU, levels=3)

# 创建一个假设的输入张量，尺寸为 [1, 3, 64, 64]，表示1张3通道的64x64图像
input_tensor = torch.randn(1, 3, 64, 64)

# 通过模型传递输入并打印输出
output = model(input_tensor)
print("Output shape:", output.shape)  # 应该得到 [1, 8, 64, 64]
