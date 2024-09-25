import torch
import torch.nn as nn


class ConvBnActBlockX2(nn.Module):
    """(conv => BN => activation) * 2"""

    def __init__(self, in_ch, out_ch, activation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            activation,
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            activation,
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    """Up Convolution Block"""

    def __init__(self, in_ch, out_ch, activation):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            activation
        )

    def forward(self, x):
        return self.up(x)


# 多任务分割的UNet模型
class UNet_MT(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.ReLU(inplace=True)

        filters = [num_filters, num_filters * 2, num_filters * 4, num_filters * 8, num_filters * 16]

        # Down Sampling
        self.down1 = ConvBnActBlockX2(in_dim, filters[0], activation)
        self.down2 = ConvBnActBlockX2(filters[0], filters[1], activation)
        self.down3 = ConvBnActBlockX2(filters[1], filters[2], activation)
        self.down4 = ConvBnActBlockX2(filters[2], filters[3], activation)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bridge
        self.bridge = ConvBnActBlockX2(filters[3], filters[4], activation)

        # Up Sampling with multiple branches
        self.up1 = UpConv(filters[4], filters[3], activation)
        self.up_conv1 = ConvBnActBlockX2(filters[4], filters[3], activation)
        self.up2 = UpConv(filters[3], filters[2], activation)
        self.up_conv2 = ConvBnActBlockX2(filters[3], filters[2], activation)
        self.up3 = UpConv(filters[2], filters[1], activation)
        self.up_conv3 = ConvBnActBlockX2(filters[2], filters[1], activation)
        self.up4 = UpConv(filters[1], filters[0], activation)
        self.up_conv4 = ConvBnActBlockX2(filters[1], filters[0], activation)

        # Final Convolutions for each branch
        self.final_conv1 = nn.Conv2d(filters[0], out_dim, 1)
        self.final_conv2 = nn.Conv2d(filters[0], out_dim, 1)
        self.final_conv3 = nn.Conv2d(filters[0], out_dim, 1)
        self.active = nn.Sigmoid()

    def forward(self, x):
        # Down-sampling
        d1 = self.down1(x)
        d2 = self.down2(self.max_pool(d1))
        d3 = self.down3(self.max_pool(d2))
        d4 = self.down4(self.max_pool(d3))
        b = self.bridge(self.max_pool(d4))

        # Up-sampling with concatenation
        u1 = self.up_conv1(torch.cat([self.up1(b), d4], dim=1))
        u2 = self.up_conv2(torch.cat([self.up2(u1), d3], dim=1))
        u3 = self.up_conv3(torch.cat([self.up3(u2), d2], dim=1))
        u4 = self.up_conv4(torch.cat([self.up4(u3), d1], dim=1))

        # Generating outputs for each branch
        out_1 = self.final_conv1(u4)
        out_2 = self.final_conv2(u4)
        out_3 = self.final_conv3(u4)

        # Optionally apply activation function
        # out_1 = self.active(out_1)
        # out_2 = self.active(out_2)
        # out_3 = self.active(out_3)

        return out_1, out_2, out_3


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet_MT(in_dim=3, out_dim=1, num_filters=32).to(device)

    # Assuming a batch size of 4 and image dimensions 256x256
    input_tensor = torch.rand(4, 3, 256, 256).to(device)
    out_1, out_2, out_3 = model(input_tensor)
    print("Output sizes:")
    print("Out 1:", out_1.size())
    print("Out 2:", out_2.size())
    print("Out 3:", out_3.size())
