import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_comprise.polyp_pvt.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
代码来源于paper:
https://github.com/DengPingFan/Polyp-PVT/tree/main/lib
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


class PVTEncoderNet_1(nn.Module):
    def __init__(self, channel=32):
        super(PVTEncoderNet_1, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = '/root/nfs/gaobin/wt/PolypSeg_2/model_comprise/polyp_pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.head = BasicConv2d(1024, channel, 1)
        self.out = nn.Conv2d(32, 1, 1)



    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        # pdb.set_trace()
        x0_h, x0_w = x.size(2), x.size(3)
        x1 = F.interpolate(x1, size=(x0_h, x0_w),
                           mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x2, size=(x0_h, x0_w),
                           mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x3, size=(x0_h, x0_w),
                           mode='bilinear', align_corners=ALIGN_CORNERS)
        x4 = F.interpolate(x4, size=(x0_h, x0_w),
                           mode='bilinear', align_corners=ALIGN_CORNERS)

        feats = torch.cat([x1, x2, x3, x4], 1)
        # pdb.set_trace()
        head = self.head(feats)
        out = self.out(head)

        return out


if __name__ == '__main__':
    model = PVTEncoderNet_1().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction = model(input_tensor)
    print(prediction.size())
