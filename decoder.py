import torch
import torch.nn as nn
import torch.nn.functional as F
from LGAv2 import Backbone
from timm.models.layers import trunc_normal_
import math
from thop import clever_format, profile
import cv2
import numpy as np


def autopad(k, p=None, d=1):
    '''
    k: kernel
    p: padding
    d: dilation
    '''
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, paddin g, groups, dilation, activation)."""
    default_act = nn.ReLU(inplace=True)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DWConv(Conv):
    """Depth-wise convolution with args(ch_in, ch_out, kernel, stride, dilation, activation)."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                Conv(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                Conv(in_channel, out_channel, k=1, s=1, p=0)
                )

    def forward(self, x):
        return self.conv(x)

class DSConv5x5(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1):
        super(DSConv5x5, self).__init__()
        self.conv = nn.Sequential(
                Conv(in_channel, in_channel, k=5, s=stride, p=2*dilation, d=dilation, g=in_channel),
                Conv(in_channel, out_channel, k=1, s=1, p=0)
                )

    def forward(self, x):
        return self.conv(x)

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvOut(nn.Module):
    def __init__(self, in_channels):
        super(ConvOut, self).__init__()
        self.conv = nn.Sequential(
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channels, 1, 1, stride=1, padding=0),
                nn.Sigmoid()
                )

    def forward(self, x):
        return self.conv(x)

# class DM5(nn.Module):
#     def __init__(self, in_channels, ou_channels):
#         super().__init__()
#
#         self.conv1 = DWConv(in_channels, in_channels, k=3, s=1, d=1)
#         self.pwconv1 = Conv(in_channels, in_channels, 1, 1)
#         self.conv2 = DWConv(in_channels, in_channels, k=3, s=1, d=2)
#         self.pwconv2 = Conv(in_channels, in_channels, 1, 1)
#         self.conv3 = DWConv(in_channels, in_channels, k=3, s=1, d=3)
#         # self.pwconv3 = Conv(in_channels, in_channels, 1, 1)
#         # self.conv4 = DWConv(in_channels, in_channels, k=3, s=1, d=8)
#         self.conv4_ = nn.Conv2d(int(4 * in_channels), ou_channels, 1)
#         self.conv4 = nn.Conv2d(ou_channels, ou_channels, 3, 1, 1)
#         self.conv4b = nn.Conv2d(in_channels, ou_channels, 3, 1, 1)
#         self.conv_out = ConvOut(in_channels=ou_channels)
#         self.conv_outb = ConvOut(in_channels=ou_channels)
#
#     def forward(self, x):
#         out1 = self.conv1(x)
#         out2 = self.conv2(self.pwconv1(x + out1))
#         out3 = self.conv3(self.pwconv2(x + out1 + out2))
#         # out4 = self.conv4(self.pwconv3(x + out3 + out2 + out1))
#         outputb = self.conv4b(x + out1 + out2 + out3)
#         out_salb = self.conv_outb(outputb)
#
#         output_ = self.conv4_(torch.cat((x, out1, out2, out3), dim=1))
#         output = self.conv4(output_ * outputb + output_)
#         out_sal = self.conv_out(output)
#
#         return output, out_sal, outputb, out_salb


class DM5(nn.Module):
    def __init__(self, in_channels, ou_channels):
        super().__init__()

        self.conv1 = nn.Sequential(Conv(in_channels, in_channels, 3, 1, 1), Conv(in_channels, ou_channels, 1, 1, 0), Conv(ou_channels, ou_channels, 3, 1, 1))
        self.conv2 = nn.Sequential(Conv(in_channels, in_channels, 3, 1, 1), Conv(in_channels, ou_channels, 1, 1, 0), Conv(ou_channels, ou_channels, 3, 1, 1))
        self.conv_out = ConvOut(in_channels=ou_channels)
        self.conv_outb = ConvOut(in_channels=ou_channels)

    def forward(self, x):
        output = self.conv1(x)
        outputb = self.conv2(x)
        out_sal = self.conv_out(output)
        out_salb = self.conv_outb(outputb)
        return output, out_sal, outputb, out_salb

def extract_edge_from_binary_mask(tensor_mask):
    B, C, H, W = tensor_mask.size()
    edge_maps = []

    for i in range(B):
        mask_np = tensor_mask[i, 0].detach().cpu().numpy()  # 取出单张 mask
        mask_np = (mask_np * 255).astype(np.uint8)          # 转成 0/255，适配 Canny

        edge = cv2.Canny(mask_np, 10, 100)                 # 提取边缘
        edge_maps.append(torch.from_numpy(edge).float() / 255.0)  # 转回 tensor，并归一化

    edge_tensor = torch.stack(edge_maps).unsqueeze(1).to(tensor_mask.device)  # (B, 1, H, W)
    return edge_tensor

class DM(nn.Module):
    def __init__(self, in_channels, ou_channels):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = nn.ReLU(inplace=True)
        self.reduced_channels = in_channels // 16
        self.fc1 = nn.Conv2d(in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.fc1b = nn.Conv2d(in_channels, self.reduced_channels, 1, bias=False)
        self.fc2b = nn.Conv2d(self.reduced_channels, in_channels, 1, bias=False)

        self.conv1 = Conv(in_channels, in_channels, 1)

        self.conv2 = Conv(in_channels, in_channels, 1)
        self.conv3 = DWConv(2 * in_channels, in_channels, 3, 1, 1)

        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=int(2 * in_channels), data_format='channels_first'),
            nn.Conv2d(int(2 * in_channels), int(2 * in_channels), kernel_size=3, stride=1,
                      padding=1, dilation=1, groups=int(2 * in_channels)),
            nn.Conv2d(int(2 * in_channels), int(0.5 * in_channels), 1))
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=int(2 * in_channels), data_format='channels_first'),
            nn.Conv2d(int(2 * in_channels), int(2 * in_channels), kernel_size=3, stride=1,
                      padding=1, dilation=1, groups=int(2 * in_channels)),
            nn.Conv2d(int(2 * in_channels), int(0.5 * in_channels), 1))

        self.conv4 = Conv(in_channels, ou_channels, 1)
        self.conv4b = Conv(in_channels, ou_channels, 1)
        self.conv5 = DWConv(2 * ou_channels, ou_channels, 3)
        self.conv_out = ConvOut(in_channels=ou_channels)
        self.conv_outb = ConvOut(in_channels=ou_channels)

    def forward(self, r_b, sal, b, sal_b, r):
        B, C, H, W = r.size()

        avg_pool_out = self.avg_pool(r_b)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))
        max_pool_out = self.max_pool(r_b)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))
        out = self.sigmoid(avg_out + max_out)
        rc = r * out

        featurer = r_b * sal
        y2 = F.interpolate(featurer, (H, W), mode='bilinear', align_corners=True)
        ra = self.conv1(rc * y2)

        y1 = F.interpolate(r_b, (H, W), mode='bilinear', align_corners=True)


        avg_pool_outb = self.avg_pool(b)
        avg_outb = self.fc2b(self.activation(self.fc1b(avg_pool_outb)))
        max_pool_outb = self.max_pool(b)
        max_outb = self.fc2b(self.activation(self.fc1b(max_pool_outb)))
        outb = self.sigmoid(avg_outb + max_outb)
        bc = r * outb

        featureb = b * sal_b
        y2b = F.interpolate(featureb, (H, W), mode='bilinear', align_corners=True)
        ba = self.conv2(bc * y2b)
        y1b = F.interpolate(b, (H, W), mode='bilinear', align_corners=True)
        fusionb = self.conv3(torch.cat([y1b, ba], dim=1))
        outputb = self.conv4b(fusionb)
        out_salb = self.conv_outb(outputb)

        y1c = torch.chunk(y1, 2, dim=1)
        rac = torch.chunk(ra, 2, dim=1)
        rc = torch.chunk(r, 2, dim=1)
        fusionbc = torch.chunk(fusionb, 2, dim=1)
        y3 = self.g0(torch.cat((y1c[0], rac[0], rc[0], fusionbc[0]), dim=1))
        y4 = self.g1(torch.cat((y1c[1], rac[1], rc[1], fusionbc[1]), dim=1))

        output_ = self.conv4(torch.cat((y3, y4), dim=1) + r)
        output = self.conv5(torch.cat((outputb, output_), dim=1)) + output_
        out_sal = self.conv_out(output)

        return output, out_sal, outputb, out_salb

class self_net(nn.Module):
    def __init__(self, inchannel=3):
        super(self_net, self).__init__()
        self.partnet = Backbone(c=inchannel)
        self.rd5 = DM5(128, 96)
        self.rd4 = DM(96, 64)
        self.rd3 = DM(64, 32)
        self.rd2 = DM(32, 16)
        self.rd1 = DM(16, 16)

        # self.convb1_1 = nn.Conv2d(16, 1, 1, stride=1, padding=0)
        # self.convb2_2 = nn.Conv2d(32, 1, 1, stride=1, padding=0)
        # self.convb2_3 = nn.Conv2d(32, 1, 1, stride=1, padding=0)
        self.out = nn.Sequential(nn.Dropout2d(p=0.1), nn.Conv2d(16, 1, 1, stride=1, padding=0))

        self.sig = nn.Sigmoid()

        self.apply(self._init_weights)

    def forward(self, input, epoch, hw_range):
        b, c, h, w = input.size()
        p_f = self.partnet(input, epoch, hw_range)

        F5, F5_sal, b5, b5_sal = self.rd5(p_f[4])

        F4, F4_sal, b4, b4_sal = self.rd4(F5, F5_sal, b5, b5_sal, p_f[3])

        F3, F3_sal, b3, b3_sal = self.rd3(F4, F4_sal, b4, b4_sal, p_f[2])

        F2, F2_sal, b2, b2_sal = self.rd2(F3, F3_sal, b3, b3_sal, p_f[1])

        F1, F1_sal, b1, b1_sal = self.rd1(F2, F2_sal, b2, b2_sal, p_f[0])

        # b1_1 = self.sig(interpolate(self.convb1_1(p_f[5]), input.size()[2:]))
        #
        # b2_2 = self.sig(interpolate(self.convb2_2(p_f[6]), input.size()[2:]))
        #
        # b2_3 = self.sig(interpolate(self.convb2_3(p_f[7]), input.size()[2:]))

        out1 = self.sig(interpolate(self.out(F1), input.size()[2:]))

        # out1 = interpolate(F1_sal, input.size()[2:])
        out2 = interpolate(F2_sal, input.size()[2:])
        out3 = interpolate(F3_sal, input.size()[2:])
        out4 = interpolate(F4_sal, input.size()[2:])
        out5 = interpolate(F5_sal, input.size()[2:])

        b1 = interpolate(b1_sal, input.size()[2:])
        b2 = interpolate(b2_sal, input.size()[2:])
        b3 = interpolate(b3_sal, input.size()[2:])
        b4 = interpolate(b4_sal, input.size()[2:])
        b5 = interpolate(b5_sal, input.size()[2:])

        return out1, out2, out3, out4, out5, b1, b2, b3, b4, b5

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)

if __name__ == '__main__':
    # 创建模型实例
    model = self_net().cuda()

    # 假设输入的形状为 (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 352, 352).cuda()  # 输入一个 batch 的数据

    # 使用 profile 函数计算 FLOPs
    flops, params = profile(model, inputs=(input_tensor, 200, [0, 18]))  # epoch=1, hw_range=[0,18]

    print('GFLOPs : {:.2f} G'.format(flops / 1e9))
    print('Params : {:.2f} M'.format(params / 1e6))