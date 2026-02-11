import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from thop import clever_format, profile
from einops import rearrange



#   Channel attention block (CAB)
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return x * self.sigmoid(out)

#   Spatial attention block (SAB)
class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

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

class StdPool(nn.Module):
    def __init__(self):
        super(StdPool, self).__init__()

    def forward(self, x):
        b, c, _, _ = x.size()

        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        std = std.reshape(b, c, 1, 1)

        return std

class Pool(nn.Module):
    def __init__(self, pool_types=['avg', 'max', 'std']):
        """Constructs a Pool module.
        Args:
            k_size: kernel size
            pool_types: pooling type. 'avg': average pooling, 'max': max pooling, 'std': standard deviation pooling.
        """
        super(Pool, self).__init__()

        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool2d(1))
            elif pool_type == 'std':
                self.pools.append(StdPool())
            else:
                raise NotImplementedError

        # self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)

        self.weight = nn.Parameter(torch.rand(3))
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feats = [pool(x) for pool in self.pools]

        if len(feats) == 2:
            # print("Pool-2")
            weight = torch.sigmoid(self.weight)
            out = weight[0] * feats[0] + weight[1] * feats[1]
        elif len(feats) == 3:
            # print("Pool-3")
            weight = torch.sigmoid(self.weight)
            out = weight[0] * feats[0] + weight[1] * feats[1] + weight[2] * feats[2]
        else:
            assert False, "Feature Extraction Exception!"

        return out


class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    # (1, 3, 6, 8)
    # (1, 4, 8,12)
    def __init__(self, grids=(7, 5, 3, 1), channels=16):
        super(PSPModule, self).__init__()

        self.grids = grids
        self.channels = channels

    def forward(self, feats):
        b, c, h, w = feats.size()

        ar = w / h

        return torch.cat([
            F.adaptive_max_pool2d(feats, (self.grids[0], max(1, round(ar * self.grids[0])))).view(b, self.channels, -1),
            F.adaptive_max_pool2d(feats, (self.grids[1], max(1, round(ar * self.grids[1])))).view(b, self.channels, -1),
            F.adaptive_max_pool2d(feats, (self.grids[2], max(1, round(ar * self.grids[2])))).view(b, self.channels, -1),
            F.adaptive_max_pool2d(feats, (self.grids[3], max(1, round(ar * self.grids[3])))).view(b, self.channels, -1)
        ], dim=2)



class FDB(nn.Module):
    def __init__(self, inter_channels=16, grids=(7, 5, 3, 1)):  # 先ce后ffm
        super(FDB, self).__init__()
        self.grids = grids
        self.inter_channels = inter_channels

        self.query_conva = nn.Conv2d(in_channels=inter_channels, out_channels=inter_channels, kernel_size=1)
        self.query_convc = nn.Conv2d(in_channels=inter_channels, out_channels=inter_channels, kernel_size=1)

        self.key_conva = nn.Conv1d(in_channels=inter_channels, out_channels=inter_channels, kernel_size=1)
        self.value_conva = nn.Conv1d(in_channels=inter_channels, out_channels=inter_channels, kernel_size=1)
        self.key_pspa = PSPModule(grids, inter_channels)
        self.value_pspa = PSPModule(grids, inter_channels)

        self.key_convc = nn.Conv1d(in_channels=inter_channels, out_channels=inter_channels, kernel_size=1)
        self.value_convc = nn.Conv1d(in_channels=inter_channels, out_channels=inter_channels, kernel_size=1)
        self.key_pspc = PSPModule(grids, inter_channels)
        self.value_pspc = PSPModule(grids, inter_channels)

        self.softmax = nn.Softmax(dim=-1)

        self.a = nn.Conv2d(in_channels=inter_channels, out_channels=inter_channels, kernel_size=1)
        self.c = nn.Conv2d(in_channels=inter_channels, out_channels=inter_channels, kernel_size=1)

    def forward(self, x, a, c):

        m_batchsize, m_c, h, w = x.size()

        querya = self.query_conva(x).view(m_batchsize, m_c, -1).permute(0, 2, 1)  ##  b c n ->  b n c
        queryc = self.query_convc(x).view(m_batchsize, m_c, -1).permute(0, 2, 1)  ##  b c n ->  b n c

        keya = self.key_conva(self.key_pspa(a))  ## b c s
        sim_mapa = torch.matmul(querya, keya)
        sim_mapa = self.softmax(sim_mapa)
        valuea = self.value_conva(self.value_pspa(a))
        contexta = torch.bmm(valuea, sim_mapa.permute(0, 2, 1))
        contexta = self.a(contexta.view(m_batchsize, self.inter_channels, h, w))

        keyc = self.key_convc(self.key_pspc(c))  ## b c s
        sim_mapc = torch.matmul(queryc, keyc)
        sim_mapc = self.softmax(sim_mapc)
        valuec = self.value_convc(self.value_pspc(c))
        contextc = torch.bmm(valuec, sim_mapc.permute(0, 2, 1))
        contextc = self.c(contextc.view(m_batchsize, self.inter_channels, h, w))

        cb = contexta * c + c
        ab = contextc * a + a


        out = x + ab + cb

        return out

def channel_shuffle(x, groups):
    """
    Perform channel shuffle operation on the input tensor.

    Parameters:
    - x: input tensor of shape (batch_size, channels, height, width)
    - groups: number of groups to split the channels into

    Returns:
    - output tensor with shuffled channels
    """
    batch_size, channels, height, width = x.size()

    # Ensure that the channels are divisible by the number of groups
    assert channels % groups == 0, 'Channels must be divisible by groups.'

    # Split channels into groups
    channels_per_group = channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # Shuffle the groups of channels
    x = x.permute(0, 2, 1, 3, 4).contiguous()

    # Combine the groups back into a single tensor
    x = x.view(batch_size, channels, height, width)

    return x


class BDF(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.conv1 = DWConv(c, c, k=3, s=1, d=1)
        self.conv2 = DWConv(c, c, k=3, s=1, d=2)
        self.conv3 = DWConv(c, c, k=3, s=1, d=3)

        lambd = 1.0
        gamma = 1.0
        temp = round(abs((math.log2(c) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1
        self.pool = Pool()
        self.conv = nn.Sequential(nn.Conv1d(1, 1, kernel_size=kernel, padding=(kernel - 1) // 2, stride=2, bias=False),
                                  nn.Conv1d(1, 1, kernel_size=kernel, padding=(kernel - 1) // 2, stride=2, bias=False))

        self.group_compress = nn.ModuleList([
            nn.Conv2d(4, 1, kernel_size=1) for _ in range(c)
        ])
        self.fc = nn.Linear(c, c)

        self.g = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1),
            nn.BatchNorm2d(c),
            nn.Sigmoid(),
        )

        self.group_compress_spacial = nn.ModuleList([
            nn.Conv2d(4, 1, kernel_size=1) for _ in range(c)
        ])
        self.pw = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.sa = SAB()

        self.pw2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(c, c, (1, 3), padding=(0, 1), groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True))
        self.conv31 = nn.Sequential(
            nn.Conv2d(c, c, (3, 1), padding=(1, 0), groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True))
        self.pw3 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True))

        self.conv13_2 = nn.Sequential(
            nn.Conv2d(c, c, (1, 3), padding=(0, 1), groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True))
        self.conv31_2 = nn.Sequential(
            nn.Conv2d(c, c, (3, 1), padding=(1, 0), groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True))
        self.pw4 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True))

        self.conv13_3 = nn.Sequential(
            nn.Conv2d(c, c, (1, 3), padding=(0, 1), groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True))
        self.conv31_3 = nn.Sequential(
            nn.Conv2d(c, c, (3, 1), padding=(1, 0), groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True))
        self.pw5 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True))

        self.pw6 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        # self.bconv = nn.Conv2d(c, c, kernel_size=1)
        # self.bnorm = nn.BatchNorm2d(c)
        # self.bs = nn.Sigmoid()
        # self.boundary = nn.Sequential(
        #     nn.Conv2d(c, c, kernel_size=1),
        #     nn.BatchNorm2d(c),
        #     nn.Sigmoid())
        #
        # self.p_conv = nn.Sequential(
        #     nn.Conv2d(c, 1, kernel_size=1),
        #     nn.BatchNorm2d(1),
        #     nn.Sigmoid())

        self.attention = FDB(inter_channels=c)

        self.out = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # B, C, H, W = x.shape
        res = x

        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)

        combined = torch.cat((x, out1, out2, out3), dim=1)
        pooled = self.pool(combined)
        pooled = channel_shuffle(pooled, 4)
        # print('pool', pooled.shape)
        y_local = self.conv(pooled.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # print('local', y_local.shape)
        B, C, H, W = pooled.shape  # pooled 是 channel shuffle 后的特征
        assert C % 4 == 0, "C 必须能被 4 整除"
        groups = torch.chunk(pooled, C // 4, dim=1)  # 划分为 C//4 组，每组 [B, 4, H, W]
        compressed = [conv(g) for conv, g in zip(self.group_compress, groups)]
        y_global = torch.cat(compressed, dim=1)  # 拼接回 [B, C, H, W]
        y_global = self.fc(torch.flatten(y_global, 1)).unsqueeze(-1).unsqueeze(-1)
        # print('global', y_global.shape)
        y = self.g(y_local + y_global)
        c = x * y

        spacial = channel_shuffle(combined, 4)
        groups_spacial = torch.chunk(spacial, C // 4, dim=1)  # 划分为 C//4 组，每组 [B, 4, H, W]
        compressed_spacial = [conv(g) for conv, g in zip(self.group_compress_spacial, groups_spacial)]
        combined2 = self.pw(torch.cat(compressed_spacial, dim=1))  # 拼接回 [B, C, H, W]

        # combined2 = self.pw(combined)
        y2 = self.sa(combined2)
        a = x * y2

        combined3 = x + out1 + out2 + out3
        ca = combined3 * y * y2
        d = self.pw2(ca)

        d3 = self.pw3(self.conv13(d) + self.conv31(d))
        d5 = self.pw4(self.conv13_2(d3+d) + self.conv31_2(d3+d))
        d7 = self.pw5(self.conv13_3(d5 + d3 + d) + self.conv31_3(d5 + d3 + d))
        yb = self.pw6(d + d3 + d5 + d7 + combined3)

        # yb = self.boundary(yb1)
        # log_yb = torch.log2(yb + 1e-10)
        # entropy = -1 * yb * log_yb + yb1
        # att = entropy + a + c

        att = self.attention(yb, a, c)

        out = self.out(att) + res

        return out, yb


if __name__ == '__main__':
    x = torch.randn(2, 16, 178, 178).cuda()
    model = BDF(16).cuda()
    output, yb = model(x)
    print(output.shape)