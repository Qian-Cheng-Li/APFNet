import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from thop import clever_format, profile
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from local_Conv import ARConv
from MFE_v4 import BDF


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
        return x * self.sigmoid(out)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class eca_layer(nn.Module):
    """
    Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
        source: https://github.com/BangguWu/ECANet
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

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

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
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


class LinearSelfAttention(nn.Module):
    def __init__(self, embed_dim, attn_dropout=0):
        super().__init__()
        self.num_heads = embed_dim // 8
        self.head_dim = embed_dim // self.num_heads  # 每个头的通道数
        self.qkv_proj = nn.Conv2d(embed_dim, 3 * embed_dim, kernel_size=1, bias=True)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.qkv_proj2 = nn.Conv2d(embed_dim, 3 * embed_dim, kernel_size=1, bias=True)

        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True)
        self.embed_dim = embed_dim
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, split_size_or_sections=[self.embed_dim, self.embed_dim, self.embed_dim], dim=1)

        # 将q, k, v 维度重新调整为 [batch_size, num_heads, head_dim, height, width]
        q = q.view(batch_size, self.num_heads, self.head_dim, height, width).permute(0, 1, 4, 3, 2)
        k = k.view(batch_size, self.num_heads, self.head_dim, height, width).permute(0, 1, 4, 3, 2)
        v = v.view(batch_size, self.num_heads, self.head_dim, height, width).permute(0, 1, 4, 3, 2)

        attn = q @ k.transpose(-2, -1) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(batch_size, in_channels, height, width)

        qkv2 = self.qkv_proj2(x_weighted)
        q2, k2, v2 = torch.split(qkv2, split_size_or_sections=[self.embed_dim, self.embed_dim, self.embed_dim], dim=1)

        # 将q2, k2, v2 维度重新调整为 [batch_size, num_heads, head_dim, height, width]
        q2 = q2.view(batch_size, self.num_heads, self.head_dim, height, width)
        k2 = k2.view(batch_size, self.num_heads, self.head_dim, height, width)
        v2 = v2.view(batch_size, self.num_heads, self.head_dim, height, width)

        context_score = F.softmax(q2, dim=-1)
        context_score = self.attn_dropout(context_score)

        context_vector = k2 * context_score
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        out = F.relu(v2) * context_vector.expand_as(v2)
        out = out.view(batch_size, in_channels, height, width)
        out = self.out_proj(out)
        return out


class LinearAttnFFN(nn.Module):
    def __init__(self, embed_dim, ffn_latent_dim, dropout=0, attn_dropout=0):
        super().__init__()
        self.pre_norm_attn = nn.Sequential(
            LayerNorm(embed_dim),
            LinearSelfAttention(embed_dim, attn_dropout),
            nn.Dropout(dropout)
        )
        self.pre_norm_ffn = nn.Sequential(
            LayerNorm(embed_dim),
            nn.Conv2d(embed_dim, ffn_latent_dim, kernel_size=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(ffn_latent_dim, embed_dim, kernel_size=1, stride=1, bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # self attention
        x = x + self.pre_norm_attn(x)
        # Feed Forward network
        x = x + self.pre_norm_ffn(x)
        return x


class InvertedResidualMSD(nn.Module):
    def __init__(self, inp, oup, expand_ratio=4):
        super(InvertedResidualMSD, self).__init__()

        hidden_dim = int(round(inp * expand_ratio))

        # 1x1 expansion conv
        self.expansion = Conv(inp, hidden_dim, k=1, s=1, p=0)

        # Depthwise convs with different dilation rates
        splits = hidden_dim // 4
        self.dw_convs = nn.ModuleList([
            Conv(splits, splits, k=3, s=1, p=d, g=splits, d=d) for d in [1, 2, 3, 4]
        ])

        self.res_convs = nn.ModuleList([
            Conv(splits, splits, k=1, s=1, p=0) for _ in range(4)
        ])

        # Channel compression convs for each group output
        self.fuse_convs = nn.ModuleList([
            nn.Conv2d(splits, oup // 4, kernel_size=1, bias=False) for _ in range(4)
        ])

    def forward(self, x):
        out = self.expansion(x)  # [B, 4C, H, W]
        splits = torch.chunk(out, 4, dim=1)  # → 4 blocks, each [B, C, H, W]

        dw_outs = []
        for i, conv in enumerate(self.dw_convs):
            dw_out = conv(splits[i])
            dw_outs.append(torch.chunk(dw_out, 4, dim=1))  # Each → 4 parts: [B, C/4, H, W]

        # 拼接不同 dilation 输出中同一位置的 4 份，得到 4 个组，每个组 [B, C, H, W]
        groups = [torch.cat([dw_outs[j][i] for j in range(4)], dim=1) for i in range(4)]

        outputs = []
        for i in range(4):
            group_i = groups[i]  # [B, C, H, W]
            residual_i = x
            # if residual_i.shape[2:] != group_i.shape[2:]:
            #     residual_i = F.interpolate(residual_i, size=group_i.shape[2:], mode='bilinear', align_corners=True)
            group_i = self.res_convs[i](group_i)
            group_i = group_i + residual_i
            output_i = self.fuse_convs[i](group_i)  # → [B, oup//4, H, W]
            outputs.append(output_i)

        return outputs  # 输出为 List[Tensor], 每个 shape: [B, oup//4, H, W]


class CAF(nn.Module):
    def __init__(self, in_dim, bias=True):
        super(CAF, self).__init__()
        self.chanel_in = in_dim

        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d(self.chanel_in,  self.chanel_in*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in*3, self.chanel_in*3, kernel_size=3, stride=1, padding=1,
                                    groups=self.chanel_in*3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()

        x_input = x.view(m_batchsize, N*C, height, width)
        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(m_batchsize, N, -1)
        k = k.view(m_batchsize, N, -1)
        v = v.view(m_batchsize, N, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)
        out_1 = out_1.view(m_batchsize, -1, height, width)

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(m_batchsize, N, C, height, width)

        out = out_1+x
        out = out.view(m_batchsize, -1, height, width)
        return out


class OTBlock(nn.Module):
    def __init__(self, inp, ffn_multiplier, attn_blocks):
        super(OTBlock, self).__init__()

        # local representation
        self.local_rep = InvertedResidualMSD(inp, inp)
        # global representation
        self.global_rep1 = nn.Sequential()
        ffn_dims = [int((ffn_multiplier * int(0.25 * inp)) // 16 * 16)] * attn_blocks
        for i in range(attn_blocks):
            ffn_dim = ffn_dims[i]
            self.global_rep1.add_module(f'LinearAttnFFN_{i}', LinearAttnFFN(int(0.25 * inp), ffn_dim))
        self.global_rep1.add_module('LayerNorm2D', LayerNorm(int(0.25 * inp)))

        self.global_rep2 = nn.Sequential()
        ffn_dims = [int((ffn_multiplier * int(0.25 * inp)) // 16 * 16)] * attn_blocks
        for i in range(attn_blocks):
            ffn_dim = ffn_dims[i]
            self.global_rep2.add_module(f'LinearAttnFFN_{i}', LinearAttnFFN(int(0.25 * inp), ffn_dim))
        self.global_rep2.add_module('LayerNorm2D', LayerNorm(int(0.25 * inp)))

        self.global_rep3 = nn.Sequential()
        ffn_dims = [int((ffn_multiplier * int(0.25 * inp)) // 16 * 16)] * attn_blocks
        for i in range(attn_blocks):
            ffn_dim = ffn_dims[i]
            self.global_rep3.add_module(f'LinearAttnFFN_{i}', LinearAttnFFN(int(0.25 * inp), ffn_dim))
        self.global_rep3.add_module('LayerNorm2D', LayerNorm(int(0.25 * inp)))

        self.global_rep4 = nn.Sequential()
        ffn_dims = [int((ffn_multiplier * int(0.25 * inp)) // 16 * 16)] * attn_blocks
        for i in range(attn_blocks):
            ffn_dim = ffn_dims[i]
            self.global_rep4.add_module(f'LinearAttnFFN_{i}', LinearAttnFFN(int(0.25 * inp), ffn_dim))
        self.global_rep4.add_module('LayerNorm2D', LayerNorm(int(0.25 * inp)))

        self.caf = CAF(in_dim=inp)

        self.conv_proj = Conv(2 * inp, inp, k=1, s=1, p=0, act=True)

    def unfolding_pytorch(self, feature_map, H, W):
        batch_size, in_channels, img_h, img_w = feature_map.shape
        # 计算需要填充的高度和宽度
        pad_h = (H - img_h % H) % H
        pad_w = (W - img_w % W) % W
        # 进行填充操作
        feature_map = F.pad(feature_map, (0, pad_w, 0, pad_h))
        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(H, W),
            stride=(H, W),
        )
        patches = patches.reshape(
            batch_size, in_channels, H * W, -1
        )
        return patches, (img_h, img_w), (pad_h, pad_w)

    def folding_pytorch(self, patches, output_size, padding, H, W):
        batch_size, in_dim, patch_size, n_patches = patches.shape
        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)
        pad_h, pad_w = padding
        feature_map = F.fold(
            patches,
            output_size=(output_size[0] + pad_h, output_size[1] + pad_w),
            kernel_size=(H, W),
            stride=(H, W),
        )
        # 去除填充部分
        feature_map = feature_map[:, :, :output_size[0], :output_size[1]]
        return feature_map

    def forward(self, x, epoch, hw_range):

        fm_loc = self.local_rep(x)

        x1, output_size1, padding1 = self.unfolding_pytorch(fm_loc[0], int(2), int(2))
        x2, output_size2, padding2 = self.unfolding_pytorch(fm_loc[1], int(3), int(3))
        x3, output_size3, padding3 = self.unfolding_pytorch(fm_loc[2], int(4), int(4))
        x4, output_size4, padding4 = self.unfolding_pytorch(fm_loc[3], int(5), int(5))

        x1 = self.global_rep1(x1)
        x1 = self.folding_pytorch(patches=x1, output_size=output_size1, padding=padding1, H=int(2), W=int(2))
        x1 = x1 + fm_loc[0]

        x2 = self.global_rep2(x2)
        x2 = self.folding_pytorch(patches=x2, output_size=output_size2, padding=padding2, H=int(3), W=int(3))
        x2 = x2 + fm_loc[1]

        x3 = self.global_rep3(x3)
        x3 = self.folding_pytorch(patches=x3, output_size=output_size3, padding=padding3, H=int(4), W=int(4))
        x3 = x3 + fm_loc[2]

        x4 = self.global_rep4(x4)
        x4 = self.folding_pytorch(patches=x4, output_size=output_size4, padding=padding4, H=int(5), W=int(5))
        x4 = x4 + fm_loc[3]

        fusion = torch.cat([x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1), x4.unsqueeze(1)], dim=1)

        out = self.caf(fusion) + x

        return out


class Backbone(nn.Module):
    def __init__(self, c=3):

        super().__init__()

        # model size
        channels = [16, 32, 64, 96, 128]
        # channels = [32, 64, 128, 192, 256]
        # channels = [32, 64, 128, 256, 512]


        # default shown in paper
        ffn_multiplier = 4

        self.layer_1_1 = Conv(c, channels[0], k=3, s=2, p=1)
        self.layer_1_2 = BDF(c=channels[0])

        self.layer_2_1 = DSConv3x3(channels[0], channels[1], stride=2)
        self.layer_2_2 = BDF(c=channels[1])
        self.layer_2_3 = BDF(c=channels[1])

        self.layer_3_1 = DSConv3x3(channels[1], channels[2], stride=2)
        self.layer_3_2 = OTBlock(channels[2], ffn_multiplier, 4)

        self.layer_4_1 = DSConv3x3(channels[2], channels[3], stride=2)
        self.layer_4_2 = OTBlock(channels[3], ffn_multiplier, 6)

        self.layer_5_1 = DSConv3x3(channels[3], channels[4], stride=2)
        self.layer_5_2 = OTBlock(channels[4], ffn_multiplier, 3)

    def forward(self, x, epoch, hw_range):
        x1_1 = self.layer_1_1(x)
        x1, b1 = self.layer_1_2(x1_1)

        x2_1 = self.layer_2_1(x1)
        x2_2, b2_2 = self.layer_2_2(x2_1)
        x2, b2 = self.layer_2_3(x2_2)

        x3_1 = self.layer_3_1(x2)
        x3 = self.layer_3_2(x3_1, epoch, hw_range)

        x4_1 = self.layer_4_1(x3)
        x4 = self.layer_4_2(x4_1, epoch, hw_range)

        x5_1 = self.layer_5_1(x4)
        x5 = self.layer_5_2(x5_1, epoch, hw_range)

        return [x1, x2, x3, x4, x5, b1, b2_2, b2]


if __name__ == '__main__':
    # 创建模型实例
    model = Backbone().cuda()

    # 假设输入的形状为 (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 352, 352).cuda()  # 输入一个 batch 的数据

    # 使用 profile 函数计算 FLOPs
    flops, params = profile(model, inputs=(input_tensor, 200, [0, 18]))  # epoch=1, hw_range=[0,18]

    print('GFLOPs : {:.2f} G'.format(flops / 1e9))
    print('Params : {:.2f} M'.format(params / 1e6))