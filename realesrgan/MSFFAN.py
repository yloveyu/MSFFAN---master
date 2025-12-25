import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights, make_layer, pixel_unshuffle


class LayerNorm(nn.Module):
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

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)



class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y



class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


# 基础块（Block）
class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res = res + x
        return res


# 分组（Group）
class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()

        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res = res + x
        return res



class FFA(nn.Module):
    def __init__(self, gps = 3, blocks = 5, conv=default_conv):
        super(FFA, self).__init__()
        self.gps = 3
        self.dim = 48
        kernel_size = 3
        pre_process = [conv(48, self.dim, kernel_size)]


        assert self.gps == 3


        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)


        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


        self.palayer = PALayer(self.dim)

        post_process = [
            conv(self.dim, self.dim, kernel_size),

        ]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_process)

    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        out = self.palayer(out)
        x = self.post(out)
        return x + x1


class MY_ResidualDenseBlock(nn.Module):

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(MY_ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class MY_RRDB(nn.Module):

    def __init__(self, num_feat, num_grow_ch=32):
        super(MY_RRDB, self).__init__()
        self.conv1 = nn.Conv2d(num_feat , num_feat, kernel_size=1)
        self.rdb1 = MY_ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = MY_ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = MY_ResidualDenseBlock(num_feat, num_grow_ch)
        # self.norm = LayerNorm(num_feat, data_format='channels_first')
        # self.lpa = LPA(num_feat)
        # self.gsau = GSAU(num_feat)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = out + x

        out = self.conv1(out)
        return out * 0.2 + x


@ARCH_REGISTRY.register()
class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block with LPA"""

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=5, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale

        self.conv_first_1 = nn.Conv2d(num_in_ch * 4, num_feat, 3, 1, 1)
        self.conv_first_2 = nn.Conv2d(num_in_ch * 16, num_feat, 3, 1, 1)
        self.body = make_layer(MY_RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        #self.conv_ch = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.ffa = FFA(num_feat)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat_1 = pixel_unshuffle(x, scale=2)
            feat_2 = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat_1 = self.conv_first_1(feat_1)
        body_feat_1 = self.conv_body(self.body(feat_1))
        feat_1 = feat_1 + body_feat_1
        feat_2 = self.conv_first_2(feat_2)
        body_feat_2 = self.conv_body(self.body(feat_2))
        feat_2 = feat_2 + body_feat_2

        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat_2, scale_factor=2, mode='nearest')))
        feat = feat_1 + feat
        feat = self.ffa(feat)
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

