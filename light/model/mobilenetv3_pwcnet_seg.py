"""MobileNet3 for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from light.model.base import BaseModel
from light.map.warp import warp_kornia
from light.map.warp_xy import warp_kornia_xy

from light.map.pwcnet_model import Net, pwcflow
from light.map.pwcnet_model_mobilenetv3 import Net_MobileNetV3, pwcflow_mobilenetv3

from light.utils.load_model import convert_state_dict
from light.data import datasets

__all__ = ['MobileNetV3Seg', 'MobileNetV3Seg_loosely', 'MobileNetV3Seg_tightly', 'get_mobilenet_v3_large_seg', 'get_mobilenet_v3_small_seg']


class MobileNetV3Seg(BaseModel):
    def __init__(self, args, aux=False, backbone='mobilenetv3_large', pretrained_base=False, **kwargs):
        super(MobileNetV3Seg, self).__init__(args.nclass, aux, backbone, pretrained_base, **kwargs)
        mode = backbone.split('_')[-1]

        self.args = args
        self.nclass = args.nclass
        self.head = _Head(self.nclass, mode, **kwargs)
        inter_channels = 40 if mode == 'large' else 24

    def forward(self, x):
        """
        input x : numpy.array
        """
        assert x[0].shape == x[1].shape
        img_size = x[0].size()[2:]

        # 120 feature extractor
        _, _, _, _, c4_120 = self.base_forward(x[0])
        F_120_16 = self.head(c4_120)

        # 60 feature extractor
        _, _, _, _, c4_60 = self.base_forward(x[1])
        F_60_16 = self.head(c4_60)

        # output
        seg_result_60 = F.interpolate(F_60_16, img_size, mode='bilinear', align_corners=True)
        seg_result_120 = F.interpolate(F_120_16, img_size, mode='bilinear', align_corners=True)

        outputs = list()
        outputs.append([seg_result_120, seg_result_60])
        return tuple(outputs)


class MobileNetV3Seg_loosely(BaseModel):
    def __init__(self, args, aux=False, backbone='mobilenetv3_large', pretrained_base=False, **kwargs):
        super(MobileNetV3Seg_loosely, self).__init__(args.nclass, aux, backbone, pretrained_base, **kwargs)
        mode = backbone.split('_')[-1]

        self.flow_Network = Net()
        self.flow = pwcflow

        self.args=args
        self.scale=args.scale
        self.nclass = args.nclass
        self.warp = warp_kornia()
        self.head = _Head(self.nclass, mode, **kwargs)

        if args.FFM_120 == 'FFM_res131_120':
            self.FFM_120 = FeatureFusionModule_res_131_120(self.nclass)
        elif args.FFM_120 == 'FFM_res_120':
            self.FFM_120 = FeatureFusionModule_res_120(self.nclass)

        if args.FFM_60 == 'FFM_res':
            self.FFM_60 = FeatureFusionModule_res(self.nclass)
        elif args.FFM_60 == 'FFM_res131':
            self.FFM_60 = FeatureFusionModule_res_131(self.nclass)


        inter_channels = 40 if mode == 'large' else 24


    def forward(self, x):
        """
        input x : numpy.array
        """
        assert x[0].shape == x[1].shape
        img_size = x[0].size()[2:]
        scale_size = (int(img_size[0]*self.scale),int(img_size[1]*self.scale))

        # 120 feature extractor
        _, c1_120, _, _, c4_120 = self.base_forward(x[0])
        F_120_16 = self.head(c4_120)
        F_120_8_scale = F.interpolate(F_120_16, scale_size, mode='bilinear', align_corners=True)

        # 60 feature extractor
        _, c1_60, _, _, c4_60 = self.base_forward(x[1])
        F_60_8_scale = F.interpolate(c1_60, scale_size, mode='bilinear', align_corners=True)#channal 40

        #flow input
        I_120_1_scale = F.interpolate(x[0], scale_size, mode='bilinear', align_corners=True)
        I_120_1_scale_warped = self.warp(I_120_1_scale, '120_60', self.scale)
        I_60_1_scale = F.interpolate(x[1], scale_size, mode='bilinear', align_corners=True)

        #flow output
        grid_120_60, grid_60_120 = self.flow(I_120_1_scale_warped, I_60_1_scale, self.flow_Network)
        F120_60_scale = remap(self.warp(F_120_8_scale, '120_60', self.scale), grid_120_60)

        #fusion
        seg_result_60_scale = self.FFM_60(F120_60_scale,F_60_8_scale)
        seg_result_60_scale_120 = self.warp(remap(seg_result_60_scale, grid_60_120), '60_120', self.scale)
        seg_result_120_scale = self.FFM_120(F_120_8_scale, seg_result_60_scale_120)

        # output
        seg_result_60 = F.interpolate(seg_result_60_scale, img_size, mode='bilinear', align_corners=True)
        seg_result_120 = F.interpolate(seg_result_120_scale, img_size, mode='bilinear', align_corners=True)

        outputs = list()
        outputs.append([seg_result_120, seg_result_60])
        return tuple(outputs)


class MobileNetV3Seg_tightly(BaseModel):
    def __init__(self, args, aux=False, backbone='mobilenetv3_large', pretrained_base=False, **kwargs):
        super(MobileNetV3Seg_tightly, self).__init__(args.nclass, aux, backbone, pretrained_base, **kwargs)
        mode = backbone.split('_')[-1]

        self.flow_Network = Net_MobileNetV3()
        self.flow = pwcflow_mobilenetv3

        self.args = args
        self.scale = args.scale
        self.nclass = args.nclass

        self.warp_1_16 = warp_kornia_xy(args, '120_60', args.batch_size, 0.062914, 0.0625)
        self.warp_1_8 = warp_kornia_xy(args, '120_60', args.batch_size, 0.125827, 0.125)
        self.warp_1_4 = warp_kornia_xy(args, '120_60', args.batch_size, 0.251655, 0.25)
        self.warp_2_5 = warp_kornia_xy(args, '120_60', args.batch_size, 0.402649, 0.4)
        self.warp_2_5_60_120 = warp_kornia_xy(args, '60_120', args.batch_size, 0.402649, 0.4)

        self.head = _Head(self.nclass, mode, **kwargs)

        if args.FFM_120 == 'FFM_res131_120':
            self.FFM_120 = FeatureFusionModule_res_131_120(self.nclass)
        elif args.FFM_120 == 'FFM_res_120':
            self.FFM_120 = FeatureFusionModule_res_120(self.nclass)

        if args.FFM_60 == 'FFM_res':
            self.FFM_60 = FeatureFusionModule_res(self.nclass)
        elif args.FFM_60 == 'FFM_res131':
            self.FFM_60 = FeatureFusionModule_res_131(self.nclass)

        inter_channels = 40 if mode == 'large' else 24


    def forward(self, x):
        """
        input x : numpy.array
        """
        assert x[0].shape == x[1].shape
        img_size = x[0].size()[2:]
        scale_size = (int(img_size[0] * self.scale), int(img_size[1] * self.scale))

        # 120 feature extractor
        c0_120, c1_120, c2_120, _, c4_120 = self.base_forward(x[0])
        F_120_16 = self.head(c4_120)
        F_120_8 = F.interpolate(F_120_16, c1_120.size()[2:], mode='bilinear', align_corners=True)
        F_120_8_scale = F.interpolate(F_120_8, scale_size, mode='bilinear', align_corners=True)

        # 60 feature extractor
        c1_60 = self.base_forward60(x[1])
        # F_60_8_scale = F.interpolate(c1_60, scale_size, mode='bilinear', align_corners=True)  # channal 40


        # flow input
        I_120_1_warped = [self.warp_1_16(c2_120),
                          self.warp_1_8(c1_120),
                          self.warp_1_4(c0_120)]
        I_120_1_warped_scale = [F.interpolate(I_120_1_warped[0], (32, 48), mode='bilinear', align_corners=True),
                                F.interpolate(I_120_1_warped[1], (64, 96), mode='bilinear', align_corners=True),
                                F.interpolate(I_120_1_warped[2], (128, 192), mode='bilinear', align_corners=True)]
        I_60_1_scale = F.interpolate(x[1], scale_size, mode='bilinear', align_corners=True)

        # flow output
        grid_120_60, grid_60_120, F_60_8 = self.flow(I_120_1_warped_scale, I_60_1_scale, self.flow_Network)
        # F_60_8_scale = F.interpolate(F_60_8, scale_size, mode='bilinear', align_corners=True)

        # fusion
        F120_60_scale = remap(self.warp_2_5(F_120_8_scale), grid_120_60)
        F120_60_8_scale_8 = F.interpolate(F120_60_scale, c1_120.size()[2:], mode='bilinear', align_corners=True)  # channal 40
        seg_result_60_8 = self.FFM_60(c1_60, F120_60_8_scale_8)
        seg_result_60_8_scale = F.interpolate(seg_result_60_8, scale_size, mode='bilinear', align_corners=True)  # channal 40

        seg_result_60_scale_120 = self.warp_2_5_60_120(remap(seg_result_60_8_scale, grid_60_120))
        seg_result_60_scale_120_8 = F.interpolate(seg_result_60_scale_120, c1_120.size()[2:], mode='bilinear',align_corners=True)  # channal 40
        seg_result_120_8 = self.FFM_120(F_120_8, seg_result_60_scale_120_8)


        # output
        seg_result_60 = F.interpolate(seg_result_60_8, img_size, mode='bilinear', align_corners=True)
        seg_result_120 = F.interpolate(seg_result_120_8, img_size, mode='bilinear', align_corners=True)

        outputs = list()
        outputs.append([seg_result_120, seg_result_60])
        return tuple(outputs)


def remap(imgs, grid):
    """
    input: imgs tensor NCHW
    """
    results = F.grid_sample(imgs, grid)
    return results

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class atrConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, dilation=2):
        super().__init__()
        self.atrconv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2 * dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, input):
        x = self.atrconv(input)
        return self.relu(self.bn(x))

class Residual131(torch.nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.convblock_in = ConvBlock(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.convblock_mid = ConvBlock(mid_channels, mid_channels)
        self.conv1_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        res = x
        x = self.convblock_in(x)
        x = self.convblock_mid(x)
        x = self.conv1_out(x)
        x = self.bn(x)
        x = self.relu(x + res)
        return x



class FeatureFusionModule_res(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        self.convblock = ConvBlock(46, 46)
        self.conv3 = nn.Conv2d(46, 46,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(46)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(46, num_classes,kernel_size=1, stride=1, padding=0)
    def forward(self, input_1, input_2):
        feature = torch.cat((input_1, input_2), dim=1)
        x = self.convblock(feature)
        x = self.bn(self.conv3(x))
        x = self.relu(x + feature)
        x = self.conv1(x)
        return x

class FeatureFusionModule_res_120(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        self.convblock = ConvBlock(12, 64)
        self.conv3 = nn.Conv2d(64, 12,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(12)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(12, num_classes,kernel_size=1, stride=1, padding=0)
    def forward(self, input_1, input_2):
        feature = torch.cat((input_1, input_2), dim=1)
        x = self.convblock(feature)
        x = self.bn(self.conv3(x))
        x = self.relu(x + feature)
        x = self.conv1(x)
        return x

class FeatureFusionModule_res_131(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.atrconvblock = atrConvBlock(40, 40)
        self.con1 = nn.Conv2d(40, num_classes,kernel_size=1, stride=1, padding=0)
        self.residual131 = Residual131(6, 32, 6)
    def forward(self, input_1, input_2):
        x = self.atrconvblock(input_2)
        x = self.con1(x)
        x = self.residual131(input_1 + x)
        return x

class FeatureFusionModule_res_131_120(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.residual131 = Residual131(6, 32, 6)
        #self.con1 = nn.Conv2d(12, num_classes,kernel_size=1, stride=1, padding=0)

    def forward(self, input_1, input_2):
        #x = torch.cat((input_1, input_2), dim=1)

        x = input_1 + input_2
        x = self.residual131(x)
        #x = self.con1(x)
        return x



class _Head(nn.Module):
    def __init__(self, nclass, mode='small', norm_layer=nn.BatchNorm2d, **kwargs):
        super(_Head, self).__init__()
        in_channels = 960 if mode == 'large' else 576
        self.lr_aspp = _LRASPP(in_channels, norm_layer, **kwargs)
        self.project = nn.Conv2d(128, nclass, 1)

    def forward(self, x):
        x = self.lr_aspp(x)
        return self.project(x)

class _LRASPP(nn.Module):
    """Lite R-ASPP"""

    def __init__(self, in_channels, norm_layer, **kwargs):
        super(_LRASPP, self).__init__()
        out_channels = 128
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )
        self.b1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(49,49), stride=(16,20)),  # check it
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat2 = F.interpolate(feat2, size, mode='bilinear', align_corners=True)
        x = feat1 * feat2  # check it
        return x


def get_mobilenet_v3_large_pwcflow_seg(args,
                                        dataset='yuanqu',
                                        pretrained=False,
                                        root='~/.torch/models',
                                        pretrained_base=False,
                                        pretrained_path="",
                                        **kwargs):
    acronyms = {
        'citys': 'citys',
        'yuanqu': 'yuanqu'
    }
    if args.model_mode == 'mobilenetv3':
        model = MobileNetV3Seg(args=args, backbone='mobilenetv3_large', pretrained_base=pretrained_base, **kwargs)
    elif args.model_mode == 'mobilenetv3_loosely':
        model = MobileNetV3Seg_loosely(args=args, backbone='mobilenetv3_large', pretrained_base=pretrained_base, **kwargs)
    elif args.model_mode == 'mobilenetv3_tightly':
        model = MobileNetV3Seg_tightly(args=args, backbone='mobilenetv3_large', pretrained_base=pretrained_base, **kwargs)

    if pretrained:
        model.load_state_dict(convert_state_dict(torch.load(pretrained_path)))
    return model


def get_mobilenet_v3_small_pwcflow_seg(args,
                                        dataset='yuanqu',
                                        pretrained=False,
                                        root='~/.torch/models',
                                        pretrained_base=False,
                                        pretrained_path="",
                                        **kwargs):
    acronyms = {
        'citys': 'citys',
        'yuanqu': 'yuanqu'

    }

    if args.model_mode == 'mobilenetv3':
        model = MobileNetV3Seg(args=args, backbone='mobilenetv3_small', pretrained_base=pretrained_base, **kwargs)
    elif args.model_mode == 'mobilenetv3_loosely':
        model = MobileNetV3Seg_loosely(args=args, backbone='mobilenetv3_small', pretrained_base=pretrained_base, **kwargs)
    elif args.model_mode == 'mobilenetv3_tightly':
        model = MobileNetV3Seg_tightly(args=args, backbone='mobilenetv3_small', pretrained_base=pretrained_base, **kwargs)

    if pretrained:
        model.load_state_dict(convert_state_dict(torch.load(pretrained_path)))
    return model


if __name__ == '__main__':
    model = get_mobilenet_v3_small_seg()
