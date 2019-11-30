import torch.nn as nn

__all__ = ['FeaturePyramidMobileNetV3', 'OpticalFlowEstimator', '_ConvBNReLU', '_DWConvBNReLU',
           '_Hswish', '_ConvBNHswish', 'SEModule', 'Bottleneck']


class FeaturePyramidMobileNetV3(nn.Module):

    def __init__(self, args):
        super(FeaturePyramidMobileNetV3, self).__init__()
        self.args = args

        self.convs = []

        width_mult = 1.0
        self.in_channels = int(16 * width_mult) if width_mult > 1.0 else 16
        self.con1 = _ConvBNHswish(3, self.in_channels, 3, 2, 1, norm_layer=nn.BatchNorm2d)

        layer1_setting = [
            # k, exp_size, c, se, nl, s
            [3, 16, 16, False, 'RE', 1],
            [3, 64, 24, False, 'RE', 2],
            [3, 72, 24, False, 'RE', 1], ]
        layer2_setting = [
            [5, 72, 40, True, 'RE', 2],
            [5, 120, 40, True, 'RE', 1],
            [5, 120, 40, True, 'RE', 1], ]
        layer3_setting = [
            [3, 240, 80, False, 'HS', 2],
            [3, 200, 80, False, 'HS', 1],
            [3, 184, 80, False, 'HS', 1],
            [3, 184, 80, False, 'HS', 1],
            [3, 480, 112, True, 'HS', 1],
            [3, 672, 112, True, 'HS', 1],
            [5, 672, 112, True, 'HS', 1], ]

        self.layer1 = self._make_layer(Bottleneck, layer1_setting, width_mult, norm_layer=nn.BatchNorm2d)
        self.layer2 = self._make_layer(Bottleneck, layer2_setting, width_mult, norm_layer=nn.BatchNorm2d)
        self.layer3 = self._make_layer(Bottleneck, layer3_setting, width_mult, norm_layer=nn.BatchNorm2d)

        self.convs.append(self.layer1)
        self.convs.append(self.layer2)
        self.convs.append(self.layer3)

        self.out_chs_list = [24, 40, 112]

    def get_output_channels(self):
        return self.out_chs_list

    def _make_layer(self, block, block_setting, width_mult, dilation=1, norm_layer=nn.BatchNorm2d):
        layers = list()
        for k, exp_size, c, se, nl, s in block_setting:
            out_channels = int(c * width_mult)
            stride = s if (dilation == 1) else 1
            exp_channels = int(exp_size * width_mult)
            layers.append(
                block(self.in_channels, out_channels, exp_channels, k, stride, dilation, se, nl, norm_layer))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        feature_pyramid = []

        x = self.con1(x)

        for i, conv in enumerate(self.convs):
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


class OpticalFlowEstimatorBad(nn.Module):

    def __init__(self, args, ch_in):
        super(OpticalFlowEstimatorBad, self).__init__()
        self.args = args

        self.in_channels = ch_in

        layer_setting = [
            # k, exp_size, c, se, nl, s
            [3, 128, 128, False, 'RL', 1],
            [3, 96, 96, False, 'RL', 1],
            [3, 64, 64, False, 'RL', 1],
            [3, 32, 32, False, 'RL', 1]]

        self.layer = self._make_layer(Bottleneck, layer_setting, 1.0, norm_layer=nn.BatchNorm2d)

        self.layer_out = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1,
                      groups=1, bias=False),
            nn.BatchNorm2d(2))

        self.convs = nn.Sequential(
            self.layer,
            self.layer_out
        )

    def forward(self, x):
        return self.convs(x)


class OpticalFlowEstimator(nn.Module):

    def __init__(self, args, ch_in):
        super(OpticalFlowEstimator, self).__init__()
        self.args = args

        self.in_channels = ch_in

        self.convs = nn.Sequential(
            _DWConvBNReLU(ch_in, ch_in, 128, 1),
            _DWConvBNReLU(128, 128, 96, 1),
            _DWConvBNReLU(96, 96, 64, 1),
            _DWConvBNReLU(64, 64, 32, 1),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1,
                      groups=1, bias=False),
            nn.BatchNorm2d(2)
        )

    def forward(self, x):
        return self.convs(x)


class _ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# -----------------------------------------------------------------
#                      For MobileNet
# -----------------------------------------------------------------
class _DWConvBNReLU(nn.Module):
    """Depthwise Separable Convolution in MobileNet.
    depthwise convolution + pointwise convolution
    """

    def __init__(self, in_channels, dw_channels, out_channels, stride, dilation=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_DWConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            _ConvBNReLU(in_channels, dw_channels, 3, stride, dilation, dilation, in_channels, norm_layer=norm_layer),
            _ConvBNReLU(dw_channels, out_channels, 1, norm_layer=norm_layer))

    def forward(self, x):
        return self.conv(x)


# -----------------------------------------------------------------
#                      For MobileNetV3
# -----------------------------------------------------------------
class _Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(_Hswish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return x * self.relu6(x + 3.) / 6.


class _Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(_Hsigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return self.relu6(x + 3.) / 6.


class _ConvBNHswish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNHswish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.act = _Hswish(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            _Hsigmoid(True)
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        out = self.avg_pool(x).view(n, c)
        out = self.fc(out).view(n, c, 1, 1)
        return x * out.expand_as(x)


class Identity(nn.Module):
    def __init__(self, in_channels):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, exp_size, kernel_size, stride, dilation=1, se=False, nl='RE',
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super(Bottleneck, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels
        if nl == 'HS':
            act = _Hswish
        else:
            act = nn.ReLU
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, exp_size, 1, bias=False),
            norm_layer(exp_size),
            act(True),
            # dw
            nn.Conv2d(exp_size, exp_size, kernel_size, stride, (kernel_size - 1) // 2 * dilation,
                      dilation, groups=exp_size, bias=False),
            norm_layer(exp_size),
            SELayer(exp_size),
            act(True),
            # pw-linear
            nn.Conv2d(exp_size, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class _DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(_DWConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups=in_channels, bias=bias)

    def forward(self, x):
        return self.conv(x)
