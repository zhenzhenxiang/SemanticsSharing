"""MobileNet3 for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import kornia as dgm
import numpy as np
from light.model.base import BaseModel
from light.map.warp import warp_kornia
from light.map.liteflow import Network, liteflow
#from light.model import convert_state_dict
__all__ = ['MobileNetV3Seg', 'get_mobilenet_v3_large_seg', 'get_mobilenet_v3_small_seg']
 # lite flow net init        
 # self.liteflow_model = liteflownet(args.liteflow_model_path).cuda().train()

class MobileNetV3Seg(BaseModel):
    def __init__(self, nclass, args, aux=False, backbone='mobilenetv3_large', pretrained_base=False, **kwargs):
        super(MobileNetV3Seg, self).__init__(nclass, aux, backbone, pretrained_base, **kwargs)
        
        mode = backbone.split('_')[-1]
        self.liteflow_Network = Network()
        self.scale=args.scale
        self.warp = warp_kornia()
        self.head = _Head(nclass, mode, **kwargs)
        self.FFM_conv_120 = FeatureFusionModule_conv(nclass)
        self.FFM_conv_60 = FeatureFusionModule_conv(nclass)
        self.FFM_bi = FeatureFusionModule_bi(nclass)
        
        inter_channels = 40 if mode == 'large' else 24
        self.auxlayer = nn.Conv2d(inter_channels, nclass, 1)

    def forward(self, x):
        """ 
        input x : numpy.array
        """
        assert x[0].shape == x[1].shape
        size = x[0].size()[2:]
        scale_size = (int(size[0]*self.scale),int(size[1]*self.scale))
        #size = [1208,1920]              
        
        # 120 feature extractor
        c1_120, c2_120, c3_120, c4_120 = self.base_forward(x[0])
        F_120_16 = self.head(c4_120)        
        F_120_8 = F.interpolate(F_120_16, c1_120.size()[2:], mode='bilinear', align_corners=True)
        if self.aux:
            auxout = self.auxlayer(c1_120)                               
            F_120_8 = F_120_8 + auxout
        
        F_120_8_scale = F.interpolate(F_120_8, scale_size, mode='bilinear', align_corners=True)
        
            
            
        # 60 feature extractor
        c1_60, c2_60, c3_60, c4_60 = self.base_forward(x[1])
        F_60_8 =  self.auxlayer(c1_60)
        F_60_8_scale = F.interpolate(F_60_8, scale_size, mode='bilinear', align_corners=True)
        
        
        #liteflow input
        I_120_1_warped = self.warp(x[0], '120_60')        
        I_120_1_warped_scale = F.interpolate(I_120_1_warped, scale_size, mode='bilinear', align_corners=True) # 0-255
        I_60_1_scale = F.interpolate(x[1], scale_size, mode='bilinear', align_corners=True)     #  0-255
        
        
        #liteflow output
        grid_120_60, grid_60_120 = liteflow(I_120_1_warped_scale, I_60_1_scale, self.liteflow_Network)
        F60_coarse_scale = remap(self.warp(F_120_8_scale, '120_60'), grid_120_60)
        F120_fine_scale = remap(self.warp(F_60_8_scale, '60_120'), grid_60_120)
        
        F120_fine_8 = F.interpolate(F120_fine_scale, c1_120.size()[2:], mode='bilinear', align_corners=True)
        
        #fusion120 
        # FFM_bi is ok
        seg_result_120_scale = self.FFM_conv_120(F120_fine_8, F_120_8)
        seg_result_60_scale = self.FFM_conv_60(F_60_8_scale, F60_coarse_scale)
        
        seg_result_120 = F.interpolate(seg_result_120_scale, size, mode='bilinear', align_corners=True)
        seg_result_60 = F.interpolate(seg_result_60_scale, size, mode='bilinear', align_corners=True)   
            
        #x = F.interpolate(x, size, mode='bilinear', align_corners=True)
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
    
class FeatureFusionModule_bi(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels      
        self.in_channels = num_classes * 2
        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1, padding=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

class FeatureFusionModule_conv(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels      
        self.in_channels = num_classes * 2
        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1,padding=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        x = self.convblock(x)
        x = self.conv1(x)
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


def get_mobilenet_v3_large_liteflow_seg(dataset='citys',                                       
                                        pretrained=False, 
                                        root='~/.torch/models',                               
                                        pretrained_base=False,
                                        pretrained_path="",
                                        **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
        'yuanqu': 'yuanqu'
    }
    from light.data import datasets
    model = MobileNetV3Seg(datasets[dataset].NUM_CLASS, 
                           backbone='mobilenetv3_large',
                           pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        def convert_state_dict(state_dict):
            """Converts a state dict saved from a dataParallel module to normal
                module state_dict inplace
                :param state_dict is the loaded DataParallel model_state
            """
            if not next(iter(state_dict)).startswith("module."):
                return state_dict  # abort if dict is not a DataParallel model_state
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            return new_state_dict
        #from ..model import get_model_file, convert_state_dict
        #model.load_state_dict(convert_state_dict(torch.load(get_model_file('mobilenetv3_large_%s_best_model' % (acronyms[dataset]), root=root))))
        model.load_state_dict(convert_state_dict(torch.load(pretrained_path)))
        #model.load_state_dict(torch.load(pretrained_path))

    return model


def get_mobilenet_v3_small_liteflow_seg(dataset='citys', pretrained=False, root='~/.torch/models',
                               pretrained_base=False, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from light.data import datasets
    model = MobileNetV3Seg(datasets[dataset].NUM_CLASS, backbone='mobilenetv3_small',
                           pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from ..model import get_model_file
        model.load_state_dict(
            torch.load(get_model_file('mobilenetv3_small_%s_best_model' % (acronyms[dataset]), root=root)))
    return model


if __name__ == '__main__':
    model = get_mobilenet_v3_small_seg()
