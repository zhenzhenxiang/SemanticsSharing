"""MobileNet3 for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import time

import scipy.misc as misc
import kornia as dgm
import numpy as np
from light.model.base import BaseModel
from light.map.warp import warp_kornia
from light.map.pwcnet_model import Net, pwcflow


#from light.model import convert_state_dict
__all__ = ['MobileNetV3Seg', 'get_mobilenet_v3_large_seg', 'get_mobilenet_v3_small_seg']
 # lite flow net init        
 # self.liteflow_model = liteflownet(args.liteflow_model_path).cuda().train()
 
class MobileNetV3Seg(BaseModel):
    def __init__(self, nclass, args, aux=False, backbone='mobilenetv3_large', pretrained_base=False, **kwargs):
        super(MobileNetV3Seg, self).__init__(nclass, aux, backbone, pretrained_base, **kwargs)
        self.num = 0
        mode = backbone.split('_')[-1]

        self.flow_Network = Net()
        self.flow = pwcflow

        self.args=args
        self.scale=args.scale
        self.nclass = nclass
        self.FFM_120_flag = args.FFM_120_flag
        self.FFM_60_flag = args.FFM_60_flag
        self.warp = warp_kornia()
        self.head = _Head(nclass, mode, **kwargs)
        
        if args.FFM_120 == 'FFM_bi':
            self.FFM_120 = FeatureFusionModule_bi(nclass)            
        elif args.FFM_120 == 'FFM_conv':
            self.FFM_120 = FeatureFusionModule_conv(nclass)
        elif args.FFM_120 == 'FFM_basic_add':
            self.FFM_120 = FeatureFusionModule_basic_add(nclass)
        elif args.FFM_120 == 'FFM_basic_cat':
            self.FFM_120 = FeatureFusionModule_basic_cat(nclass)
        elif args.FFM_120 == 'FFM_res':
            self.FFM_120 = FeatureFusionModule_res(nclass)
        elif args.FFM_120 == 'FFM_res131':
            self.FFM_120 = FeatureFusionModule_res_131(nclass)
        elif args.FFM_120 == 'FFM_res131_120':
            self.FFM_120 = FeatureFusionModule_res_131_120(nclass)
        elif args.FFM_120 == 'FFM_res_120':
            self.FFM_120 = FeatureFusionModule_res_120(nclass)
            
            
        
        
        if args.FFM_60 == 'FFM_bi':
            self.FFM_60 = FeatureFusionModule_bi(nclass)
        elif args.FFM_60 == 'FFM_conv':            
            self.FFM_60 = FeatureFusionModule_conv(nclass)
        elif args.FFM_60 == 'FFM_basic_add':
            self.FFM_60 = FeatureFusionModule_basic_add(nclass)
        elif args.FFM_60 == 'FFM_basic_cat':
            self.FFM_60 = FeatureFusionModule_basic_cat(nclass)
        elif args.FFM_60 == 'FFM_res':
            self.FFM_60 = FeatureFusionModule_res(nclass)
        elif args.FFM_60 == 'FFM_res131':
            self.FFM_60 = FeatureFusionModule_res_131(nclass)
        
        
        
        inter_channels = 40 if mode == 'large' else 24


    def forward(self, x):
        """ 
        input x : numpy.array
        """
        assert x[0].shape == x[1].shape
        img_size = x[0].size()[2:]
        scale_size = (int(img_size[0]*self.scale),int(img_size[1]*self.scale))
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
        F_60_8_scale = F.interpolate(c1_60, scale_size, mode='bilinear', align_corners=True)#channal 40
        
        
        #liteflow input
        I_120_1_scale = F.interpolate(x[0], scale_size, mode='bilinear', align_corners=True) 
        I_120_1_scale_warped = self.warp(I_120_1_scale, '120_60', self.scale)          
        I_60_1_scale = F.interpolate(x[1], scale_size, mode='bilinear', align_corners=True)    
        
        #liteflow output          
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

    
    
class FeatureFusionModule_bi(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()        
        self.atrconvblock = atrConvBlock(40, 40)
        self.convblock = ConvBlock(in_channels=46, out_channels=num_classes)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    def forward(self, input_1, input_2):
        x = self.atrconvblock(input_2)
        x = torch.cat((input_1, x), dim=1)        
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
        # self.in_channels = input_1.channels + input_2.channels    # add  
        self.in_channels = num_classes * 2                          # concat   
        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=self.in_channels)
        self.conv1 = nn.Conv2d(self.in_channels, num_classes, kernel_size=1)
    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)  # concat 
        #x = input_1 + input_2                    # add 
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        x = self.convblock(x)
        x = self.conv1(x)
        return x
    
class FeatureFusionModule_basic_cat(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()         
        self.convblock = ConvBlock(40, num_classes)
        self.conv1 = nn.Conv2d(num_classes*2, num_classes, kernel_size=1)
    def forward(self, input_1, input_2):          
        #concat
        x = torch.cat((input_1, self.convblock(input_2)), dim=1)
        x = self.conv1(x)
        return x
    

    
    
class FeatureFusionModule_basic_add(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()   
        self.convblock = ConvBlock(40, num_classes)
    def forward(self, input_1, input_2):
        #add
        x = input_1 + self.convblock(input_2)
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
        if False:
            pretrained_dict_mobilenetv3 = convert_state_dict(torch.load(pretrained_path))
            model_dict = model.state_dict()
            pretrained_dict_mobilenetv3 = {k: v for k, v in pretrained_dict_mobilenetv3.items() if
                                           (k in model_dict and v.size() == model_dict[k].size())}   

            model_dict.update(pretrained_dict_mobilenetv3)     
            model.load_state_dict(model_dict)
        if True:
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
