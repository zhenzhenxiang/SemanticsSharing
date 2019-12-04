import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import imageio
import numpy as np
from light.map.pwcnet_modules import (WarpingLayer, FeaturePyramidExtractor, FeaturePyramidExtractorDW, CostVolumeLayer, OpticalFlowEstimator, OpticalFlowEstimatorDW, ContextNetwork, get_grid)
from light.map.pwcnet_modules_mobilenetv3 import FeaturePyramidMobileNetV3
from light.map.pwcnet_args_mobilenetv3 import pwcnet_args_mobilenetv3
from spatial_correlation_sampler import SpatialCorrelationSampler



class Net_MobileNetV3(nn.Module):
    def __init__(self):
        super(Net_MobileNetV3, self).__init__()
        self.args = pwcnet_args_mobilenetv3()
        args = self.args

#         self.feature_pyramid_extractor1 = FeaturePyramidMobileNetV3(args).to(args.device)
        self.feature_pyramid_extractor2 = FeaturePyramidExtractor(args).to(args.device)

        self.warping_layer = WarpingLayer(args)

        self.corr = SpatialCorrelationSampler(
            kernel_size=1, patch_size=(args.search_range * 2) + 1, stride=1, dilation_patch=1).to(args.device)

        self.flow_estimators = []
        for l, ch in enumerate(args.lv_chs[:1:-1]):
            layer = OpticalFlowEstimator(args, ch + (args.search_range * 2 + 1) ** 2 + 2).to(args.device)
            self.add_module(f'FlowEstimator(Lv{l})', layer)
            self.flow_estimators.append(layer)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None: nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None: nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        args = self.args
#         x1_raw = x[:, :, 0, :, :].contiguous()        
        x2_raw = x[1].contiguous()
#         x2_raw = x[:, :, 1, :, :].contiguous()        

        # resize to a multiple of 32
        h_raw = list(x2_raw.size())[2]
        w_raw = list(x2_raw.size())[3]
        h_dst = int(math.floor(math.ceil(h_raw / 32.0) * 32.0))
        w_dst = int(math.floor(math.ceil(w_raw / 32.0) * 32.0))
#         x1_dst = F.interpolate(x1_raw, (h_dst, w_dst), mode='bilinear', align_corners=True)
        x2_dst = F.interpolate(x2_raw, (h_dst, w_dst), mode='bilinear', align_corners=True)
        # generate feature pyramid
      
#         x1_pyramid = self.feature_pyramid_extractor1(x1_dst)
        x1_pyramid = x[0]
        x2_pyramid = self.feature_pyramid_extractor2(x2_dst)

        # outputs
        flows = []
        for l, x1 in enumerate(x1_pyramid):
            # upsample flow and scale the displacement
            if l == 0:
                shape = list(x1.size())
                shape[1] = 2
                flow = torch.zeros(shape).to(args.device)

            else:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2

            # TODO Correct the sign of 'flow', should be inverted
            x2 = x2_pyramid[l]            
            x2_warp = self.warping_layer(x2, -flow)

            # correlation
            corr = self.corr(x1, x2_warp)
            b, ph, pw, h, w = corr.size()
            corr = corr.view(b, ph * pw, h, w)
            corr /= x1.size(1)

            # concat and estimate flow
            # ATTENTION: `+ flow` makes flow estimator learn to estimate residual flow
            flow = self.flow_estimators[l](torch.cat([x1, corr, flow], dim=1)) + flow

            flows.append(flow)

        # scale to match the size of input
        shape_f = list(flows[-1].size())
        scale_h = float(h_raw) / float(shape_f[2])
        scale_w = float(w_raw) / float(shape_f[3])

        flow = F.interpolate(flow, (h_raw, w_raw), mode='bilinear', align_corners=True)
        flow[:, 0, :, :] *= scale_h
        flow[:, 1, :, :] *= scale_w

        flows.append(flow)
        return flows, x2_pyramid[1]   

    #####################################################################################################################################################
    
    def forward_ori(self, x):
        args = self.args      
        x2_raw = x[1].contiguous()     

        # resize to a multiple of 32
        h_raw = list(x2_raw.size())[2]
        w_raw = list(x2_raw.size())[3]
        h_dst = int(math.floor(math.ceil(h_raw / 32.0) * 32.0))
        w_dst = int(math.floor(math.ceil(w_raw / 32.0) * 32.0))
#         x1_dst = F.interpolate(x1_raw, (h_dst, w_dst), mode='bilinear', align_corners=True)
        x2_dst = F.interpolate(x2_raw, (h_dst, w_dst), mode='bilinear', align_corners=True)
        # generate feature pyramid
      
#         x1_pyramid = self.feature_pyramid_extractor1(x1_dst)
        x1_pyramid = x[0]
        x2_pyramid = self.feature_pyramid_extractor2(x2_dst)        

        # outputs
        flows = []
        for l, x1 in enumerate(x1_pyramid):
            # upsample flow and scale the displacement
            if l == 0:
                shape = list(x1.size())
                shape[1] = 2
                flow = torch.zeros(shape).to(args.device)
            else:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2

            # TODO Correct the sign of 'flow', should be inverted
            x2 = x2_pyramid[l]            
            x2_warp = self.warping_layer(x2, -flow)

            # correlation
            corr = self.corr(x1, x2_warp)
            b, ph, pw, h, w = corr.size()
            corr = corr.view(b, ph * pw, h, w)
            corr /= x1.size(1)

            # concat and estimate flow
            # ATTENTION: `+ flow` makes flow estimator learn to estimate residual flow
            flow = self.flow_estimators[l](torch.cat([x1, corr, flow], dim=1)) + flow

            flows.append(flow)

        # scale to match the size of input
        shape_f = list(flows[-1].size())
        scale_h = float(h_raw) / float(shape_f[2])
        scale_w = float(w_raw) / float(shape_f[3])

        flow = F.interpolate(flow, (h_raw, w_raw), mode='bilinear', align_corners=True)
        flow[:, 0, :, :] *= scale_h
        flow[:, 1, :, :] *= scale_w

        flows.append(flow)
        return flows, x2_pyramid[1]

    
def pwcflow_mobilenetv3(tensorFirst, tensorSecond, pwcflow_Network):
    
#     for i in range(len(tensorFirst)):
#         for j in range(tensorFirst[i].shape[1]):
#             tensorFirst[i][:,j,:,:] = (tensorFirst[i][:,j,:,:] * 0.229 + 0.485)*255           
#         print(i,j)
    
#     tensorSecond[:,0,:,:] = (tensorSecond[:,0,:,:] * 0.229 + 0.485)*255
#     tensorSecond[:,1,:,:] = (tensorSecond[:,1,:,:] * 0.224 + 0.456)*255
#     tensorSecond[:,2,:,:] = (tensorSecond[:,2,:,:] * 0.225 + 0.406)*255    
    
    x = [tensorFirst, tensorSecond]         
    flows, F_60_8= pwcflow_Network(x)  
    flow = flows[-1] 
    
#     import imageio
#     from light.map.pwcnet_flow_utils import vis_flow, save_flow 
#     import numpy as np
#     o = flow[0, :, :, :].cpu()
#     o = np.array(o.data).transpose(1, 2, 0)
#     flow_vis = vis_flow(o)
#     imageio.imwrite('./flow2.png', flow_vis)
    
    shape = list(flow.size())
    shape_t = list(tensorSecond.size())
    if shape != shape_t:
        scale_h = float(shape_t[2]) / float(shape[2])
        scale_w = float(shape_t[3]) / float(shape[3])
        flow = F.upsample(flow, size=(shape_t[2], shape_t[3]), mode='bilinear')
        flow[:, 0, :, :] *= scale_h
        flow[:, 1, :, :] *= scale_w            
 
    flow_for_grip = torch.zeros_like(flow).cuda()
    flow_for_grip[:,0,:,:] = flow[:,0,:,:] / ((flow.size(3) - 1.0) / 2.0)
    flow_for_grip[:,1,:,:] = flow[:,1,:,:] / ((flow.size(2) - 1.0) / 2.0)    
    
    grid_120_60 = (get_grid(tensorSecond) - flow_for_grip).permute(0, 2, 3, 1)
    grid_60_120 = (get_grid(tensorSecond) + flow_for_grip).permute(0, 2, 3, 1)    
    
    return grid_120_60, grid_60_120, F_60_8

def pwcflow_mobilenetv3_ori(tensorFirst, tensorSecond, pwcflow_Network):
    
    
#     tensorFirst[:,0,:,:] = (tensorFirst[:,0,:,:] * 0.229 + 0.485)*255
#     tensorFirst[:,1,:,:] = (tensorFirst[:,1,:,:] * 0.224 + 0.456)*255
#     tensorFirst[:,2,:,:] = (tensorFirst[:,2,:,:] * 0.225 + 0.406)*255
    
#     tensorSecond[:,0,:,:] = (tensorSecond[:,0,:,:] * 0.229 + 0.485)*255
#     tensorSecond[:,1,:,:] = (tensorSecond[:,1,:,:] * 0.224 + 0.456)*255
#     tensorSecond[:,2,:,:] = (tensorSecond[:,2,:,:] * 0.225 + 0.406)*255
    
    x = [tensorFirst, tensorSecond] 
    flows, F_60_8= pwcflow_Network(x)  
    flow = flows[-1]
    
#     import imageio
#     from light.map.pwcnet_flow_utils import vis_flow, save_flow 
#     import numpy as np
#     o = flow[0, :, :, :].cpu()
#     o = np.array(o.data).transpose(1, 2, 0)
#     flow_vis = vis_flow(o)
#     imageio.imwrite('./floe.png', flow_vis)
    
    shape = list(flow.size())
    shape_t = list(tensorSecond.size())
    if shape != shape_t:
        scale_h = float(shape_t[2]) / float(shape[2])
        scale_w = float(shape_t[3]) / float(shape[3])
        flow = F.upsample(flow, size=(shape_t[2], shape_t[3]), mode='bilinear')
        flow[:, 0, :, :] *= scale_h
        flow[:, 1, :, :] *= scale_w
            
 
    flow_for_grip = torch.zeros_like(flow).cuda()
    flow_for_grip[:,0,:,:] = flow[:,0,:,:] / ((flow.size(3) - 1.0) / 2.0)
    flow_for_grip[:,1,:,:] = flow[:,1,:,:] / ((flow.size(2) - 1.0) / 2.0)
    
    
    grid_120_60 = (get_grid(tensorSecond) - flow_for_grip).permute(0, 2, 3, 1)
    grid_60_120 = (get_grid(tensorSecond) + flow_for_grip).permute(0, 2, 3, 1)
    
    
    return grid_120_60, grid_60_120, F_60_8


if __name__ == '__main__':
        # Get environment
    # Build Model
    # ============================================================
    from pwcnet_args_mobilenetv3 import pwcnet_args_mobilenetv3
    from pwcnet_flow_utils import vis_flow, save_flow
    from pwcnet_modules import (WarpingLayer, FeaturePyramidExtractor, FeaturePyramidExtractorDW, CostVolumeLayer, OpticalFlowEstimator, OpticalFlowEstimatorDW, ContextNetwork)
    args = pwcnet_args_mobilenetv3()
    model = Net_MobileNetV3()
    model.load_state_dict(torch.load(args.load))
    model.to(args.device).eval()
    warp = WarpingLayer(args)

    # Load Data
    # ============================================================
    input_list = list()
    with open(os.path.join(args.input_dir,args.input), 'r') as input_file:
        lines = input_file.readlines()
        for l in lines:
            if l.strip():
                img1 = os.path.join(args.input_dir, l.strip().split(',')[0])
                img2 = os.path.join(args.input_dir, l.strip().split(',')[1])
                input_list.append([img1, img2])
    print('Total samples:', len(input_list))

    # split input list with batch size
    input_splits = [input_list[i:i + args.batch_size] for i in range(0, len(input_list), args.batch_size)]
    print('Total iter:', len(input_splits))
    for i, split in enumerate(input_splits):
        print('Processing {}/{}'.format(i + 1, len(input_splits)))
        # generate mini-batch inputs
        x_list = list()
        for img_pair in split:

            x1_raw, x2_raw = map(imageio.imread, img_pair)
            x1_raw = np.array(x1_raw)
            x2_raw = np.array(x2_raw)
            print(x1_raw.shape, x2_raw.shape )
            H, W = x1_raw.shape[:2]
            x1_raw = x1_raw[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
            x2_raw = x2_raw[np.newaxis, :, :, :].transpose(0, 3, 1, 2)

            x = np.stack([x1_raw, x2_raw], axis=2)
            print(x.shape)
            x_list.append(x)

        x = np.concatenate(x_list, axis=0)
        x = torch.Tensor(x).to(args.device)
        print(x.shape)

        # Forward Pass
        # ============================================================
        with torch.no_grad():
            flows,_= model(x)

        # warp input
        print(len(flows))
        o = flows[-1]     
        print(o.shape)
        shape = list(o.size())
        print(shape)
        shape_t = list(x[:, :, 0, :, :].size())
        print(shape_t)      
        if shape != shape_t:
            scale_h = float(shape_t[2]) / float(shape[2])
            scale_w = float(shape_t[3]) / float(shape[3])
            o = F.upsample(o, size=(shape_t[2], shape_t[3]), mode='bilinear')
            o[:, 0, :, :] *= scale_h
            o[:, 1, :, :] *= scale_w
        x_warped = warp(x[:, :, 0, :, :], -o).cpu().numpy().astype(np.uint8)

        for i, img_pair in enumerate(split):

            flow = o[i, :, :, :].cpu()
            flow = np.array(flow.data).transpose(1, 2, 0)

            # save to file
            flow_dir = os.path.join(args.output, 'flow')
            flow_vis_dir = os.path.join(args.output, 'flow_vis')
            img_warped_dir = os.path.join(args.output, 'flow_warped')

            if os.path.exists(flow_dir) is False:
                os.mkdir(flow_dir)
            if os.path.exists(flow_vis_dir) is False:
                os.mkdir(flow_vis_dir)
            if os.path.exists(img_warped_dir) is False:
                os.mkdir(img_warped_dir)

            img_base_name = os.path.basename(img_pair[0])
            flow_file = img_base_name.replace(os.path.splitext(img_base_name)[-1], '.flo')
            flow_vis_file = flow_file.replace('.flo', '.png')
            img_warped_file = img_base_name

            save_flow(os.path.join(flow_dir,flow_file), flow)
            flow_vis = vis_flow(flow)
            imageio.imwrite(os.path.join(flow_vis_dir,flow_vis_file), flow_vis)
            imageio.imwrite(os.path.join(img_warped_dir,img_warped_file), x_warped[i, :, :, :].transpose(1, 2, 0))           
    print('Finished!')
