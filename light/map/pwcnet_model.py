import torch
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import PIL.Image
import os,sys
import imageio
from spatial_correlation_sampler import SpatialCorrelationSampler

from light.map.pwcnet_args import pwcnet_args
from light.map.pwcnet_flow_utils import vis_flow, save_flow
from light.map.pwcnet_modules import (WarpingLayer, FeaturePyramidExtractor, FeaturePyramidExtractorDW, CostVolumeLayer, OpticalFlowEstimator, OpticalFlowEstimatorDW, ContextNetwork, get_grid)
# from correlation_package.modules.correlation import Correlation



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.args = pwcnet_args()
        args = self.args

        self.feature_pyramid_extractor = FeaturePyramidExtractor(args).to(args.device)
        
        self.warping_layer = WarpingLayer(args)
        self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=(args.search_range*2)+1, stride=1).to(args.device)
        self.flow_estimators = []
        for l, ch in enumerate(args.lv_chs[::-1]):
            
            layer = OpticalFlowEstimator(args, ch + (args.search_range*2+1)**2 + 2).to(args.device)
            self.add_module(f'FlowEstimator(Lv{l})', layer)
            self.flow_estimators.append(layer)

        if args.context:
            self.context_networks = []
            for l, ch in enumerate(args.lv_chs[::-1]):
                layer = ContextNetwork(args, ch + 2).to(args.device)
                self.add_module(f'ContextNetwork(Lv{l})', layer)
                self.context_networks.append(layer)

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
        x1_raw = x[:,:,0,:,:].contiguous()
        x2_raw = x[:,:,1,:,:].contiguous()
        
        # on the bottom level are original images
        t_pyramid = time()
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        flows = []
        estimation_time = 0

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            # upsample flow and scale the displacement
            if l == 0:
                shape = list(x1.size()); shape[1] = 2
                flow = torch.zeros(shape).to(args.device)
                pyramid_time = time() - t_pyramid
                # print('Pyramid time: {:.3f}ms'.format(pyramid_time * 1e3))
                t_flowestimate = time()
            else:
                # flow = F.upsample(flow, scale_factor = 2, mode = 'bilinear') * 2
                shape = list(x1.size())
                shape_f = list(flow.size())
                if shape != shape_f:
                    scale_h = float(shape[2]) / float(shape_f[2])
                    scale_w = float(shape[3]) / float(shape_f[3])
                    flow = F.upsample(flow, size=(shape[2], shape[3]), mode='bilinear')
                    flow[:, 0, :, :] *= scale_h
                    flow[:, 1, :, :] *= scale_w

            # TODO Correct the sign of 'flow', should be inversed
            x2_warp = self.warping_layer(x2, flow)

            # correlation
            corr = self.corr(x1, x2_warp)
            b, ph, pw, h, w = corr.size()
            corr = corr.view(b, ph * pw, h, w)
            corr /= x1.size(1)

            if args.corr_activation:
                F.leaky_relu_(corr)

            # concat and estimate flow
            # ATTENTION: `+ flow` makes flow estimator learn to estimate residual flow
            if args.residual:
                flow_coarse = self.flow_estimators[l](torch.cat([x1, corr, flow], dim = 1)).to(flow.device) + flow
            else:
                flow_coarse = self.flow_estimators[l](torch.cat([x1, corr, flow], dim = 1)).to(flow.device)

            if args.context:
                flow_fine = self.context_networks[l](torch.cat([x1, flow], dim = 1)).to(flow.device)
                flow = flow_coarse + flow_fine
            else:
                flow = flow_coarse

            if l == args.output_level:
                flow = F.upsample(flow, scale_factor = 2 ** (args.num_levels - args.output_level - 1), mode = 'bilinear') * 2 ** (args.num_levels - args.output_level - 1)
                flows.append(flow)
                estimation_time = time() - t_flowestimate
                break
            else:
                flows.append(flow)
        return flows
    

def pwcflow(tensorFirst, tensorSecond, pwcflow_Network):    

    tensorFirst[:,0,:,:] = (tensorFirst[:,0,:,:] * 0.229 + 0.485)*255
    tensorFirst[:,1,:,:] = (tensorFirst[:,1,:,:] * 0.224 + 0.456)*255
    tensorFirst[:,2,:,:] = (tensorFirst[:,2,:,:] * 0.225 + 0.406)*255
    
    tensorSecond[:,0,:,:] = (tensorSecond[:,0,:,:] * 0.229 + 0.485)*255
    tensorSecond[:,1,:,:] = (tensorSecond[:,1,:,:] * 0.224 + 0.456)*255
    tensorSecond[:,2,:,:] = (tensorSecond[:,2,:,:] * 0.225 + 0.406)*255
    
    
#     tensorFirst = torch.tensor(unnormlize(first_image_tensor.cpu().data.numpy()[0].transpose(1,2,0))).cuda() 
#     tensorSecond = torch.tensor(unnormlize(second_image_tensor.cpu().data.numpy()[0].transpose(1,2,0))).cuda()
    
    x = torch.stack([tensorFirst, tensorSecond], 2)   
    flows = pwcflow_Network(x)    
    
    flow = flows[-1]
    shape = list(flow.size())
    shape_t = list(x[:, :, 0, :, :].size())
    if shape != shape_t:
        scale_h = float(shape_t[2]) / float(shape[2])
        scale_w = float(shape_t[3]) / float(shape[3])
        flow = F.upsample(flow, size=(shape_t[2], shape_t[3]), mode='bilinear')
        flow[:, 0, :, :] *= scale_h
        flow[:, 1, :, :] *= scale_w
            
 
    flow_for_grip = torch.zeros_like(flow).cuda()
    flow_for_grip[:,0,:,:] = flow[:,0,:,:] / ((flow.size(3) - 1.0) / 2.0)
    flow_for_grip[:,1,:,:] = flow[:,1,:,:] / ((flow.size(2) - 1.0) / 2.0)
    
    
    grid_120_60 = (get_grid(tensorFirst) - flow_for_grip).permute(0, 2, 3, 1)
    grid_60_120 = (get_grid(tensorFirst) + flow_for_grip).permute(0, 2, 3, 1)
    
    
    return grid_120_60, grid_60_120
    
    

    
    

if __name__ == '__main__':
        # Get environment
    # Build Model
    # ============================================================
    from pwcnet_args import pwcnet_args
    from pwcnet_flow_utils import vis_flow, save_flow
    from pwcnet_modules import (WarpingLayer, FeaturePyramidExtractor, FeaturePyramidExtractorDW, CostVolumeLayer, OpticalFlowEstimator, OpticalFlowEstimatorDW, ContextNetwork)
    args = pwcnet_args()
    model = Net()
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
            flows= model(x)

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
