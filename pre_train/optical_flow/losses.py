import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def L1loss(x, y): return (x - y).abs().mean()
def L2loss(x, y): return torch.norm(x - y, p = 2, dim = 1).mean()

def training_loss(args, flow_pyramid, flow_gt_pyramid):
    return sum(w * L2loss(flow, gt) for w, flow, gt in zip(args.weights, flow_pyramid, flow_gt_pyramid))
    
def robust_training_loss(args, flow_pyramid, flow_gt_pyramid):
    return sum((w * L1loss(flow, gt) + args.epsilon) ** args.q for w, flow, gt in zip(args.weights, flow_pyramid, flow_gt_pyramid))
    


def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()


class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue


class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue


class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.args = args
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, outputs, target):
        lossvalue = self.loss(outputs[-1], target)
        epevalue = EPE(outputs[-1], target)
        return [lossvalue, epevalue]


class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, outputs, target):
        t = target
        o = outputs[-1]
        shape = list(o.size())
        shape_t = list(t.size())
        if shape != shape_t:
            scale_h = float(shape_t[2]) / float(shape[2])
            scale_w = float(shape_t[3]) / float(shape[3])
            o = F.upsample(o, size=(shape_t[2], shape_t[3]), mode='bilinear')
            o[:, 0, :, :] *= scale_h
            o[:, 1, :, :] *= scale_w
        lossvalue = self.loss(o, t)
        epevalue = EPE(o, t)
        return [lossvalue, epevalue]


class MultiScale(nn.Module):
    def __init__(self, args, startScale = 5, numScales = 6, l_weight= 0.32, norm= 'L2'):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)]).to(args.device)
        self.args = args
        self.l_type = norm
        self.div_flow = 0.05
        assert(len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1': self.loss = L1()
        else: self.loss = L2()

        self.multiScales = [nn.AvgPool2d(2**l, 2**l) for l in range(args.num_levels)][::-1][:args.output_level]
        self.loss_labels = ['MultiScale-'+self.l_type, 'EPE'],

    def forward(self, outputs, target):
        args = self.args
        # if flow is normalized, every output is multiplied by its size
        # correspondingly, groundtruth should be scaled at each level
        targets = [avg_pool(target) / 2 ** (args.num_levels - l - 1) for l, avg_pool in enumerate(self.multiScales)] + [target]
        loss, epe = 0, 0
        loss_levels, epe_levels = [], []
        for w, o, t in zip(args.weights, outputs, targets):
            # print(f'flow值域: ({o.min()}, {o.max()})')
            # print(f'gt值域: ({t.min()}, {t.max()})')
            # print(f'EPE:', EPE(o, t))
            shape = list(o.size())
            shape_t = list(t.size())
            if shape != shape_t:
                scale_h = float(shape_t[2]) / float(shape[2])
                scale_w = float(shape_t[3]) / float(shape[3])
                o = F.upsample(o, size=(shape_t[2], shape_t[3]), mode='bilinear')
                o[:, 0, :, :] *= scale_h
                o[:, 1, :, :] *= scale_w
            loss += w * self.loss(o, t)
            epe += EPE(o, t)
            loss_levels.append(self.loss(o, t))
            epe_levels.append(EPE(o, t))
        return [loss, epe, loss_levels, epe_levels]


class MultiScale_MobileNetV3(nn.Module):
    def __init__(self, args, norm='L2'):
        super(MultiScale_MobileNetV3,self).__init__()

        self.args = args
        if norm == 'L1': self.loss = L1()
        else: self.loss = L2()

        self.multiScales = [nn.AvgPool2d(2 ** (l+1), 2 ** (l+1)) for l in range(len(args.weights))][::-1]

    def forward(self, outputs, target):
        # check size of weights and outputs
        assert(len(outputs) == len(self.args.weights))

        # if flow is normalized, every output is multiplied by its size
        # correspondingly, groundtruth should be scaled at each level
        total_loss, total_epe = 0, 0
        loss_levels, epe_levels = [], []

        # resize to a multiple of 32
        h_raw = list(target.size())[2]
        w_raw = list(target.size())[3]
        h_dst = int(math.floor(math.ceil(h_raw / 32.0) * 32.0))
        w_dst = int(math.floor(math.ceil(w_raw / 32.0) * 32.0))

        scale_h = float(h_dst) / float(h_raw)
        scale_w = float(w_dst) / float(w_raw)

        target_dst = F.interpolate(target, (h_dst, w_dst), mode='bilinear', align_corners=True)
        target[:, 0, :, :] *= scale_h
        target[:, 1, :, :] *= scale_w

        targets = [avg_pool(target_dst) / 2 ** (len(self.multiScales) - l) for l, avg_pool in enumerate(self.multiScales)]

        # for i, t in enumerate(targets):
        #     print('t{}: min {}, max {}'.format(i, np.min(t.cpu().numpy()), np.max(t.cpu().numpy())))

        for i, (w, o) in enumerate(zip(self.args.weights, outputs)):
            if i != len(outputs) - 1:
                t = targets[i]
            else:
                t = target

            loss_ = w * self.loss(o, t)
            epe_ = EPE(o, t)

            loss_levels.append(loss_)
            epe_levels.append(epe_)

            total_epe += epe_
            total_loss += loss_

        return [total_loss, total_epe, loss_levels, epe_levels]

