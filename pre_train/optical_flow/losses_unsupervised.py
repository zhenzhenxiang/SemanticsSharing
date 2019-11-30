# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import L1loss, EPE
from modules import WarpingLayer
from warp import warp_kornia


def smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def get_smooth_loss(flow, img):
    """Computes the smoothness loss for a flow image
        The color image is used for edge-aware smoothness
    """
    smoothness = 0
    for i in range(2):
        smoothness += smooth_loss(flow[:, i, :, :].unsqueeze(1), img)

    return smoothness / 2


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class SelfLoss(nn.Module):
    """Layer to compute the unsupervised loss between a pair of images
    """

    def __init__(self, args):
        super(SelfLoss, self).__init__()
        self.args = args
        self.SSIM = SSIM()
        self.warp = WarpingLayer(args)

    def forward(self, imgs, flows, flow_gt):
        img1 = imgs[:, :, 0, :, :].contiguous()
        img2 = imgs[:, :, 1, :, :].contiguous()
        flow = flows[-1]

        # scale flow to match the size of the input image
        shape = list(flow.size())
        shape_t = list(img1.size())
        if shape != shape_t:
            scale_h = float(shape_t[2]) / float(shape[2])
            scale_w = float(shape_t[3]) / float(shape[3])
            flow = F.upsample(flow, size=(shape_t[2], shape_t[3]), mode='bilinear')
            flow[:, 0, :, :] *= scale_h
            flow[:, 1, :, :] *= scale_w

        # warp input image with flow
        img1_warped = self.warp(img1, -flow)

        # L1 loss
        l1_val = L1loss(img1_warped, img2)

        # SSIM loss
        ssim_val = self.SSIM(img1_warped, img2).mean()

        # smoothness loss
        smooth_val = get_smooth_loss(flow, img1)

        # EPE
        epe = EPE(flow, flow_gt)

        # summary
        loss = l1_val * 0.01 + ssim_val + smooth_val
        self_loss_group = list()
        self_loss_group.append(l1_val)
        self_loss_group.append(ssim_val)
        self_loss_group.append(smooth_val)

        return [loss, self_loss_group, epe]


class SelfLossOrigin(nn.Module):
    """Layer to compute the unsupervised loss between a pair of images
    """

    def __init__(self, args):
        super(SelfLossOrigin, self).__init__()
        self.args = args
        self.SSIM = SSIM()
        self.warp = WarpingLayer(args)
        self.warp_homo = warp_kornia(args, '120_60', args.batch_size, 1.0, 1.0)

    def forward(self, imgs, flows, flow_gt):
        img1 = imgs[:, :, 0, :, :].contiguous()
        img2 = imgs[:, :, 1, :, :].contiguous()
        flow = flows[-1]

        # warp input image with homography matrix
        img1_warped_homo = self.warp_homo(img1)

        # scale flow to match the size of the input image
        shape = list(flow.size())
        shape_t = list(img1_warped_homo.size())
        if shape != shape_t:
            scale_h = float(shape_t[2]) / float(shape[2])
            scale_w = float(shape_t[3]) / float(shape[3])
            flow = F.upsample(flow, size=(shape_t[2], shape_t[3]), mode='bilinear')
            flow[:, 0, :, :] *= scale_h
            flow[:, 1, :, :] *= scale_w
            flow_gt = F.upsample(flow_gt, size=(shape_t[2], shape_t[3]), mode='bilinear')
            flow_gt[:, 0, :, :] *= scale_h
            flow_gt[:, 1, :, :] *= scale_w

        # warp input image with flow
        img1_warped = self.warp(img1_warped_homo, -flow)

        # L1 loss
        l1_val = L1loss(img1_warped, img2)

        # SSIM loss
        ssim_val = self.SSIM(img1_warped, img2).mean()

        # smoothness loss
        smooth_val = get_smooth_loss(flow, img1)

        # EPE
        epe = EPE(flow, flow_gt)

        # summary
        loss = l1_val * 0.01 + ssim_val + smooth_val
        self_loss_group = list()
        self_loss_group.append(l1_val)
        self_loss_group.append(ssim_val)
        self_loss_group.append(smooth_val)

        return [loss, self_loss_group, epe]
