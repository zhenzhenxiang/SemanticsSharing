from datetime import datetime
import argparse
import imageio
import cv2
import numpy as np
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import os
from collections import OrderedDict

# from light.map.pwcnet_model_mobilenetv3 import Net_MobileNetV3
# from light.map.pwcnet_modules import WarpingLayer
# from losses import L1loss, L2loss, training_loss, robust_training_loss, MultiScale, MultiScale_MobileNetV3, EPE, L2Loss
# from losses_unsupervised import get_smooth_loss, SSIM, SelfLoss
# from dataset import (FlyingChairs, FlyingThings, Sintel, SintelFinal, SintelClean, KITTI, YuanquSimulate, YuanquLiteFlowNet)

# import tensorflow as tf
# from summary import summary as summary_
# from logger import Logger
# from pathlib import Path
# from flow_utils import (vis_flow, save_flow)


def pwcnet_args_mobilenetv3():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # mode selection
    # ============================================================
    modes = parser.add_subparsers(title='modes',
                                  description='valid modes',
                                  help='additional help',
                                  dest='subparser_name')

    # shared args
    # ============================================================
    parser.add_argument('--device', type=str, default='cuda')

    # dataset
    parser.add_argument('--num_workers', default=8, type=int, help='num of workers')

    # normalization args
    parser.add_argument('--input-norm', action='store_true')
    parser.add_argument('--rgb_max', type=float, default=255)
    parser.add_argument('--batch-norm', action='store_true')

    # pyramid args
    parser.add_argument('--lv_chs', nargs='+', type=int, default=[3, 16, 24, 40, 112])
    parser.add_argument('--output_level', type=int, default=4)

    # correlation args
    parser.add_argument('--corr', type=str, default='cost_volume')
    parser.add_argument('--search_range', type=int, default=4)
    parser.add_argument('--corr_activation', action='store_true')

    # flow estimator
    parser.add_argument('--residual', action='store_true', default=True)

 


    # args for predict
    # ============================================================
    # pred_parser.add_argument('-i', '--input', nargs=2, required=True)
    parser.add_argument('--batch_size', default=4, type=int, help='mini-batch size')
    parser.add_argument('--load', default='../../scripts/checkpoint/200000.pkl', type=str)
    parser.add_argument('--input_dir', default='./test_data', type=str,  help='Folder containing text file and images pairs')
    parser.add_argument('-i', '--input', default='untitled.txt', type=str, help='Text file containing image pairs')
    parser.add_argument('-o', '--output', default='./pwcnet_test_output', type=str, help='Folder to save output flows')
    parser.add_argument('--context', action='store_true', default=False, help='With context module')


    args = parser.parse_args()

    args.num_levels = len(args.lv_chs)
    args.device = torch.device(args.device)

    # check args
    # ============================================================
    # if args.subparser_name == 'train':
    #     assert len(args.weights) >= args.output_level + 1

    return args


# def hello_world(args):
#     from functools import reduce
#     from operator import mul
#     model = Net_MobileNetV3(args).to(args.device)
#     state = model.state_dict()
#     total_size = 0
#     for key, value in state.items():
#         print(f'{key}: {value.size()}')
#         total_size += reduce(mul, value.size())
#     print(f'Parameters: {total_size} Size: {total_size * 4 / 1024 / 1024} MB')