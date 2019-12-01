
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from time import time
import sys
import argparse
import cv2
def pwcnet_args():

    args = argparse.Namespace()

    args.device = 'cuda'
    args.device = torch.device(args.device)
    args.num_workers = 8
    args.input_norm = False
    args.rgb_max = 255
    args.batch_norm = False
    args.lv_chs = [3, 16, 32, 64, 96, 128, 192]
    args.output_level = 4
    args.search_range = 4
    args.corr_activation = False
    args.residual = True
    args.context = False
    args.num_levels = len(args.lv_chs)

    return args