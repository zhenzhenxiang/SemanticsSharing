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


def pwcnet_args_mobilenetv3():

    args = argparse.Namespace()

    args.device = 'cuda'
    args.device = torch.device(args.device)
    args.num_workers = 8
    args.input_norm = False
    args.rgb_max = 255
    args.batch_norm = False
    args.lv_chs = [3, 16, 24, 40, 112]
    args.output_level = 4
    args.search_range = 4
    args.corr_activation = False
    args.residual = True
    args.context = False
    args.num_levels = len(args.lv_chs)

    return args

