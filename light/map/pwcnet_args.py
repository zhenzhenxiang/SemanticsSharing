
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from time import time
import sys
import argparse
import cv2
def pwcnet_args():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # mode selection
    # ============================================================
    modes = parser.add_subparsers(title='modes',
                                  description='valid modes',
                                  help='additional help',
                                  dest='subparser_name')
    parser.add_argument('--device', type=str, default='cuda')

    # dataset
    parser.add_argument('--num_workers', default=8, type=int, help='num of workers')

    # normalization args
    parser.add_argument('--input-norm', action='store_true')
    parser.add_argument('--rgb_max', type=float, default=255)
    parser.add_argument('--batch-norm', action='store_true')

    # pyramid args
    parser.add_argument('--lv_chs', nargs='+', type=int, default=[3, 16, 32, 64, 96, 128, 192])
    parser.add_argument('--output_level', type=int, default=4)

    # correlation args
    parser.add_argument('--corr', type=str, default='cost_volume')
    parser.add_argument('--search_range', type=int, default=4)
    parser.add_argument('--corr_activation', action='store_true')

    # flow estimator
    parser.add_argument('--residual', action='store_true', default=True)
    
    parser.add_argument('--context', action='store_true', default=False, help='With context module')
    
    
    
    # main
    parser.add_argument('--batch_size', default=4, type=int, help='mini-batch size')
    parser.add_argument('--load', default='./pwcnet_model.pkl', type=str)
    parser.add_argument('--input_dir', default='./test_data', type=str,  help='Folder containing text file and images pairs')
    parser.add_argument('-i', '--input', default='untitled.txt', type=str, help='Text file containing image pairs')
    parser.add_argument('-o', '--output', default='./pwcnet_test_output', type=str, help='Folder to save output flows')
    
   
    args = parser.parse_args()
    
    args.num_levels = len(args.lv_chs)
    args.device = torch.device(args.device)

    # check args
    # ============================================================
    if args.subparser_name == 'train':
        assert len(args.weights) >= args.output_level + 1
    return args