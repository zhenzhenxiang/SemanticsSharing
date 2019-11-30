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

from model import Net
from modules import WarpingLayer
from losses import L1loss, L2loss, training_loss, robust_training_loss, MultiScale, EPE, L2Loss
from losses_unsupervised import get_smooth_loss, SSIM, SelfLoss
from dataset import (FlyingChairs, FlyingThings, Sintel, SintelFinal, SintelClean, KITTI, YuanquSimulate, YuanquLiteFlowNet)

import tensorflow as tf
from summary import summary as summary_
from logger import Logger
from pathlib import Path
from flow_utils import (vis_flow, save_flow)


def main():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # mode selection
    # ============================================================
    modes = parser.add_subparsers(title='modes',
                                  description='valid modes',
                                  help='additional help',
                                  dest='subparser_name')

    parser.set_defaults(func=hello_world)
    summary_parser = modes.add_parser('summary')
    summary_parser.set_defaults(func=summary)
    train_parser = modes.add_parser('train')
    train_parser.set_defaults(func=train)
    pred_parser = modes.add_parser('pred')
    pred_parser.set_defaults(func=pred)
    test_parser = modes.add_parser('eval')
    test_parser.set_defaults(func=test)

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
    parser.add_argument('--lv_chs', nargs='+', type=int, default=[3, 16, 32, 64, 96, 128, 192])
    parser.add_argument('--output_level', type=int, default=4)

    # correlation args
    parser.add_argument('--corr', type=str, default='cost_volume')
    parser.add_argument('--search_range', type=int, default=4)
    parser.add_argument('--corr_activation', action='store_true')

    # flow estimator
    parser.add_argument('--residual', action='store_true', default=True)

    # args for summary
    # ============================================================
    summary_parser.add_argument('-i', '--input_shape', type=int, nargs='*', default=(3, 2, 384, 448))

    # args for train
    # ============================================================
    # dataflow
    train_parser.add_argument('--crop_type', type=str, default='random')
    train_parser.add_argument('--crop_shape', type=int, nargs='+', default=[384, 448])
    train_parser.add_argument('--resize_shape', nargs=2, type=int, default=None)
    train_parser.add_argument('--resize_scale', type=float, default=None)
    train_parser.add_argument('--load', type=str, default=None)

    train_parser.add_argument('--batch_size', default=8, type=int, help='mini-batch size')
    train_parser.add_argument('--dataset_dir', type=str, required=True)
    train_parser.add_argument('--dataset', type=str,
                              choices=['FlyingChairs', 'FlyingThings', 'SintelFinal', 'SintelClean', 'KITTI', 'YuanquSimulate', 'YuanquLiteFlowNet'],
                              required=True)

    train_parser.add_argument('--context', action='store_true', default=False, help='With context module')

    # loss
    train_parser.add_argument('--weights', nargs='+', type=float, default=[0.32, 0.08, 0.02, 0.01, 0.005])
    train_parser.add_argument('--epsilon', default=0.02)
    train_parser.add_argument('--q', type=int, default=0.4)
    train_parser.add_argument('--loss', type=str, default='MultiScale', choices=['MultiScale', 'L2Loss', 'SelfLoss'])
    train_parser.add_argument('--optimizer', type=str, default='Adam')

    # optimize
    train_parser.add_argument('--lr', type=float, default=1e-4)
    train_parser.add_argument('--momentum', default=4e-4)
    train_parser.add_argument('--beta', default=0.99)
    train_parser.add_argument('--weight_decay', type=float, default=4e-4)
    train_parser.add_argument('--total_step', type=int, default=200 * 1000)

    # summary & log args
    train_parser.add_argument('--log_dir', default='train_log/' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    train_parser.add_argument('--summary_interval', type=int, default=100)
    train_parser.add_argument('--log_interval', type=int, default=100)
    train_parser.add_argument('--checkpoint_interval', type=int, default=100)
    train_parser.add_argument('--gif_input', type=str, default=None)
    train_parser.add_argument('--gif_output', type=str, default='gif')
    train_parser.add_argument('--gif_interval', type=int, default=100)
    train_parser.add_argument('--max_output', type=int, default=3)

    # args for predict
    # ============================================================
    # pred_parser.add_argument('-i', '--input', nargs=2, required=True)
    pred_parser.add_argument('-i', '--input', type=str, required=False, help='Text file containing image pairs')
    pred_parser.add_argument('--input_dir', type=str, required=True, help='Folder containing text file and images pairs')
    pred_parser.add_argument('-o', '--output', type=str, required=True, help='Folder to save output flows')
    pred_parser.add_argument('--load', type=str, required=True)
    pred_parser.add_argument('--batch_size', default=8, type=int, help='mini-batch size')
    pred_parser.add_argument('--context', action='store_true', default=False, help='With context module')


    # args for test
    # ============================================================
    test_parser.add_argument('--load', type=str, required=True)
    test_parser.add_argument('--dataset_dir', type=str, required=True)
    test_parser.add_argument('--dataset', type=str,
                             choices=['FlyingChairs', 'FlyingThings', 'SintelFinal', 'SintelClean', 'KITTI', 'YuanquSimulate', 'YuanquLiteFlowNet'],
                             required=True)
    test_parser.add_argument('--batch_size', default=8, type=int, help='mini-batch size')
    test_parser.add_argument('--context', action='store_true', default=False, help='With context module')

    args = parser.parse_args()

    args.num_levels = len(args.lv_chs)
    args.device = torch.device(args.device)

    # check args
    # ============================================================
    if args.subparser_name == 'train':
        assert len(args.weights) >= args.output_level + 1

    args.func(args)


def hello_world(args):
    from functools import reduce
    from operator import mul
    model = Net(args).to(args.device)
    state = model.state_dict()
    total_size = 0
    for key, value in state.items():
        print(f'{key}: {value.size()}')
        total_size += reduce(mul, value.size())
    print(f'Parameters: {total_size} Size: {total_size * 4 / 1024 / 1024} MB')


def summary(args):
    model = Net(args).to(args.device)
    summary_(model, args.input_shape)


def train(args):
    # Build Model
    # ============================================================
    model = Net(args).to(args.device)
    if args.load is not None:
        model.load_state_dict(torch.load(args.load))

    # Prepare Dataloader
    # ============================================================
    train_dataset = eval(args.dataset)(args.dataset_dir, 'train')
    eval_dataset = eval(args.dataset)(args.dataset_dir, 'test')

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers,
                             pin_memory=True)

    # Init logger
    logger = Logger(args.log_dir)
    p_log = Path(args.log_dir)

    forward_time = 0
    backward_time = 0

    # Start training
    # ============================================================
    data_iter = iter(train_loader)
    iter_per_epoch = len(train_loader)
    criterion = eval(args.loss)(args)

    # build criterion
    optimizer = eval('torch.optim.' + args.optimizer)(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = get_lr_scheduler(optimizer, args.dataset)

    total_loss = 0
    total_epe = 0
    total_loss_levels = [0] * args.num_levels
    total_epe_levels = [0] * args.num_levels
    total_self_loss_group = [0] * 3
    # training
    # ============================================================
    for step in range(1, args.total_step + 1):
        # Reset the data_iter
        if (step) % iter_per_epoch == 0: data_iter = iter(train_loader)

        # Load Data
        # ============================================================
        data, target = next(data_iter)

        # shape: B,3,H,W
        squeezer = partial(torch.squeeze, dim=2)
        # shape: B,2,H,W
        data, target = [d.to(args.device) for d in data], [t.to(args.device) for t in target]

        x1_raw = data[0][:, :, 0, :, :]
        x2_raw = data[0][:, :, 1, :, :]
        if data[0].size(0) != args.batch_size: continue
        flow_gt = target[0]

        # Forward Pass
        # ============================================================
        t_forward = time.time()
        flows, summaries = model(data[0])
        forward_time += time.time() - t_forward

        # Compute Loss
        # ============================================================
        if args.loss == 'MultiScale':
            loss, epe, loss_levels, epe_levels = criterion(flows, flow_gt)
            for l, (loss_, epe_) in enumerate(zip(loss_levels, epe_levels)):
                total_loss_levels[l] += loss_.item()
                total_epe_levels[l] += epe_.item()
        elif args.loss == 'SelfLoss':
            loss, self_loss_group, epe = criterion(data[0], flows, flow_gt)
            for l, loss_ in enumerate(self_loss_group):
                total_self_loss_group[l] += loss_.item()
        else:
            loss, epe = criterion(flows, flow_gt)

        total_loss += loss.item()
        total_epe += epe.item()

        # backward
        # ============================================================
        t_backward = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        backward_time += time.time() - t_backward

        # Collect Summaries & Output Logs
        # ============================================================
        if step % args.summary_interval == 0:
            # Scalar Summaries
            # ============================================================
            logger.scalar_summary('lr', optimizer.param_groups[0]['lr'], step)
            logger.scalar_summary('loss', total_loss / step, step)
            logger.scalar_summary('EPE', total_epe / step, step)

            if args.loss == 'MultiScale':
                for l, (loss_, epe_) in enumerate(zip(loss_levels, epe_levels)):
                    logger.scalar_summary(f'loss_lv{l}', total_loss_levels[l] / step, step)
                    logger.scalar_summary(f'EPE_lv{l}', total_epe_levels[l] / step, step)

            if args.loss == 'SelfLoss':
                logger.scalar_summary(f'L1_loss', total_self_loss_group[0] / step, step)
                logger.scalar_summary(f'SSIM_loss', total_self_loss_group[1] / step, step)
                logger.scalar_summary(f'Sooth_loss', total_self_loss_group[2] / step, step)

        # save model
        if step % args.checkpoint_interval == 0:
            torch.save(model.state_dict(), str(p_log / f'{step}.pkl'))
        # print log
        if step % args.log_interval == 0:
            print(
                f'Step [{step}/{args.total_step}], Loss: {total_loss / step:.4f}, EPE: {total_epe / step:.4f}, Forward: {forward_time / step * 1000:.3f} ms, Backward: {backward_time / step * 1000:.3f} ms')

        if step % args.gif_interval == 0:
            ...


def pred(args):
    # Get environment
    # Build Model
    # ============================================================
    model = Net(args)
    model.load_state_dict(torch.load(args.load))
    model.to(args.device).eval()
    warp = WarpingLayer(args)

    # Load Data
    # ============================================================
    input_list = list()
    with open(os.path.join(args.input_dir, args.input), 'r') as input_file:
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

            H, W = x1_raw.shape[:2]

            x1_raw = x1_raw[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
            x2_raw = x2_raw[np.newaxis, :, :, :].transpose(0, 3, 1, 2)

            x = np.stack([x1_raw, x2_raw], axis=2)
            x_list.append(x)

        x = np.concatenate(x_list, axis=0)
        x = torch.Tensor(x).to(args.device)

        # Forward Pass
        # ============================================================
        with torch.no_grad():
            flows, summaries = model(x)

        # warp input
        o = flows[-1]
        shape = list(o.size())
        shape_t = list(x[:, :, 0, :, :].size())
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
            img_warped_file = flow_file.replace('.flo', '_flow_warped.png')

            save_flow(os.path.join(flow_dir,flow_file), flow)
            flow_vis = vis_flow(flow)
            imageio.imwrite(os.path.join(flow_vis_dir,flow_vis_file), flow_vis)
            imageio.imwrite(os.path.join(img_warped_dir,img_warped_file), x_warped[i, :, :, :].transpose(1, 2, 0))

    print('Finished!')

def test(args):
    print('load model...')
    model = Net(args)
    model.load_state_dict(torch.load(args.load))
    model.to(args.device).eval()

    warp = WarpingLayer(args)
    ssim = SSIM()

    print('build eval dataset...')
    test_dataset = eval(args.dataset)(args.dataset_dir, 'test')
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True)

    total_batches = len(test_loader)

    # logs
    # ============================================================
    time_logs = []
    total_epe = 0
    total_L1 = 0
    total_smooth = 0
    total_SSIM = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        # Forward Pass
        # ============================================================
        data, target = [d.to(args.device) for d in data], [t.to(args.device) for t in target]
        t_start = time.time()
        with torch.no_grad():
            flows, summaries = model(data[0])
        time_logs.append(time.time() - t_start)

        # Compute EPE
        # ============================================================
        o = flows[-1]
        t = target[0]
        # check shape
        shape = list(o.size())
        shape_t = list(t.size())
        if shape != shape_t:
            scale_h = float(shape_t[2]) / float(shape[2])
            scale_w = float(shape_t[3]) / float(shape[3])
            o = F.upsample(o, size=(shape_t[2], shape_t[3]), mode='bilinear')
            o[:, 0, :, :] *= scale_h
            o[:, 1, :, :] *= scale_w
        epe = EPE(o, t)
        total_epe += epe.item()

        # Warp input
        # ============================================================
        input_img = data[0][:,:,0,:,:].contiguous()
        input_img_warped = warp(input_img, -o)

        # Compute L1
        # ============================================================
        target_img = data[0][:,:,1,:,:].contiguous()
        l1_val = L1loss(input_img_warped, target_img)
        total_L1 += l1_val.item()

        # Compute SSIM
        # ============================================================
        ssim_val = ssim(input_img_warped, target_img)
        total_SSIM += ssim_val.mean().item()

        # Compute smoothness
        # ============================================================
        smooth_val = get_smooth_loss(o, input_img)
        total_smooth += smooth_val.item()

        print(
            f'[{batch_idx + 1}/{total_batches}]  Time: {np.mean(time_logs):.3f}s  EPE:{total_epe / (batch_idx + 1):.3f}  '
            f'L1:{total_L1 / (batch_idx + 1):.3f}  SSIM:{total_SSIM / (batch_idx + 1):.3f}  '
            f'Smooth:{total_smooth / (batch_idx + 1):.3f}')


def get_lr_scheduler(optimizer, dataset_name):
    if dataset_name in ['FlyingChairs', 'FlyingThings']:
        milestones = [150000, 250000, 350000, 400000]
    elif dataset_name == 'KITTI':
        milestones = [1000, 1500]
    elif dataset_name in ['SintelFinal', 'SintelClean']:
        milestones = [600, 900]
    elif dataset_name in ['YuanquSimulate', 'YuanquLiteFlowNet']:
        milestones = [100000, 150000, 200000, 250000]
    else:
        raise ValueError('Unknown dataset name {}'.format(dataset_name))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5)
    return scheduler


if __name__ == '__main__':
    main()