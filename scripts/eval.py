import os
import sys
import time
import shutil
import datetime
import argparse

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from light.utils.distributed import *
from light.utils.logger import setup_logger
from light.utils.metric import SegmentationMetric
from light.data import get_segmentation_dataset
from light.model import get_segmentation_model
from light.utils.load_model import *

from light.utils.visualize import get_color_pallete
import numpy as np
import scipy.misc as misc

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.cuda.set_device(0)


def parse_args():
    parser = argparse.ArgumentParser(description='Light Model for Segmentation')
    # model
    parser.add_argument('--model', type=str, default='mobilenetv3_large_pwcflow',
                        help='backbone model name (default: mobilenet)')
    parser.add_argument('--model-mode', type=str, default='mobilenetv3_tightly',
                        help='model name (default: mobilenetv3)')
    # choose from ['mobilenetv3', 'mobilenetv3_loosely', 'mobilenetv3_tightly']
    parser.add_argument('--aux', action='store_true', default=False, help='Auxiliary loss')

    # dataset path
    parser.add_argument('--dataset', type=str, default='yuanqu', help='dataset name (default: citys)')
    parser.add_argument('--dataset-path', type=str, default='../data',
                        help='dataset path')
    parser.add_argument('--train-file', type=str, default='image_train',
                        help='the train-set filename under dataset path')
    parser.add_argument('--val-file', type=str, default='image_val',
                        help='the val-set filename under dataset path')
    parser.add_argument('--label-file', type=str, default='image_label',
                        help='the label-set(include train and val) filename under dataset path')

    # data
    parser.add_argument('--nclass', type=int, default=6, metavar='N', help='class (default: 6)')
    parser.add_argument('--base-size', type=int, default=1208, help='base image size')
    parser.add_argument('--crop-size', type=int, default=1216, help='crop image size')
    parser.add_argument('--re-size', type=int, default=(1920, 1216), help='resize image')
    parser.add_argument('--scale', type=int, default=0.4, help='image size of generating light flow ')
    #
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 4)')
    # cuda
    parser.add_argument('--workers', '-j', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)

    # checkpoint and log
    parser.add_argument('--resume', type=str, default='./checkpoint/mobilenetv3_large_pwcnet_yuanqu_tightly.pth',
                        help='the whole net')
    parser.add_argument('--mobilenetv3-model-path', type=str, default='./checkpoint/mobilenetv3_large_yuanqu.pth',
                        help='baseline module')
    parser.add_argument('--pwcnet-model-path', type=str, default='./checkpoint/pwcnet_model.pkl', help='flow module')
    parser.add_argument('--FFM120-model-path', type=str, default='./checkpoint/FFM120.pth', help='only FFM120 module')
    parser.add_argument('--FFM60-model-path', type=str, default='./checkpoint/FFM60.pth', help='only FFM60  module')

    parser.add_argument('--log-dir', default='./runs/eval', help='Directory for saving log models')

    # （unique set)
    # which result
    parser.add_argument('--combined', action='store_true', default=True, help='whether to store combined prediction')
    parser.add_argument('--save-pre', action='store_true', default=True, help='whether to store prediction')
    parser.add_argument('--save-pre-path', type=str, default='../data/eval_results',
                        help='the path to save evaluator prediction')
    args = parser.parse_args()
    #######################################################################################################
    # args.scale = 0.4
    # args.workers = 4
    #
    # args.resume = './checkpoint/mobilenetv3_large_yuanqu_newest_frompsp.pth'
    # args.pwcnet_model_path = './checkpoint/pwcnet_model.pkl'
    # args.FFM60_model_path = './checkpoint/FFM60.pth'
    # args.FFM120_model_path = './checkpoint/FFM120.pth'
    #
    # args.dataset_path = '/workspace/ShareData/Data/yuanqu/video/all/'
    # args.train_file = 'image_train'
    # args.val_file = 'image_val'
    # args.label_file = 'label_L'

    return args


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        # dataset and dataloader
        val60_data_kwargs = {'transform': input_transform,
                             'base_size': args.base_size, 'crop_size': args.crop_size, 're_size': args.re_size,
                             }
        valset = get_segmentation_dataset(args.dataset, args=args, split='val', mode='val_onlyrs', **val60_data_kwargs)

        val_sampler = make_data_sampler(valset, True, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size)
        self.val60_loader = data.DataLoader(dataset=valset,
                                            batch_sampler=val_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(args.model,
                                            dataset=args.dataset,
                                            args=self.args,
                                            norm_layer=BatchNorm2d).to(self.device)

        self.model = load_model(args.resume, self.model)

        # evaluation metrics
        self.metric_120 = SegmentationMetric(valset.num_class)
        self.metric_60 = SegmentationMetric(valset.num_class)

        self.best_pred = 0.0

    def evaluate(self):
        is_best = False
        self.metric_120.reset()
        self.metric_60.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()

        loss = [[], []]
        for i, (image, target, _) in enumerate(self.val60_loader):
            for index in range(len(image)):
                image[index] = image[index].to(self.device)
            for index in range(len(target)):
                target[index] = target[index].to(self.device)

            with torch.no_grad():
                outputs = model(image)

            self.metric_120.update(outputs[0][0], target[0])
            self.metric_60.update(outputs[0][1], target[1])

            if self.args.save_pre:
                self.save_pred(image, target, _, outputs)

        pixAcc_120, mIoU_120, Iou_120 = self.metric_120.get()
        val_mIou_120 = mIoU_120
        val_mpixAcc_120 = pixAcc_120
        logger.info("120 Validation: mpixAcc: {:.3f}, mIoU: {:.3f}".format(val_mpixAcc_120, val_mIou_120))

        for i, j in enumerate(Iou_120):
            logger.info("class {:d} : {:.3f}".format(i, j))

        pixAcc_60, mIoU_60, Iou_60 = self.metric_60.get()
        val_mIou_60 = mIoU_60
        val_mpixAcc_60 = pixAcc_60
        logger.info("60 Validation: mpixAcc: {:.3f}, mIoU: {:.3f}".format(val_mpixAcc_60, val_mIou_60))

        for i, j in enumerate(Iou_60):
            logger.info("class {:d} : {:.3f}".format(i, j))
        synchronize()

    def save_pred(self, image, target, image_name, outputs):
        def unnormlize(img, mean, std):
            mean = np.expand_dims(mean, axis=0)
            mean = np.repeat(mean, img.shape[1], axis=0)
            mean = np.expand_dims(mean, axis=0)
            mean = np.repeat(mean, img.shape[0], axis=0)

            std = np.expand_dims(std, axis=0)
            std = np.repeat(std, img.shape[1], axis=0)
            std = np.expand_dims(std, axis=0)
            std = np.repeat(std, img.shape[0], axis=0)

            img = (img * std + mean) * 255.

            return img

        mean = np.array([.485, .456, .406])
        std = np.array([.229, .224, .225])
        ################################### 120 ##########################################
        pred_120 = torch.argmax(outputs[0][0], 1)
        pred_120 = pred_120.cpu().data.numpy()
        predict_120 = pred_120.squeeze(0)

        mask_120 = get_color_pallete(predict_120, self.args.dataset)
        mask_120 = np.asarray(mask_120.convert('RGB'))
        misc.imsave(
            os.path.join(self.args.save_pre_path, str(image_name[1])[2:-2] + '_' + self.args.model_mode + '.png'),
            mask_120)
        if self.args.combined:
            image_120 = image[0]
            image_120 = image_120.cpu().data.numpy()[0].transpose(1, 2, 0)
            image_120 = np.array(unnormlize(image_120, mean, std), dtype=np.int32)

            target_120 = target[0].cpu().data.numpy()
            target_120 = target_120.squeeze(0)
            target_120 = get_color_pallete(target_120, self.args.dataset)
            target_120 = np.asarray(target_120.convert('RGB'))

            combine1 = np.concatenate((image_120, image_120 * 0.5 + mask_120 * 0.5), axis=1)
            combine2 = np.concatenate((target_120, mask_120), axis=1)
            mask_120 = np.concatenate((combine1, combine2), axis=0)

        misc.imsave(
            os.path.join(self.args.save_pre_path, str(image_name[0])[2:-2] + '_' + self.args.model_mode + '_4.png'),
            mask_120)

        ################################### 60 ##########################################
        pred_60 = torch.argmax(outputs[0][1], 1)
        pred_60 = pred_60.cpu().data.numpy()
        predict_60 = pred_60.squeeze(0)

        mask_60 = get_color_pallete(predict_60, self.args.dataset)
        mask_60 = np.asarray(mask_60.convert('RGB'))
        misc.imsave(
            os.path.join(self.args.save_pre_path, str(image_name[1])[2:-2] + '_' + self.args.model_mode + '.png'),
            mask_60)
        if self.args.combined:
            image_60 = image[1]
            image_60 = image_60.cpu().data.numpy()[0].transpose(1, 2, 0)
            image_60 = np.array(unnormlize(image_60, mean, std), dtype=np.int32)

            target_60 = target[1].cpu().data.numpy()
            target_60 = target_60.squeeze(0)
            target_60 = get_color_pallete(target_60, self.args.dataset)
            target_60 = np.asarray(target_60.convert('RGB'))

            combine1 = np.concatenate((image_60, image_60 * 0.5 + mask_60 * 0.5), axis=1)
            combine2 = np.concatenate((target_60, mask_60), axis=1)
            mask_60 = np.concatenate((combine1, combine2), axis=0)

        misc.imsave(
            os.path.join(self.args.save_pre_path, str(image_name[1])[2:-2] + '_' + self.args.model_mode + '_4.png'),
            mask_60)


if __name__ == '__main__':
    args = parse_args()
    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"

    if args.distributed:
        args.lr = args.lr * num_gpus
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    args.distributed = False

    args.model_mode_list = ['mobilenetv3', 'mobilenetv3_loosely', 'mobilenetv3_tightly']
    args.FFM_list = ['FFM_res', 'FFM_res131', 'FFM_res131_120', 'FFM_res_120']  # parallel_fusion
    args.FFM_120 = 'FFM_res131_120'
    args.FFM_60 = 'FFM_res'
    if args.FFM_120 not in args.FFM_list or args.FFM_60 not in args.FFM_list:
        raise RuntimeError("Fusion module name is ERROR!! " + "\n")
    if args.model_mode not in args.model_mode_list:
        raise RuntimeError("Model mode name is ERROR!! " + "\n")

    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    filename = ts + '_combined' + str(args.combined)

    args.log_dir = os.path.join(args.log_dir, args.model_mode, filename)

    logger = setup_logger(args.model, args.log_dir, get_rank(),
                          filename='{}_{}_'.format(args.model, args.dataset) + filename + '_log.txt')

    logger.info("dataset process is mine  xaiv  class=6")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    if args.save_pre:
        if not os.path.exists(args.save_pre_path):
            os.makedirs(args.save_pre_path)
        args.batch_size = 1

    evaluator = Evaluator(args)
    evaluator.evaluate()
    torch.cuda.empty_cache()


