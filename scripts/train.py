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
from tensorboardX import SummaryWriter

from torchvision import transforms
from light.utils.distributed import *
from light.utils.logger import setup_logger
from light.utils.lr_scheduler import WarmupPolyLR
from light.utils.metric import SegmentationMetric
from light.data import get_segmentation_dataset
from light.model import get_segmentation_model
from light.utils.load_model import *
from light.nn import MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss

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
    parser.add_argument('--use_DataParallel', action='store_true', default=False, help='Forced parallel train')

    parser.add_argument('--nclass', type=int, default=6, metavar='N', help='class (default: 6)')
    parser.add_argument('--base-size', type=int, default=1208, help='base image size')
    parser.add_argument('--crop-size', type=int, default=1216, help='crop image size')
    parser.add_argument('--re-size', type=int, default=(1920, 1216), help='resize image')
    parser.add_argument('--scale', type=int, default=0.4, help='image size of generating light flow ')
    #
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 4)')
    # cuda
    parser.add_argument('--workers', '-j', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)

    # training hyper params

    parser.add_argument('--aux-weight', type=float, default=0.4, help='auxiliary loss weight')
    parser.add_argument('--ohem', action='store_true', default=False, help='OHEM Loss for cityscapes dataset')

    parser.add_argument('--epochs', type=int, default=175, metavar='N', help='number of epochs to train (default: 240)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')

    parser.add_argument('--lr', type=float, default=0.015, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--warmup-iters', type=int, default=0, help='warmup iters')
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3, help='lr = warmup_factor * lr')
    parser.add_argument('--warmup-method', type=str, default='linear', help='method of warmup')

    parser.add_argument('--save-dir', default='./runs/train', help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=20, help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='./runs/train', help='Directory for saving log models')
    parser.add_argument('--log-iter', type=int, default=5, help='print log every log-iter')
    parser.add_argument('--start-step', type=int, default=1, help='continue training from this steps')
    parser.add_argument('--val-epoch', type=int, default=20, help='run validation every val-epoch')

    parser.add_argument('--fixed_mobilenetv3', action='store_true', default=True, help='fix mobilenetv3')
    parser.add_argument('--fixed_pwcnet', action='store_true', default=True, help='fix pwcnet')
    parser.add_argument('--fixed_FFM120', action='store_true', default=False, help='fix FFM120')
    parser.add_argument('--fixed_FFM60', action='store_true', default=True, help='fix FFM60')

    # checkpoint and log
    parser.add_argument('--resume', type=str, default='./checkpoint/mobilenetv3_large_pwcnet_yuanqu_loosely.pth',
                        help='the whole net')
    parser.add_argument('--mobilenetv3-model-path', type=str,
                        default='./checkpoint/mobilenetv3_large_pwcnet_yuanqu_tightly.pth', help='baseline module')
    parser.add_argument('--pwcnet-model-path', type=str, default='./checkpoint/pwcnet_model.pkl', help='flow module')
    parser.add_argument('--FFM120-model-path', type=str, default='./checkpoint/FFM120.pth', help='only FFM120 module')
    parser.add_argument('--FFM60-model-path', type=str, default='./checkpoint/FFM60.pth', help='only FFM60  module')

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
    #
    # args.batch_size = 32
    # args.start_step = 1
    # args.epochs = 175
    # args.lr = 0.01
    #
    # args.fixed_mobilenetv3 = True
    # args.fixed_pwcnet = True
    # args.fixed_FFM120 = False
    # args.fixed_FFM60 = True
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        # dataset and dataloader
        train_data_kwargs = {'transform': input_transform,
                             'base_size': args.base_size, 'crop_size': args.crop_size, 're_size': args.re_size,
                             }
        trainset = get_segmentation_dataset(args.dataset, args=args, split='train', mode='train_onlyrs',
                                            **train_data_kwargs)

        args.iters_per_epoch = len(trainset) // (args.num_gpus * args.batch_size)
        args.max_iters = args.epochs * args.iters_per_epoch

        train_sampler = make_data_sampler(trainset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iters)
        self.train_loader = data.DataLoader(dataset=trainset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)

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

        self.model = load_modules(args, self.model)
        self.model = fix_model(args, self.model)

        # optimizer
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # create criterion
        if args.ohem:
            min_kept = int(args.batch_size // args.num_gpus * args.crop_size ** 2 // 16)
            self.criterion = MixSoftmaxCrossEntropyOHEMLoss(args.aux, args.aux_weight, min_kept=min_kept,
                                                            ignore_index=-1).to(self.device)
        else:
            self.criterion = MixSoftmaxCrossEntropyLoss(args.aux, args.aux_weight, ignore_index=-1).to(self.device)

        # lr scheduling
        self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                         max_iters=args.max_iters,
                                         power=0.9,
                                         warmup_factor=args.warmup_factor,
                                         warmup_iters=args.warmup_iters,
                                         warmup_method=args.warmup_method)

        if args.use_DataParallel:
            self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

        elif args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[args.local_rank],
                                                             output_device=args.local_rank,
                                                             find_unused_parameters=True)

        # evaluation metrics
        self.metric_120 = SegmentationMetric(trainset.num_class)
        self.metric_60 = SegmentationMetric(trainset.num_class)

        self.best_pred = 0.0

    def train(self, writer):
        save_to_disk = get_rank() == 0
        epochs, max_iters = self.args.epochs, self.args.max_iters
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.args.iters_per_epoch
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        start_time = time.time()
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.model.train()
        for iteration, (images, targets, _) in enumerate(self.train_loader):
            iteration += self.args.start_step
            self.lr_scheduler.step()

            for index in range(len(images)):
                images[index] = images[index].to(self.device)
            for index in range(len(targets)):
                targets[index] = targets[index].to(self.device)

            outputs = self.model(images)
            loss_dict = self.criterion(outputs, targets)
            losses = sum(loss for loss in loss_dict.values())
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], iteration)
            writer.add_scalar("Loss/train_loss", losses_reduced.item(), iteration)

            if iteration % log_per_iters == 0 and save_to_disk:
                logger.info("Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || Estimated Time: {}".format(
                    iteration, max_iters, self.optimizer.param_groups[0]['lr'], losses_reduced.item(),
                    eta_string))

            if iteration % save_per_iters == 0 and save_to_disk:
                print('saving......')
                save_checkpoint(self.model, self.args, iteration=iteration, is_best=False)
                print('save over!')

            if (iteration % val_per_iters == 0):
                print('evaluating...')
                self.validate(iteration, writer)
                self.model.train()
                print('eval over!')

        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(total_training_str, total_training_time / max_iters))

    def validate(self, iteration, writer):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
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

            loss_dict = self.criterion(outputs, target)
            loss_dict_120 = loss_dict['loss_120']
            loss_dict_60 = loss_dict['loss_60']

            loss_dict_reduced_120 = reduce_loss_dict(loss_dict_120)
            loss_dict_reduced_60 = reduce_loss_dict(loss_dict_60)

            loss[0].append(loss_dict_reduced_120)
            loss[1].append(loss_dict_reduced_60)

        pixAcc_120, mIoU_120, Iou_120 = self.metric_120.get()
        val_loss_120 = sum(loss[0]) / len(loss[0])
        val_mIou_120 = mIoU_120
        val_mpixAcc_120 = pixAcc_120
        logger.info("120  Loss: {:.3f}, Validation mpixAcc: {:.3f}, mIoU: {:.3f}".format(val_loss_120, val_mpixAcc_120,
                                                                                         val_mIou_120))
        writer.add_scalar("Loss/val120_loss", val_loss_120, iteration)
        writer.add_scalar("Result/val120_mIou", val_mIou_120, iteration)
        writer.add_scalar("Result/val120_Acc", val_mpixAcc_120, iteration)

        for i, j in enumerate(Iou_120):
            logger.info("class {:d} : {:.3f}".format(i, j))
            writer.add_scalar("Class120/class_{}".format(i), Iou_120[i], iteration)

        pixAcc_60, mIoU_60, Iou_60 = self.metric_60.get()
        val_loss_60 = sum(loss[1]) / len(loss[1])
        val_mIou_60 = mIoU_60
        val_mpixAcc_60 = pixAcc_60
        logger.info("60  Loss: {:.3f}, Validation mpixAcc: {:.3f}, mIoU: {:.3f}".format(val_loss_60, val_mpixAcc_60,
                                                                                        val_mIou_60))
        writer.add_scalar("Loss/val60_loss", val_loss_60, iteration)
        writer.add_scalar("Result/val60_mIou", val_mIou_60, iteration)
        writer.add_scalar("Result/val60_Acc", val_mpixAcc_60, iteration)

        for i, j in enumerate(Iou_60):
            logger.info("class {:d} : {:.3f}".format(i, j))
            writer.add_scalar("Class60/class_{}".format(i), Iou_60[i], iteration)

        new_pred = (val_mIou_60 + val_mIou_120) / 2.0
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        save_checkpoint(self.model, self.args, iteration, is_best)
        synchronize()


def save_checkpoint(model, args, iteration, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_newest_model_{}.pth'.format(args.model, args.dataset, iteration)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = '{}_{}_best_model.pth'.format(args.model, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


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
    filename = ts + '_b' + str(args.batch_size) + '_lr' + str(args.lr) + \
               '_F' + args.FFM_120[4:] + '120' + str(args.fixed_FFM120) + '_F' + args.FFM_60[4:] + '60' + str(
        args.fixed_FFM60)

    args.log_dir = os.path.join(args.log_dir, args.model_mode, filename)
    args.save_dir = args.log_dir

    logger = setup_logger(args.model, args.log_dir, get_rank(),
                          filename='{}_{}_'.format(args.model, args.dataset) + filename + '_log.txt')
    writer = SummaryWriter(log_dir=args.save_dir)

    logger.info("dataset process is mine  xaiv  class=6")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train(writer)
    torch.cuda.empty_cache()
