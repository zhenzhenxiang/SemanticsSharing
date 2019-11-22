import os
import sys
import time
import shutil
import datetime
import argparse
import imageio

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from light.utils.distributed import *
from light.data import get_segmentation_dataset
from light.model import get_segmentation_model, convert_state_dict


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.cuda.set_device(0)


def parse_args():
    parser = argparse.ArgumentParser(description='Light Model for Segmentation')
    # model and dataset
    parser.add_argument('--model', type=str, default='mobilenetv3_large_liteflow',
                        help='model name (default: mobilenet)')
    parser.add_argument('--dataset', type=str, default='yuanqu',
                        help='dataset name (default: citys)')
    parser.add_argument('--data_angle', type=str, default=60120,
                        help='dataset name (default: fov120)')
    parser.add_argument('--base-size', type=int, default=1208,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=1216,
                        help='crop image size')
    parser.add_argument('--re_size', type=int, default=(1920, 1216),
                        help='resize for L-all net')
    ######################################### scale
    parser.add_argument('--scale', type=int, default=0.4,
                        help='resize for L-all net')
    ######################################### worker
    parser.add_argument('--workers', '-j', type=int, default=4,
                        metavar='N', help='dataloader threads')

    # training hyper params
    parser.add_argument('--ohem', action='store_true', default=False,
                        help='OHEM Loss for cityscapes dataset')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    ######################################### batch_size
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 4)')

    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    ######################################### epoch
    parser.add_argument('--epochs', type=int, default=600, metavar='N',
                        help='number of epochs to train (default: 240)')
    ######################################### lr
    parser.add_argument('--lr', type=float, default=0.015, metavar='LR',
                        help='learning rate (default: 1e-4)')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--warmup-iters', type=int, default=0,
                        help='warmup iters')
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3,
                        help='lr = warmup_factor * lr')
    parser.add_argument('--warmup-method', type=str, default='linear',
                        help='method of warmup')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log

    parser.add_argument('--liteflow_model_path', type=str, default='./checkpoint/network-kitti.pytorch',
                        help='put the path to resuming file if needed')
    # './runs/mobilenetv3/2019_08_08_18_07_52_b16_lr0.01/mobilenetv3_large_yuanqu_best_model.pth'
    parser.add_argument('--resume', type=str, default='./checkpoint/mobilenetv3_large_yuanqu_best_model.pth',
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='./runs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=1,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='./runs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=50,
                        help='print log every log-iter')
    # evaluation only
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')

    parser.add_argument('--test-path', default= '../data/test_picture/',
                        help='pictures for testing')

    parser.add_argument('--out-path',default='../data/result/',
                        help='inference result of saving file')
    args = parser.parse_args()

    #######################################################################################################
    args.scale = 0.4
    args.workers = 4
    args.resume = './checkpoint/mobilenetv3_large_liteflow_yuanqu_best_lr0.0005_Fres131_120120True_Fres60True_fintune.pth'

    args.dataset_path = '/workspace/ShareData/Data/yuanqu/video/all/'
    args.train_file = 'image_train'
    args.val_file = 'image_val'
    args.label_file = 'label_L'
#     args.out_path = '../data/results/'
#     args.test_path = '../data/test_picture/'

    args.distributed_mine = False

    args.batch_size = 1
    args.start_step = 1
    args.epochs = 175
    args.lr = 0.001

    args.FFM_list = ['FFM_bi', 'FFM_conv', 'FFM_res', 'FFM_basic_add', 'FFM_basic_cat', 'FFM_res131',
                     'FFM_res131_120', 'FFM_res_120']  # parallel_fusion
    args.FFM_120 = 'FFM_res131_120'
    args.FFM_60 = 'FFM_res'
    args.FFM_120_flag = True
    args.FFM_60_flag = True

    return args

###########################################################################################################
use_val = True
use_val_times = 1
use_save_crop = True  # work only use_val == True

Distributed_Mine = False
CUDA_NUM = 0

############################################################################################################
if not Distributed_Mine:
    cuda_num = CUDA_NUM
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_num)
    torch.cuda.set_device(cuda_num)


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





        test_data_kwargs = {'transform': input_transform,
                             'base_size': args.base_size, 'crop_size': args.crop_size, 're_size': args.re_size,
                             'data_angle': args.data_angle}
        testset = get_segmentation_dataset(args.dataset, args=args, split='test', mode='test', **test_data_kwargs)
        test_sampler = make_data_sampler(testset, False, args.distributed)
        test_batch_sampler = make_batch_data_sampler(test_sampler,1) #args.batch_size)####评价是以batch来评价的
        self.test_loader = data.DataLoader(dataset=testset,
                                            batch_sampler=test_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)

        args.iters_per_epoch = len(testset) // (args.num_gpus * args.batch_size)
        args.max_iters = args.epochs * args.iters_per_epoch

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(args.model,
                                            dataset=args.dataset,
                                            aux=args.aux,
                                            args=self.args,
                                            norm_layer=BatchNorm2d).to(self.device)

        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                pretrained_dict_mobilenetv3 = convert_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))
                model_dict = self.model.state_dict()
                pretrained_dict_mobilenetv3 = {k: v for k, v in pretrained_dict_mobilenetv3.items() if
                                               (k in model_dict and v.size() == model_dict[k].size())}
                self.pretrained_dict_mobilenetv3 = pretrained_dict_mobilenetv3
                model_dict.update(pretrained_dict_mobilenetv3)
                self.model.load_state_dict(model_dict)
            else:
                print('mobilenetv3 ---->>>  checkpoints {} does not exist!'.format(args.resume))

        else:
            print('resume ---->>>  checkpoints {} does not apply!'.format(args.resume))

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[args.local_rank],
                                                             output_device=args.local_rank,
                                                             find_unused_parameters=True)
        if args.distributed_mine:
            self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

    def train(self):
        self.model.eval()
        print('test...')
        self.validation()
        self.model.train()
        print('test over!')

    def test(self):
        if self.args.distributed or self.args.distributed_mine:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()

        ############################# all ####################################################
        for i, (image, target, _) in enumerate(self.test_loader):
            for index in range(len(image)):
                image[index] = image[index].to(self.device)
            # for index in range(len(target)):
            #     target[index] = target[index].to(self.device)
            with torch.no_grad():
                outputs = model(image)
            if use_save_crop:
                save_pred(image, target, _, outputs,self.args)

        synchronize()
        #####################################################################################


def save_pred(image, target, image_name, outputs,args):
    ##################################### 120 ############################
    from light.utils.visualize import get_color_pallete
    import numpy as np
    import scipy.misc as misc
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

    ##################################### 120  ############################
    pred_120 = torch.argmax(outputs[0][0], 1)
    pred_120 = pred_120.cpu().data.numpy()
    predict_120 = pred_120[0]

    mask_120 = get_color_pallete(predict_120, 'yuanqu')
    mask_120 = np.asarray(mask_120.convert('RGB'))
    imageio.imwrite(os.path.join(args.outdir, str(image_name[0])[2:-2] + '.png'), mask_120)
    ##################################### 60 ############################
    pred_60 = torch.argmax(outputs[0][1], 1)
    pred_60 = pred_60.cpu().data.numpy()
    predict_60 = pred_60.squeeze(0)

    mask_60 = get_color_pallete(predict_60, 'yuanqu')
    mask_60 = np.asarray(mask_60.convert('RGB'))
    imageio.imwrite(os.path.join(args.outdir, str(image_name[1])[2:-2] + '.png'), mask_60)



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
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.lr = args.lr * num_gpus

    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")

    if args.FFM_120 not in args.FFM_list or args.FFM_60 not in args.FFM_list:
        raise RuntimeError("Fusion module name is ERROR!! " + "\n")

    args.outdir = args.out_path + '{}_{}'.format(args.model, args.dataset)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    trainer = Trainer(args)
    trainer.test()
    torch.cuda.empty_cache()
