import os
import sys
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
from light.utils.load_model import *
from light.data import get_segmentation_dataset
from light.model import get_segmentation_model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def parse_args():
    parser = argparse.ArgumentParser(description='Light Model for Segmentation')
    # model and dataset
    parser.add_argument('--model', type=str, default='mobilenetv3_large_pwcflow',
                        help='backbone model name (default: mobilenet)')
    parser.add_argument('--dataset', type=str, default='yuanqu', help='dataset name (default: citys)')
    parser.add_argument('--aux', action='store_true', default=False, help='Auxiliary loss')

    parser.add_argument('--nclass', type=int, default=6, metavar='N', help='class (default: 6)')
    parser.add_argument('--base-size', type=int, default=1208, help='base image size')
    parser.add_argument('--crop-size', type=int, default=1216, help='crop image size')
    parser.add_argument('--re_size', type=int, default=(1920, 1216), help='resize for L-all net')
    parser.add_argument('--scale', type=int, default=0.4, help='resize for L-all net')

    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 4)')

    parser.add_argument('--workers', '-j', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)

    # path
    parser.add_argument('--model-mode', type=str, default='mobilenetv3',
                        choices=['mobilenetv3', 'mobilenetv3_loosely', 'mobilenetv3_tightly'],
                        help='model name (default: mobilenetv3)')
    parser.add_argument('--resume', type=str, default='./checkpoint/mobilenetv3_large_yuanqu.pth',
                        help='put the path to resuming file if needed')
    parser.add_argument('--test-path', default='../data/test_picture/',
                        help='pictures for testing')
    parser.add_argument('--out-path', default='../data/results/',
                        help='inference result of saving file')
    args = parser.parse_args()

    return args


class Tester(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        test_data_kwargs = {'transform': input_transform,
                            'base_size': args.base_size, 'crop_size': args.crop_size, 're_size': args.re_size,
                            }

        testset = get_segmentation_dataset(args.dataset, args=args, split='test', mode='test', **test_data_kwargs)
        test_sampler = make_data_sampler(testset, False, args.distributed)
        test_batch_sampler = make_batch_data_sampler(test_sampler, args.batch_size)
        self.test_loader = data.DataLoader(dataset=testset,
                                           batch_sampler=test_batch_sampler,
                                           num_workers=args.workers,
                                           pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(args.model,
                                            dataset=args.dataset,
                                            args=self.args,
                                            norm_layer=BatchNorm2d).to(self.device)
        self.model = load_model(args.resume, self.model)

    def test(self):
        print('testing...')
        torch.cuda.empty_cache()  # TODO check if it helps
        self.model.eval()
        ############################# all ####################################################
        for i, (image, target, _) in enumerate(self.test_loader):
            for index in range(len(image)):
                image[index] = image[index].to(self.device)
            with torch.no_grad():
                outputs = self.model(image)
            save_pred(image, target, _, outputs, self.args)
        print('test is over!')
        synchronize()
        #####################################################################################


def save_pred(image, target, image_name, outputs, args):
    from light.utils.visualize import get_color_pallete
    import numpy as np
    import scipy.misc as misc

    ##################################### 120 ############################
    pred_120 = torch.argmax(outputs[0][0], 1)
    pred_120 = pred_120.cpu().data.numpy()
    predict_120 = pred_120[0]

    mask_120 = get_color_pallete(predict_120, 'yuanqu')
    mask_120 = np.asarray(mask_120.convert('RGB'))
    imageio.imwrite(os.path.join(args.outdir, str(image_name[0])[2:-2] + '_' + args.model_mode + '.png'), mask_120)
    ##################################### 60 #############################
    pred_60 = torch.argmax(outputs[0][1], 1)
    pred_60 = pred_60.cpu().data.numpy()
    predict_60 = pred_60.squeeze(0)

    mask_60 = get_color_pallete(predict_60, 'yuanqu')
    mask_60 = np.asarray(mask_60.convert('RGB'))
    imageio.imwrite(os.path.join(args.outdir, str(image_name[1])[2:-2] + '_' + args.model_mode + '.png'), mask_60)


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

    args.model_mode_list = ['mobilenetv3', 'mobilenetv3_loosely', 'mobilenetv3_tightly']
    args.FFM_list = ['FFM_res', 'FFM_res131', 'FFM_res131_120', 'FFM_res_120']  # parallel_fusion
    args.FFM_120 = 'FFM_res131_120'
    args.FFM_60 = 'FFM_res'

    if args.FFM_120 not in args.FFM_list or args.FFM_60 not in args.FFM_list:
        raise RuntimeError("Fusion module name is ERROR!! " + "\n")
    if args.model_mode not in args.model_mode_list:
        raise RuntimeError("Model mode name is ERROR!! " + "\n")
    args.outdir = args.out_path + '{}_{}'.format(args.model, args.dataset)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    tester = Tester(args)
    tester.test()
    torch.cuda.empty_cache()
