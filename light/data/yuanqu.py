"""Cityscapes Dataloader"""
import os
import random
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image, ImageOps, ImageFilter

__all__ = ['yuanquSegmentation']


class yuanquSegmentation(data.Dataset):
    """Cityscapes Semantic Segmentation Dataset.
    Parameters
    ----------
    root : string
        Path to Cityscapes folder. Default is './datasets/citys'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = yuanquSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'cityscapes'
    NUM_CLASS = 6

    def __init__(self, args, root='/workspace/ShareData/Data/yuanqu/video/all/image/', split='train', mode=None, transform=None, base_size=520, crop_size=480, data_angle=120, re_size=(1920,1216), **kwargs):
        super(yuanquSegmentation, self).__init__()
        #self.num_class = NUM_CLASS
        self.root = args.test_path#args.dataset_path + args.train_file + '/'
        self.split = split
        self.mode = mode if mode is not None else split
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size
        self.re_size = re_size
        self.data_angle = data_angle
        
        #self.images_paths, self.mask_paths = _get_city_pairs(self.root, self.split, self.data_angle)
        self.images_paths, self.mask_paths = _get_city_pairs(folder=self.root,split=split,args=args)
        if split != 'test':
            assert (len(self.images_paths) == len(self.mask_paths))
            if len(self.images_paths) == 0:
                if self.split == 'train':
                    raise RuntimeError("Found 0 images in subfolders of: " + self.root + "\n")
                elif self.split == 'val':
                    raise RuntimeError("Found 0 images in subfolders of: " + args.val_file + "\n")
       
        '''
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                              23, 24, 25, 26, 27, 28, 31, 32, 33]
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        '''
        self.valid_classes = [1,2,3,4,5,6]
        self._key = np.array([-1,0,1,2,3,4,5])
        
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def _class_to_index(self, mask):
        #mask_new = mask - np.ones(shape=mask.shape,dtype=np.int)
        '''
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        a =  mask.ravel()
        index = np.digitize(a, self._mapping, right=True)
        
        return self._key[index].reshape(mask.shape)
        '''
        return mask
    def __getitem__(self, index):
        if self.mode == 'test':
            img = Image.open(self.images_paths[index]).convert('RGB')
            imgpath60 = self.images_paths[index].replace('fov120', 'fov60')
            img_60 = Image.open(imgpath60).convert('RGB')

            img = img.resize(self.re_size, Image.BILINEAR)
            img_60 = img_60.resize(self.re_size, Image.BILINEAR)
            img = self._img_transform(img)
            img_60 = self._img_transform(img_60)
            if self.transform is not None:
                img = self.transform(img)
                img_60 = self.transform(img_60)
            return [img, img_60], [], [self.images_paths[index].split('/')[-1][:-4],  imgpath60.split('/')[-1][:-4]]

        img = Image.open(self.images_paths[index]).convert('RGB')
        mask = Image.open(self.mask_paths[index]).convert('L')  
        
        #imgpath60 = self.images_paths[index].replace('/120/', '/60/').replace('fov120', 'fov60')
        #maskpath60 = self.images_paths[index].replace('/120/', '/60/').replace('fov120', 'fov60')
        
        imgpath60 = self.images_paths[index].replace('fov120', 'fov60')
        maskpath60 = self.mask_paths[index].replace('fov120', 'fov60')
        img_60 = Image.open(imgpath60).convert('RGB')
        mask_60 = Image.open(maskpath60).convert('L')  
        

        
        
        
        # synchrosized transform
        if self.mode == 'train':
            '''
            w, h = img.size
            x1 = random.randint(0, w - self.crop_size)
            y1 = random.randint(0, h - self.crop_size)
            img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            img, mask = self._img_transform(img), self._mask_transform(mask)
            '''
            img, mask = self._sync_transform(img, mask)
            img_60, mask_60 = self._sync_transform(img_60, mask_60)
        elif self.mode == 'train_all' or self.mode == 'val_all':
            img = img.resize(self.re_size, Image.BILINEAR)
            mask = mask.resize(self.re_size, Image.NEAREST)
            img_60 = img_60.resize(self.re_size, Image.BILINEAR)
            mask_60 = mask_60.resize(self.re_size, Image.NEAREST)
            img, mask = self._img_transform(img), self._mask_transform(mask)
            img_60, mask_60 = self._img_transform(img_60), self._mask_transform(mask_60)
        elif self.mode == 'val':
            
            #img, mask = self._img_transform(img), self._mask_transform(mask)
            img, mask = self._val_sync_transform(img, mask)
            img_60, mask_60 = self._val_sync_transform(img_60, mask_60)
            
        else:
            assert self.mode == 'testval' 
            #img = img.resize((w,int(604)), Image.BILINEAR)
           
            img, mask = self._img_transform(img), self._mask_transform(mask)
            img_60, mask_60 = self._img_transform(img_60), self._mask_transform(mask_60)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
            img_60 = self.transform(img_60)
       
        return [img,img_60], [mask,mask_60], [self.images_paths[index].split('/')[-1][:-4],imgpath60.split('/')[-1][:-4]]

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)        
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))
        

    def __len__(self):
        return len(self.images_paths)

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    
    @property
    def pred_offset(self):
        return 0


def _get_city_pairs(folder, args, split='train',data_angle=60):
    '''旧的数据集
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith(".jpg"):
                    imgpath = os.path.join(root, filename)
                    #foldername = os.path.basename(os.path.dirname(imgpath))
                    #maskname = filename.replace('jpg', 'png') # added baoanbo20190805 add#
                    #maskpath = os.path.join(mask_folder, maskname)# added baoanbo20190805 add#
                    maskpath = imgpath.replace('jpg', 'png').replace('leftImg8bit', 'gtFine')
                    
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        imgpath60 = imgpath.replace('/120/', '/60/').replace('fov120', 'fov60')
                        maskpath60 = maskpath.replace('/120/', '/60/').replace('fov120', 'fov60')
                        if os.path.isfile(imgpath60) and os.path.isfile(maskpath60):                            
                            img_paths.append(imgpath)
                            mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths
    '''
    def get_path_pairs(img_folder, mask_folder, args):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if 'checkpoint' in filename:
                    print('OUT mask or image:', filename)
                    continue
                if filename.endswith(".jpg") and 'fov120' in filename:
                    imgpath = os.path.join(root, filename)
                    maskpath = imgpath.replace('jpg', 'png').replace(args.val_file,args.label_file).replace(args.train_file, args.label_file)                    
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):                           
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths
    def get_path_pairs_test(img_folder, mask_folder, args):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if 'checkpoint' in filename:
                    print('OUT mask or image:', filename)
                    continue
                if filename.endswith(".jpg") and 'fov120' in filename:
                    imgpath = os.path.join(root, filename)
                    # maskpath = imgpath.replace('jpg', 'png').replace(args.val_file,args.label_file).replace(args.train_file, args.label_file)
                    if os.path.isfile(imgpath): #and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        # mask_paths.append(maskpath)
                    else:
                        print('cannot find the image:', imgpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, []

    if split in ('train', 'val'):
        #img_folder = os.path.join(folder, 'leftImg8bit/' + split + '/' + str(data_angle) + '/120')
        #mask_folder = os.path.join(folder, 'gtFine/' + split + '/' + str(data_angle) + '/120')
        img_folder = os.path.join(folder)
        mask_folder = os.path.join(folder)
        if split == 'val':            
            img_folder = os.path.join(folder.replace(args.train_file, args.val_file))
            print('evaluting : ',img_folder)
        else:
            print('training : ',img_folder)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder, args)  
        
        return img_paths, mask_paths
    elif split == 'test':
        img_folder = os.path.join(folder)
        mask_folder = '' #os.path.join(folder)
        img_paths, mask_paths = get_path_pairs_test(img_folder, mask_folder, args)

        return img_paths, mask_paths

    else:
        assert split == 'trainval'
        print('trainval set')
        train_img_folder = os.path.join(folder, 'leftImg8bit/train/120')
        train_mask_folder = os.path.join(folder, 'gtFine/train/120')
        val_img_folder120 = os.path.join(folder, 'leftImg8bit/val/120_7')
        val_mask_folder120 = os.path.join(folder, 'gtFine/val/120_7')
        #val_img_folder60 = os.path.join(folder, 'leftImg8bit/val/60')
        #val_mask_folder60 = os.path.join(folder, 'gtFine/val/60')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths120, val_mask_paths120 = get_path_pairs(val_img_folder120, val_mask_folder120)
        #val_img_paths60, val_mask_paths60 = get_path_pairs(val_img_folder60, val_mask_folder60)
        img_paths = train_img_paths + val_img_paths120 #+ val_img_paths60 
        mask_paths = train_mask_paths + val_mask_paths120 #+ val_mask_paths60
    
    return img_paths, mask_paths


if __name__ == '__main__':
    dataset = yuanquSegmentation()
    img, label = dataset[0]
