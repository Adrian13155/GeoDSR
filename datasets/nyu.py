import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import glob
import random
from PIL import Image
import tqdm

from utils import make_coord

def to_pixel_samples(depth):
    """ Convert the image to coord-RGB pairs.
        depth: Tensor, (1, H, W)
        只修改了这一行:
        coord = make_coord(depth.shape[-2:], flatten=True) # [H*W, 2]
        如果要用NYUDataset就需要像上面这样修改
    """
    coord = make_coord(depth.shape[-2:], flatten=True).view(depth.shape[-2], depth.shape[-1], 2) # [H，W, 2]
    pixel = depth.view(-1, 1) # [H*W, 1]
    return coord, pixel

class NYUDataset(Dataset):
    def __init__(self, root='/data3/tang/nyu_labeled', split='train', scale=8, augment=True, downsample='bicubic', pre_upsample=False, to_pixel=False, sample_q=None, input_size=None, noisy=False):
        super().__init__()
        self.root = root
        self.split = split
        self.scale = scale
        self.augment = augment
        self.downsample = downsample
        self.pre_upsample = pre_upsample
        self.to_pixel = to_pixel
        self.sample_q = sample_q
        self.input_size = input_size
        self.noisy = noisy

        # use the first 1000 data as training split
        if self.split == 'train':
            self.size = 1000
        else:
            self.size = 449

    def __getitem__(self, idx):
        if self.split != 'train':
            idx += 1000

        image_file = os.path.join(self.root, 'RGB', f'{idx}.jpg')
        depth_file = os.path.join(self.root, 'Depth', f'{idx}.npy')             

        image = cv2.imread(image_file) # [H, W, 3]
        depth_hr = np.load(depth_file) # [H, W]
        
        # crop after rescale
        if self.input_size is not None:
            x0 = random.randint(0, image.shape[0] - self.input_size)
            y0 = random.randint(0, image.shape[1] - self.input_size)
            image = image[x0:x0+self.input_size, y0:y0+self.input_size]
            depth_hr = depth_hr[x0:x0+self.input_size, y0:y0+self.input_size]


        h, w = image.shape[:2]

        if self.downsample == 'bicubic':
            depth_lr = np.array(Image.fromarray(depth_hr).resize((w//self.scale, h//self.scale), Image.BICUBIC)) # bicubic, RMSE=7.13
            image_lr = np.array(Image.fromarray(image).resize((w//self.scale, h//self.scale), Image.BICUBIC)) # bicubic, RMSE=7.13
            #depth_lr = cv2.resize(depth_hr, (w//self.scale, h//self.scale), interpolation=cv2.INTER_CUBIC) # RMSE=8.03, cv2.resize is different from Image.resize.
        elif self.downsample == 'nearest-right-bottom':
            depth_lr = depth_hr[(self.scale - 1)::self.scale, (self.scale - 1)::self.scale] # right-bottom, RMSE=14.22, finally reproduced it...
            image_lr = image[(self.scale - 1)::self.scale, (self.scale - 1)::self.scale] # right-bottom, RMSE=14.22, finally reproduced it...
        elif self.downsample == 'nearest-center':
            depth_lr = np.array(Image.fromarray(depth_hr).resize((w//self.scale, h//self.scale), Image.NEAREST)) # center (if even, prefer right-bottom), RMSE=8.21
            image_lr = np.array(Image.fromarray(image).resize((w//self.scale, h//self.scale), Image.NEAREST)) # center (if even, prefer right-bottom), RMSE=8.21
        elif self.downsample == 'nearest-left-top':
            depth_lr = depth_hr[::self.scale, ::self.scale] # left-top, RMSE=13.94
            image_lr = image[::self.scale, ::self.scale] # left-top, RMSE=13.94
        else:
            raise NotImplementedError

        if self.noisy:
            depth_lr = add_noise(depth_lr, sigma=0.04, inv=False)

        # normalize
        depth_min = depth_hr.min()
        depth_max = depth_hr.max()
        depth_hr = (depth_hr - depth_min) / (depth_max - depth_min)
        depth_lr = (depth_lr - depth_min) / (depth_max - depth_min)
        
        image = image.astype(np.float32).transpose(2,0,1) / 255
        image_lr = image_lr.astype(np.float32).transpose(2,0,1) / 255 # [3, H, W]

        image = (image - np.array([0.485, 0.456, 0.406]).reshape(3,1,1)) / np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
        image_lr = (image_lr - np.array([0.485, 0.456, 0.406]).reshape(3,1,1)) / np.array([0.229, 0.224, 0.225]).reshape(3,1,1)

        # follow DKN, use bicubic upsampling of PIL
        depth_lr_up = np.array(Image.fromarray(depth_lr).resize((w, h), Image.BICUBIC))

        if self.pre_upsample:
            depth_lr = depth_lr_up

        # to tensor
        image = torch.from_numpy(image).float()
        image_lr = torch.from_numpy(image_lr).float()
        depth_hr = torch.from_numpy(depth_hr).unsqueeze(0).float()
        depth_lr = torch.from_numpy(depth_lr).unsqueeze(0).float()
        depth_lr_up = torch.from_numpy(depth_lr_up).unsqueeze(0).float()

        # transform
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                return x

            image = augment(image)
            image_lr = augment(image_lr)
            depth_hr = augment(depth_hr)
            depth_lr = augment(depth_lr)
            depth_lr_up = augment(depth_lr_up)

        image = image.contiguous()
        image_lr = image_lr.contiguous()
        depth_hr = depth_hr.contiguous()
        depth_lr = depth_lr.contiguous()
        depth_lr_up = depth_lr_up.contiguous()

        # to pixel
        if self.to_pixel:
            
            hr_coord, hr_pixel = to_pixel_samples(depth_hr)

            lr_pixel = depth_lr_up.view(-1, 1)
 
            if self.sample_q is not None:
                sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
                hr_coord = hr_coord[sample_lst]
                hr_pixel = hr_pixel[sample_lst]
                lr_pixel = lr_pixel[sample_lst]

            cell = torch.ones_like(hr_coord)
            cell[:, 0] *= 2 / depth_hr.shape[-2]
            cell[:, 1] *= 2 / depth_hr.shape[-1]
        
            return {
                'image': image,
                'lr_image': image_lr,
                'lr': depth_lr,
                'hr': hr_pixel,
                'hr_depth': depth_hr,
                'lr_pixel': lr_pixel,
                'hr_coord': hr_coord,
                'min': depth_min * 100,
                'max': depth_max * 100,
                'cell': cell,
                'idx': idx,
            }   

        # 我们应该是要不to pixel的输出
        else:
            return {
                'image': image,
                'lr': depth_lr,
                'hr': depth_hr,
                'min': depth_min * 100,
                'max': depth_max * 100,
                'idx': idx,
            }

    def __len__(self):
        return self.size

def get_patch(img, gt, patch_size=16):
    th, tw = img.shape[:2]

    tp = round(patch_size)

    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))

    return img[ty:ty + tp, tx:tx + tp, :], gt[ty:ty + tp, tx:tx + tp, :]

def arugment(img,gt, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5

    if hflip: 
        img = img[:, ::-1, :].copy()
        gt = gt[:, ::-1, :].copy()
    if vflip: 
        img = img[::-1, :, :].copy()
        gt = gt[::-1, :, :].copy()

    return img, gt

class NYU_v2_datsetForGeoDSR(Dataset):
    """NYUDataset For Code Of GeoDSR.
        由于GeoDSR模型输入有各种变量,所以这里参照了https://github.com/nana01219/GeoDSR/blob/main/datasets/geo_nyu.py中的代码进行了修改
    """

    def __init__(self, root_dir, down="bicubic", scale=8, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            train (bool): train or test
            transform (callable, optional): Optional transform to be applied on a sample.
            
        """
        self.down = down
        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        self.train = train
        
        if train:
            self.depths = np.load('%s/train_depth_split.npy'%root_dir)
            # print(f"depths.shape:{self.depths.shape}")
            self.images = np.load('%s/train_images_split.npy'%root_dir)
            # print(f"image.shape:{self.images.shape}")
        else:
            self.depths = np.load('%s/test_depth.npy'%root_dir)
            # print(f"depths.shape:{self.depths.shape}")
            self.images = np.load('%s/test_images_v2.npy'%root_dir)
            # print(f"image.shape:{self.images.shape}")
    def __len__(self):
        return self.depths.shape[0]

    def __getitem__(self, idx):
        depth = self.depths[idx]
        image = self.images[idx]
        if self.train:
            image, depth = get_patch(img=image, gt=np.expand_dims(depth,2), patch_size=256)
            image, depth = arugment(img=image, gt=depth)
        h, w = depth.shape[:2]
        s = self.scale

        if self.down == "bicubic":
            # bicubic down-sampling
            lr = np.array(Image.fromarray(depth.squeeze()).resize((w//s,h//s), Image.BICUBIC))
            lr_up = np.array(Image.fromarray(lr.squeeze()).resize((w, h), Image.BICUBIC)) # AHMF,dkn,fdkn
        if self.down == "direct":
            lr = np.array(Image.fromarray(depth.squeeze()).resize((w//s,h//s)))

        bms = np.array(Image.fromarray(lr.squeeze()).resize((w, h), Image.BICUBIC))

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(depth).float()
            lr = self.transform(np.expand_dims(lr, 2)).float()
            lr_up = self.transform(np.expand_dims(lr_up, 2)).float() # AHMF
            bms = self.transform(np.expand_dims(bms, 2)).float()

        depth = depth.contiguous()
        hr_coord, hr_pixel = to_pixel_samples(depth) 
        lr_distance_h = 2/lr.shape[-2]
        lr_distance_w = 2/lr.shape[-1]
        lr_distance = torch.tensor([lr_distance_h, lr_distance_w])
        field = torch.ones([8])
        cH, cW, _ = hr_coord.shape
        ch = cH // 2
        cw = cW // 2

        f1 = abs(hr_coord[ch+1, cw-1] - hr_coord[ch, cw])
        field[0:2] =f1/lr_distance
        f2 = abs(hr_coord[ch-1, cw-1] - hr_coord[ch, cw])
        field[2:4] =f2/lr_distance
        f3 = abs(hr_coord[ch+1, cw+1] - hr_coord[ch, cw])
        field[4:6] = f3/lr_distance
        f4 = abs(hr_coord[ch-1, cw+1] - hr_coord[ch, cw])
        field[6:] = f4/lr_distance
        # print(image.shape)
        # print(lr.shape)
        # print(depth.shape)
        data = {'hr_image': image, 'lr_depth': lr, 'hr_coord': hr_coord, 'lr_depth_up': lr_up, 'field': field,"hr_depth":depth}
        
        return data


if __name__ == '__main__':
    print('===== test direct bicubic upsampling =====')
    for method in ['bicubic']:
        for scale in [4, 8, 16]:
            print(f'[INFO] scale = {scale}, method = {method}')
            d = NYUDataset(root='/data3/tang/nyu_labeled', split='test', pre_upsample=True, augment=False, scale=scale, downsample=method, noisy=False)
            #d = NYUDataset(root='/data3/tang/nyu_labeled', split='test', pre_upsample=True, augment=False, scale=scale, downsample=method, noisy=True)
            rmses = []
            for i in tqdm.trange(len(d)):
                x = d[i]
                lr = ((x['lr'].numpy() * (x['max'] - x['min'])) + x['min'])
                hr = ((x['hr'].numpy() * (x['max'] - x['min'])) + x['min'])
                rmse = np.sqrt(np.mean(np.power(lr - hr, 2)))
                rmses.append(rmse)
            print('RMSE = ', np.mean(rmses))