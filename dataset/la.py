import math
import os
import random

import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom
from PIL import Image
from torchvision import transforms
from copy import deepcopy

from dataset.transform import random_rotate, blur, obtain_cutmix_box, random_rot_flip_3d, obtain_cutmix_box_3d, blur_3d


class LAHeartDataset(Dataset):
    """ LA Dataset """

    def __init__(self, name, root=None, mode=None, size=None, id_path=None, nsample=None, num=None, transform=None):
        self.name = name  # dataset
        self.root = root  # data_root
        self.mode = mode  # train_l train_u test
        self.size = size  # crop_size

        self.transform = transform
        self.sample_list = []

        if mode == 'train_l' or mode == 'train_u':
            #print(id_path)
            with open(id_path, 'r') as f:
                self.image_list = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.image_list *= math.ceil(nsample / len(self.image_list))
                self.ids = self.image_list[:nsample]
        elif mode == 'test': # TODO test
            with open('splits/%s/test.txt' % name, 'r') as f:
                self.image_list = f.read().splitlines()
        else:
            with open('splits/%s/valtest.txt' % name, 'r') as f:
                self.image_list = f.read().splitlines()

        self.image_list = [item.strip() for item in self.image_list]

        if num is not None:
            self.image_list = self.image_list[:num]

        #print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        # print(image_name)
        h5f = h5py.File(self.root + "/" + image_name + "/mri_norm2.h5", 'r')
        img = h5f['image'][:]
        mask = h5f['label'][:]
        # sample = {'image': img, 'label': mask}
        # if self.transform:
        #     sample = self.transform(sample)
        # return sample

        if self.mode == 'val':
            return torch.from_numpy(img).float(), torch.from_numpy(mask).long()

        if self.mode == 'test':
            return torch.from_numpy(img).float(), torch.from_numpy(mask).long(), os.path.join(self.root, image_name)

        if random.random() > 0.5:
            img, mask = random_rot_flip_3d(img, mask)
        elif random.random() > 0.5:
            img, mask = random_rotate(img, mask)

        x, y, z = img.shape

        # print(x,y,z)

        # 缩放
        img = zoom(img, (self.size / x, self.size / y, self.size / z), order=0)
        mask = zoom(mask, (self.size / x, self.size / y, self.size / z), order=0)

        if self.mode == 'train_l':
            return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(np.array(mask)).long()

        img_numpy = (img * 255).astype(np.uint8)

        # if img_numpy.shape[0] == 1: # 灰度图像 TypeError: Cannot handle this data type: (1, 1, 256), |u1
        #     img_numpy = img_numpy[0]

        print(img_numpy.shape)
        # img = Image.fromarray(img_numpy)
        img_s1, img_s2 = deepcopy(img), deepcopy(img)
        img = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0


        # if random.random() < 0.8:
        #     img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = blur_3d(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box_3d(self.size, p=0.5)  # cutmix_box中有0和1两个值，为1的区域是一个矩形区域，可以根据这个区域来mix两张图像
        img_s1 = torch.from_numpy(np.array(img_s1)).unsqueeze(0).float() / 255.0

        # if random.random() < 0.8:
        #     img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = blur_3d(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box_3d(self.size, p=0.5)
        img_s2 = torch.from_numpy(np.array(img_s2)).unsqueeze(0).float() / 255.0

        return img, img_s1, img_s2, cutmix_box1, cutmix_box2  # img弱扰动，img_s强扰动



if __name__ == '__main__':
    train_set = LAHeartDataset('la','G:/Ai/dataset/LA/2018LA_Seg_Training Set','train_u',64,'G:/Ai/code/deep_learning/image_segmentation/UniMatch-main/more-scenarios/medical/splits/la/5%/unlabeled.txt')
    # print(len(train_set))
    # print(len(train_set.image_list))
    # data = train_set[0] # 0~79 total 80 samples
    # image, label = data['image'], data['label']
    img, img_s1, img_s2, cutmix_box1, cutmix_box2 = train_set[0]
    print(img.shape, img_s1.shape, img_s2.shape, cutmix_box1.shape, cutmix_box2.shape)