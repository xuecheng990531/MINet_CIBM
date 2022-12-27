import os
import numpy as np
from os import path as osp
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from PIL import Image
import cv2 as cv
import albumentations as A

tf=A.Compose([
    A.Resize(height=256,width=256,p=1)
])

tf_test=A.Compose([
    A.Resize(height=256,width=256,p=1)
])

def get_all_files(dirname):
    # Get all files from a directory name (assuming no other directories inside)
    for r, dirs, files in os.walk(dirname):
        if dirs != []:
            assert False, 'Dircetory found inside path {}'.format(r)
        files = map(lambda x: osp.join(r, x), files)
        files = list(files)
    files = sorted(files)
    return files


class HRFDataset(Dataset):

    def __init__(self, data_dir, train=True, augment=True):
        self.data_dir = data_dir
        self.train = train
        self.augment = augment

        self.images = get_all_files(osp.join(data_dir, 'img'))
        self.segs = get_all_files(osp.join(data_dir, 'label'))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        seg = self.segs[idx]
        # seg = Image.open(seg).convert
        seg = cv.imread(seg,0)
        # seg = seg.resize((512, 512))

        # seg = np.array(seg)/255.0

        # if len(seg.shape) == 3:
        #     seg = seg[:, :, 0]

        # Get image from data or gt segmentation
        img = self.images[idx]
        img = cv.imread(img)
        # img = img.resize((512, 512))
        # img = np.array(img)/255.0

        if len(img.shape) == 3:
            img = img[..., 1]  # extract green channel only
        
        if self.train:
            sample = tf(image=img,mask=seg)
            img, seg = sample['image'], sample['mask']
        else:
            sample = tf_test(image=img,mask=seg)
            img, seg = sample['image'], sample['mask']

        return ToTensor()(img),ToTensor()(seg)


if __name__ == "__main__":
    import torch
    torch.set_printoptions(profile='full')
    ds = HRFDataset("CHASEDB1/train",train=False)
    print(ds[0][0].shape)
    print(ds[0][1].shape)
