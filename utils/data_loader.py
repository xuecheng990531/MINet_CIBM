import os
import cv2
import torch
import numpy as np
np.set_printoptions(threshold=np.inf)
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import albumentations as A
torch.set_printoptions(profile="full")


class fundus_data(Dataset):
    def __init__(self, data_root: str, mode: str):
        super(fundus_data, self).__init__()
        
        assert os.path.exists(data_root), f"path '{data_root}' does not exist."
        self.imgs_dir = os.path.join(data_root, "img/")  
        self.masks_dir = os.path.join(data_root, "mask/") 

        self.image_names = [file for file in os.listdir(self.imgs_dir)]

        print(f'{mode}:dataset with {len(self.image_names)} examples.')
        self.mode = mode

    def __len__(self):
        return len(self.image_names)

    def preprocess(self, image, mask):
        if self.mode == "train":
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Resize(height=512,width=512,p=1),
                A.VerticalFlip(p=0.5),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),  
                ], p=0.2),
                A.RandomBrightnessContrast(p=0.2)
                ])
            transformed = transform(image=image, mask=mask)
            image=transformed['image']
            mask=transformed['mask']
        else:
            transform = A.Compose([
                A.Resize(height=512,width=512,p=1),
                ])
            transformed = transform(image=image, mask=mask)
            image=transformed['image']
            mask=transformed['mask']
            
        return image,mask

    def __getitem__(self, index):
        # 获取image和mask的路径
        image_name = self.image_names[index]
        # vessel data
        # if 'im' in image_name and '0' in image_name and '_' not in image_name:
        #     image_path = os.path.join(self.imgs_dir, image_name)
        #     mask_path = os.path.join(self.masks_dir, image_name.split('.')[0]+'.ah'+'.ppm')
        # elif 'img_' in image_name:
        #     image_path = os.path.join(self.imgs_dir, image_name)
        #     mask_path = os.path.join(self.masks_dir, image_name.split('.')[0]+'.tif')
        # else:

        image_path = os.path.join(self.imgs_dir, image_name)
        mask_path = os.path.join(self.masks_dir, image_name)

        # # STARE
        # image_path=os.path.join(self.imgs_dir,image_name)
        # mask_path=os.path.join(self.masks_dir,image_name.split('.')[0]+'.vk.ppm')

        # # CHASEDB1
        # image_path=os.path.join(self.imgs_dir,image_name)
        # mask_path=os.path.join(self.masks_dir,image_name.split('.')[0]+'_1stHO.png')

        # HRF
        # image_path=os.path.join(self.imgs_dir,image_name)
        # mask_path=os.path.join(self.masks_dir,image_name.split('.')[0]+'.tif')



        assert os.path.exists(image_path), f"file '{image_path}' does not exist."
        assert os.path.exists(mask_path), f"file '{mask_path}' does not exist."

        image=cv2.imread(image_path,1)
        image=image[..., 1]
        mask=cv2.imread(mask_path,0)

        image,mask = self.preprocess(image, mask)
        image=transforms.ToTensor()(image)
        mask=transforms.ToTensor()(mask)
        mask[mask>0.7]=1
        mask[mask<=0.7]=0

        return image,mask


if __name__ == '__main__':
    import torch
    data=fundus_data('sdnu_eye/train',mode='train')
    print(data[0][0].shape)
    print(data[0][1].shape)
    print(len(data))

    data=fundus_data('sdnu_eye/test',mode='test')
    print(data[0][0].shape)
    print(data[0][1].shape)
    print(len(data))