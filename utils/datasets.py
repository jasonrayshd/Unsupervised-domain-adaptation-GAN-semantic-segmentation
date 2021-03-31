from torch.utils.data import Dataset
from torchvision import transforms

import albumentations as alb
from albumentations.pytorch import ToTensorV2

import cv2
import os
import numpy as np
import sys

import matplotlib.pyplot as plt

class AerialDataset(Dataset):

    root = "./data"

    def __init__(self, mode , dataset, img_path: str, label_path: str, transform = None):
        super(AerialDataset,self).__init__()
        self.img_path = img_path
        self.image_files = sorted(os.listdir(img_path))
        self.mode = mode

        self.label_path = label_path
        self.label_files = sorted(os.listdir(label_path))
        assert len(self.image_files) == len(self.label_files)
        
        self.num = len(self.image_files)

        # TODO: recompute normalization and std

        if dataset == "potsdam":
            mean = (0.33472726, 0.36082851, 0.33697116)
            std = (0.12418044, 0.11993021, 0.12179105)
        elif dataset == "vaihinge":
            mean = (0.31385447, 0.31813186, 0.46560643)
            std = (0.13538207, 0.14157808, 0.20539479)

        if transform is None:
            self.transform = alb.Compose([
                # alb.HorizontalFlip(p = 0.9),
                # alb.RandomResizedCrop(512,512, scale=(0.75,2)),
                # alb.Normalize(mean = mean, std = std),
                # alb.Resize(512,512),
                ToTensorV2(),
            ])
        else:
            self.transform = transform
    
    def __getitem__(self, idx: int):
        image = cv2.imread(self.img_path + os.sep + self.image_files[idx])
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.label_path + os.sep + self.label_files[idx], cv2.IMREAD_UNCHANGED)  # (512, 512)

        transformed= self.transform(image=image, mask=label)

        image = transformed["image"]/255       # (3, 512, 512)
        mask = transformed["mask"].long()  # convert to Tensor.LongTensor

        return image, mask

    
    def __len__(self):
        return self.num


if __name__ == "__main__":
    # for debugging:
    import matplotlib.pyplot as plt

    dataset = AerialDataset("potsdam", "/home/admin/segmentation/task2/data/potsdam/train/cropped/images/train", 
    "/home/admin/segmentation/task2/data/potsdam/train/cropped/masks/train")

    plt.imshow(dataset[0][0].transpose(0,2).transpose(0,1))
    plt.savefig("output1.png")

    plt.imshow(dataset[0][1])
    plt.savefig("output2.png")
    
    

