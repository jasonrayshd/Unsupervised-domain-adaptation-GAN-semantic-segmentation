import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

potsdam_dataset = "/home/admin/segmentation/task2/data/potsdam/train/cropped/labels"
potsdam_dest = "/home/admin/segmentation/task2/data/potsdam/train/cropped/masks"
vaihingen_dataset = "/home/admin/segmentation/task2/data/vaihingen/train/cropped/labels"
vaihingen_dest = "/home/admin/segmentation/task2/data/vaihingen/train/cropped/masks"

potsdam_label_files = os.listdir(potsdam_dataset)
vaihingen_label_files = os.listdir(vaihingen_dataset)

RGBtoCategory = {(255,255,255):0,(0,0,255):1,(0,255,255):2,(0,255,0):3,(255,255,0):4,(255,0,0):5}

def get_mask(files, srcpath, destpath):
    for file in tqdm(files):
        img = cv2.imread(srcpath+ os. sep + file)  # BGR
        mask_img = np.empty((512,512))

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                R = 255 if img[i,j,0] >200 else 0
                G = 255 if img[i,j,1] >200 else 0
                B = 255 if img[i,j,2] >200 else 0
                mask_img[i,j] = RGBtoCategory[(R,G,B)]
        
        cv2.imwrite(f"{destpath}/{file}", mask_img)

print("processing potsdam..")
get_mask(potsdam_label_files, potsdam_dataset, potsdam_dest)
print("processing vaihingen..")
get_mask(vaihingen_label_files, vaihingen_dataset, vaihingen_dest)