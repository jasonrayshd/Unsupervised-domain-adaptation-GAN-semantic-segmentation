import os
import cv2
import numpy as np


dataset_potsdam = "../data/potsdam/train/cropped/images"
dataset_vaihingen = "../data/vaihingen/train/cropped/images"

ImgFiles_potsdam = os.listdir(dataset_potsdam)
ImgFiles_vaihingen = os.listdir(dataset_vaihingen)

print(f"number of image files:{len(ImgFiles_potsdam)}")
print(f"number of image files:{len(ImgFiles_vaihingen)}")


def get_mean_std(size, files, path):
    """
    compute mean and std of given dataset
    :param size: size of dataset, int
    :param files: files name of each images in dataset, list
    :param path: dataset path, str
    :return: mean and std, float
    """
    means = np.zeros((3))   # [0,0,0]
    stdevs = np.zeros((3))  # [0,0,0]

    for file in files:
        img = cv2.imread(path + os.sep + file) #(H, W, C)

        for i in range(3):
            # 一个通道的均值和标准差
            means[i] += img[:, :, i].mean()
            stdevs[i] += img[:, :, i].std()


    means = means / (size * 255)
    stdevs = stdevs / (size * 255)

    return means, stdevs
    
means_potsdam, stds_potsdam = get_mean_std(len(ImgFiles_potsdam), ImgFiles_potsdam, dataset_potsdam) 
means_vaihingen, stds_vaihingen = get_mean_std(len(ImgFiles_vaihingen), ImgFiles_vaihingen, dataset_vaihingen)
# 0.33472726 0.36082851 0.33697116, 0.12418044 0.11993021 0.12179105
# 0.31385447 0.31813186 0.46560643, 0.13538207 0.14157808 0.20539479

print(means_potsdam, stds_potsdam)
print(means_vaihingen, stds_vaihingen)




