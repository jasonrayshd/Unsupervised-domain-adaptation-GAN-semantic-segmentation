import os
from shutil import copyfile



def copyimages():
    root = "/home/admin/segmentation/task2/reference/pytorch-CycleGAN-and-pix2pix/results/potsdam2vaihingen/test_latest/images"
    dest = "/home/admin/segmentation/task2/data/gen"
    img_files = os.listdir(root)
    for f in img_files:
        if "rec" in f:
            copyfile(f"{root}/{f}", f"{dest}/{f}")

