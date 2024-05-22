import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets.custom_transforms as custom_transforms
from config import get_opts, get_training_size
from datasets.test_folder import TestSet
from losses.loss_functions import compute_errors
from SC_Depth import SC_Depth
from visualization import *
from path import Path

from imageio.v2 import imread

import glob
import os

import matplotlib.pyplot as plt

from datasets.specular_detection import *

def extract_number(filename):
    return int(filename.split('/')[-1].split('.')[0])

test_dirs = [
    # "/dataset/c3vd_train/c3vd/test/cecum_t2_a",
    # "/dataset/c3vd_train/c3vd/test/sigmoid_t3_a",
    "/dataset/c3vd_train/c3vd/test/trans_t2_a",
]

specular_detection = SpecularDetection(T1=200, T2_abs=230, T2_rel=2.0, T3=4, N_min=1500, N_min2=1500)

def main():

    for test_dir in test_dirs:
        img_paths = glob.glob("{}/*.png".format(test_dir))
        img_paths = sorted(img_paths, key=extract_number)
        print(len(img_paths))

        mask_list = []
        for i, img_path in enumerate(tqdm(img_paths)):
            tgt_img = imread(img_path)
            tgt_img =  cv2.resize(tgt_img, (320, 256))
            tgt_img = tgt_img.astype(np.float32) / 255.0
            mask1 = specular_detection(tgt_img)["point"].numpy().astype(np.uint8)
            # mask2 = specular_detection(tgt_img)["block"].numpy().astype(np.uint8)
            # mask = np.logical_or(mask1, mask2).astype(np.uint8) * 255
            # print(mask.max())
            mask_list.append(mask1*255)

        pred_depth = np.concatenate(mask_list)
        print(pred_depth.shape)
        np.savez_compressed("./test/c3vd_point_mask/{}.npz".format(os.path.basename(test_dir)), data=pred_depth)

if __name__ == '__main__':
    main()
