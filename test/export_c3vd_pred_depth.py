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

def extract_number(filename):
    return int(filename.split('/')[-1].split('.')[0])

test_dirs = [
    "/dataset/c3vd_train/c3vd/test/cecum_t2_a",
    # "/dataset/c3vd_train/c3vd/test/desc_t4_a_up",
    "/dataset/c3vd_train/c3vd/test/sigmoid_t3_a",
    "/dataset/c3vd_train/c3vd/test/trans_t2_a",
]
    

ckpt_path = "/home/weijunlin/project/PC-Depth/log2/no_Mask/version_0/last.ckpt"

@torch.no_grad()
def main():
    hparams = get_opts()

    # initialize network
    system = SC_Depth(hparams)

    # load ckpts
    system = system.load_from_checkpoint(ckpt_path, strict=False)

    model = system.depth_net
    model.cuda()
    model.eval()
    test_transform = custom_transforms.Compose([
        custom_transforms.RescaleTo((256, 320)),
        custom_transforms.ArrayToTensor()]
    )

    for test_dir in test_dirs:
        img_paths = glob.glob("{}/*.png".format(test_dir))
        img_paths = sorted(img_paths, key=extract_number)
        print(len(img_paths))

        pred_depth_list = []
        for img_path in img_paths:
            tgt_img = imread(img_path)
            tgt_img_tensor = test_transform([tgt_img])[0][0].unsqueeze(0)
            pred_depth = model(tgt_img_tensor.cuda())[0]
            pred_depth_list.append(pred_depth.cpu().numpy())

        pred_depth = np.concatenate(pred_depth_list)
        print(pred_depth.shape)
        np.savez_compressed("./test/c3vd_pred_depth2/aba/no_Mask/{}.npz".format(os.path.basename(test_dir)), data=pred_depth)

if __name__ == '__main__':
    main()
