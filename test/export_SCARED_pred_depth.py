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

import glob
import os

from imageio import imread

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

@torch.no_grad()
def main():
    hparams = get_opts()

    # initialize network
    system = SC_Depth(hparams)

    # load ckpts
    ckpt_path = "/home/weijunlin/project/PC-Depth/log_old/IA_SCARED/version_2/last.ckpt"
    # ckpt_path = "/home/weijunlin/project/PC-Depth/log2/IA/version_0/last.ckpt"
    system = system.load_from_checkpoint(ckpt_path, strict=False)

    model = system.depth_net
    model.cuda()
    model.eval()

    # get training resolution
    training_size = get_training_size("SCARED")

    # data loader
    test_transform = custom_transforms.Compose([
        custom_transforms.RescaleTo(training_size),
        custom_transforms.ArrayToTensor()]
    )

    root = Path("/dataset/SCARED_train")
    fpath = root/"test.txt"
    def get_image_path(folder, frame_index):
        f_str = "{:010d}.png".format(frame_index)
        image_path = os.path.join(root, folder, "image_02/", f_str)
        return image_path

    train_filenames = readlines(fpath)
    print("test data length:", len(train_filenames))
    pred_depth_list = []
    for filenames in train_filenames:
        line = filenames.split()
        folder = line[0]
        frame_index = int(line[1])
        tar_img_path = get_image_path(folder, frame_index)

        tgt_img = imread(tar_img_path)
        tgt_img_tensor = test_transform([tgt_img])[0][0].unsqueeze(0)

        pred_depth = model(tgt_img_tensor.cuda())[0]
        pred_depth_list.append(pred_depth.cpu().numpy())

    pred_depth = np.concatenate(pred_depth_list)
    print(pred_depth.shape)
    np.savez_compressed("./test/shift/temp.npz", data=pred_depth)

if __name__ == '__main__':
    main()
