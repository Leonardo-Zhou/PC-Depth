# import sys
# sys.path.append("..")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from path import Path

import datasets.custom_transforms as custom_transforms
from config import get_opts, get_training_size
from datasets.test_folder import *
from losses.loss_functions import compute_errors
from SC_Depth import SC_Depth
from visualization import *
from losses.inverse_warp import pose_vec2mat
import glob
from imageio.v2 import imread

def extract_number(filename):
    return int(filename.split('/')[-1].split('.')[0])

test_dirs = [
    "/dataset/c3vd_train/c3vd/test/cecum_t2_a",
    # "/dataset/c3vd_train/c3vd/test/desc_t4_a_up",
    "/dataset/c3vd_train/c3vd/test/sigmoid_t3_a",
    "/dataset/c3vd_train/c3vd/test/trans_t2_a",
]

# ckpt_path = "/home/weijunlin/project/PC-Depth/log/SC_c3vd_skip/skip15/last.ckpt"
ckpt_path = "/home/weijunlin/project/PC-Depth/log/HS_test/version_0/last.ckpt"
ckpt_path = "/home/weijunlin/project/PC-Depth/log2/IA_aba/no_inpaint"


@torch.no_grad()
def main():
    hparams = get_opts()

    # initialize network
    system = SC_Depth(hparams)

    # load ckpts
    system = system.load_from_checkpoint(ckpt_path, strict=False)

    model = system.pose_net
    model.cuda()
    model.eval()

    inference_transform = custom_transforms.Compose([
        custom_transforms.RescaleTo((256,320)),
        custom_transforms.ArrayToTensor()]
    )

    skip = 5
    for test_dir in test_dirs:
        img_paths = glob.glob("{}/*.png".format(test_dir))
        img_paths = sorted(img_paths, key=extract_number)
        img_lens = len(img_paths)

        all_pose = []
        i = 0
        while i + skip< img_lens:
            img_a = imread(img_paths[i])
            img_b = imread(img_paths[i+skip])
            tensor_img = inference_transform([img_a, img_b])[0]
            img_a = tensor_img[0].unsqueeze(0)
            img_b = tensor_img[1].unsqueeze(0)
            # print(img_a.shape, img_b.shape)
            pred_pose = model(img_b.cuda(), img_a.cuda())
            T34 = pose_vec2mat(pred_pose)[0].cpu().numpy()
            T = np.eye(4)
            T[:3, :] = T34
            all_pose.append(np.linalg.pinv(T))
            i = i + skip
        all_pose = np.stack(all_pose)
        print(all_pose.shape)
        fpath = "./test/c3vd_pred_pose5/aba/no_inpaint/{}.npz".format(Path(test_dir).basename())
        print("save ", fpath)
        np.savez_compressed(fpath, data=all_pose)


if __name__ == '__main__':
    main()
