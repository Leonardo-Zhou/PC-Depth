import torch.utils.data as data
import numpy as np
from imageio.v2 import imread
from path import Path
import torch
from scipy import sparse
import os

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def load_sparse_depth(filename):
    sparse_depth = sparse.load_npz(filename)
    depth = sparse_depth.todense()
    return np.array(depth)


def extract_number(filename):
        return int(filename.split('/')[-1].split('.')[0])

def crawl_folder(folder, dataset='nyu'):
    print("folder: ", folder)
    # imgs = sorted((folder/'color/').files('*.png') +
    #               (folder/'color/').files('*.jpg'))

    if dataset == 'c3vd':
        imgs = sorted(folder.files('*.png'), key=extract_number)
        depths = sorted(folder.files('*.tiff'), key=extract_number)
    elif dataset == "SCARED":
        imgs = []
        depths = []

        fpath = folder/"test.txt"
        train_filenames = readlines(fpath)

        def get_image_path(subfolder, frame_index):
            f_str = "{:010d}.png".format(frame_index)
            image_path = os.path.join(folder, subfolder, "image_02/", f_str)
            # D:\dataset\SCARED_depth\dataset2\keyframe4\groundtruth
            return image_path

        def get_image_path2(subfolder:Path, frame_index):
            depth_folder = folder.replace("SCARED_train", "SCARED_depth")
            f_str = "scene_points{:06d}.tiff".format(frame_index - 1)
            gt_depth_path = os.path.join(
                depth_folder,
                subfolder,
                "groundtruth",
                f_str)
            return gt_depth_path
        
        for filenames in train_filenames:
            line = filenames.split()
            subfolder = line[0]
            frame_index = int(line[1])
            tar_img_path = get_image_path(subfolder, frame_index)
            tar_depth_path = get_image_path2(subfolder, frame_index)
            imgs.append(tar_img_path)
            depths.append(tar_depth_path)

    return imgs, depths

class TestSet(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/color/0000000.png
        root/depth/0000000.npz or png
    """

    def __init__(self, root, transform=None, dataset='c3vd'):
        self.root = Path(root)
        self.transform = transform
        self.dataset = dataset
        self.imgs, self.depths = crawl_folder(self.root, self.dataset)

    def __getitem__(self, index):
        img = imread(self.imgs[index]).astype(np.float32)

        if self.dataset == 'nyu':
            depth = torch.from_numpy(
                imread(self.depths[index]).astype(np.float32)).float()/5000
        elif self.dataset == 'c3vd':
            depth = torch.from_numpy(
                imread(self.depths[index]).astype(np.float32)).float() * 100 /(2**16-1)
        elif self.dataset == 'SCARED':
            depth = torch.from_numpy(
                imread(self.depths[index]).astype(np.float32)).float()

        if self.transform is not None:
            img, _ = self.transform([img], None)
            img = img[0]

        return img, depth

    def __len__(self):
        return len(self.imgs)


class TestPoseSet(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/color/0000000.png
        root/depth/0000000.npz or png
    """

    def __init__(self, root, transform=None, dataset='c3vd'):
        self.root = Path(root)
        self.transform = transform
        self.dataset = dataset
        self.crawl_folder()

    def __getitem__(self, index):
        img_a = imread(self.imgs[index]).astype(np.float32)
        img_b = imread(self.imgs[index + 1]).astype(np.float32)
        img = [img_a, img_b]
        if self.transform is not None:
            img, _ = self.transform(img, None)
            # img = img[0]

        return img[0], img[1]

    def extract_number(self, filename):
        return int(filename.split('/')[-1].split('.')[0])

    def crawl_folder(self):
        # k skip frames
        imgs = sorted(self.root.files('*.png'), key=self.extract_number)
        print(imgs)
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)- 1