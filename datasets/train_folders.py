from torch import tensor
import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import os
from torchvision import transforms

from .specular_detection import SpecularDetection

def load_as_float(path):
    return imread(path).astype(np.float32)


def generate_sample_index(num_frames, skip_frames, sequence_length):
    sample_index_list = []
    k = skip_frames
    demi_length = (sequence_length-1)//2
    shifts = list(range(-demi_length * k, demi_length * k + 1, k))
    shifts.pop(demi_length)

    if num_frames > sequence_length:
        for i in range(demi_length * k, num_frames-demi_length * k):
            sample_index = {'tgt_idx': i, 'ref_idx': []}
            for j in shifts:
                sample_index['ref_idx'].append(i+j)
            sample_index_list.append(sample_index)

    return sample_index_list

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


class TrainFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self,
                 root,
                 train=True,
                 sequence_length=3,
                 transform=None,
                 skip_frames=1,
                 dataset='c3vd',
                 do_color_aug=False,
                 split="train.txt"):
        np.random.seed(0)
        random.seed(0)
        self.root = Path(root)
        self.transform = transform
        self.dataset = dataset
        self.k = skip_frames
        self.train = train
        self.do_color_aug = do_color_aug

        if dataset == "SCARED":
            self.crawl_folders_SCARED(sequence_length)
        else:
            scene_list_path = self.root/split if train else self.root/"val.txt"
            self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
            self.crawl_folders(sequence_length)

    def extract_number(self, filename):
        return int(filename.split('/')[-1].split('.')[0])
    
    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []

        print("Find Scene:")
        intrinsics = np.genfromtxt(self.root/"cam.txt", delimiter=",").astype(np.float32).reshape((3, 3))
        for scene in self.scenes:
            print(scene.name)
            imgs = sorted(scene.files('*.png'), key=self.extract_number)

            if len(imgs) < sequence_length:
                continue

            sample_index_list = generate_sample_index(
                len(imgs), self.k, sequence_length)
            for sample_index in sample_index_list:
                sample = {'intrinsics': intrinsics,
                          'tgt_img': imgs[sample_index['tgt_idx']]}

                sample['ref_imgs'] = []
                for j in sample_index['ref_idx']:
                    sample['ref_imgs'].append(imgs[j])
                sequence_set.append(sample)

        self.samples = sequence_set


    def crawl_folders_SCARED(self, sequence_length):
        # k skip frames
        intrinsics = np.array([[0.82, 0, 0.5],
                               [0, 1.02, 0.5],
                               [0, 0, 1]], dtype=np.float32)
        intrinsics[0, :] *= 1280
        intrinsics[1, :] *= 1024

        sequence_set = []
        fpath = self.root/"train.txt" if self.train else self.root/"val.txt"
        print(fpath)

        def get_image_path(folder, frame_index):
            f_str = "{:010d}.png".format(frame_index)
            image_path = os.path.join(self.root, folder, "image_02/", f_str)
            return image_path

        train_filenames = readlines(fpath)
        for filenames in train_filenames:
            line = filenames.split()
            folder = line[0]
            frame_index = int(line[1])
            tar_img_path = get_image_path(folder, frame_index)
            ref_img_paths = [get_image_path(folder, frame_index-self.k), get_image_path(folder, frame_index+self.k)]

            if os.path.isfile(tar_img_path) and os.path.isfile(ref_img_paths[0]) and os.path.isfile(ref_img_paths[1]):
                sample = {'intrinsics': intrinsics,
                             'tgt_img': tar_img_path,
                            'ref_imgs': ref_img_paths}
                sequence_set.append(sample)


        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt_img'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]

        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, 
                                              np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])

        do_color_aug = self.do_color_aug and random.random() > 0.5

        if do_color_aug:
            color_aug = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        else:
            color_aug = (lambda x: x)

        
        imgs_aug = [color_aug(img) for img in imgs]
        tgt_img = imgs[0]
        ref_imgs = imgs[1:]

        inputs = {}
        inputs["color"] = (tgt_img, ref_imgs, intrinsics)
        inputs["color_aug"] = (imgs_aug[0], imgs_aug[1:])
        

        return inputs

    def __len__(self):
        return len(self.samples)

class TrainFolderIA(TrainFolder):

    def __init__(self, 
                 root, 
                 train=False, 
                 sequence_length=3, 
                 transform=None,
                 skip_frames=1, 
                 dataset='c3vd',
                 do_color_aug=False,
                 split="train.txt"):
        super().__init__(root, train, sequence_length, transform, skip_frames, dataset)
        self.train = train
        if dataset == "c3vd":
            T1, T2_abs, T2_rel, T3, N_min = (250, 230, 2.0, 4, 400)
        elif dataset == "SCARED":
            T1, T2_abs, T2_rel, T3, N_min = (250, 230, 0.95, 4, 5000)
        self.detector = SpecularDetection(T1, T2_abs, T2_rel, T3, N_min)
        self.do_color_aug = do_color_aug

        self.brightness = 0.2
        self.contrast = 0.2
        self.saturation = 0.2
        self.hue = 0.1

        assert self.transform is not None

    
    def __getitem__(self, index):
        sample = self.samples[index]

        tgt_img = load_as_float(sample['tgt_img'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]

        imgs, intrinsics = self.transform([tgt_img] + ref_imgs, 
                                            np.copy(sample['intrinsics']))

        do_color_aug = self.do_color_aug and random.random() > 0.5

        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        
        imgs_aug = [color_aug(img) for img in imgs]
        tgt_img = imgs[0]
        ref_imgs = imgs[1:]
        tgt_specular_mask = self.detector(tgt_img)
        refs_specular_mask = []
        for ref_img in ref_imgs:
            ref_specular_mask = self.detector(ref_img)
            refs_specular_mask.append(ref_specular_mask)

        inputs = {}
        inputs["color"] = (tgt_img, ref_imgs, intrinsics, tgt_specular_mask, refs_specular_mask)
        inputs["color_aug"] = (imgs_aug[0], imgs_aug[1:])
        
        return inputs