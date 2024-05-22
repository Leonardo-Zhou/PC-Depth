import numpy as np

from imageio.v2 import imread

import glob
import os

def extract_number(filename):
    return int(filename.split('/')[-1].split('.')[0])

test_dirs = [
    "/dataset/c3vd_train/c3vd/test/cecum_t2_a",
    "/dataset/c3vd_train/c3vd/test/sigmoid_t3_a",
    "/dataset/c3vd_train/c3vd/test/trans_t2_a",
]

def main():
    for test_dir in test_dirs:
        img_paths = glob.glob("{}/*.tiff".format(test_dir))
        img_paths = sorted(img_paths, key=extract_number)
        print(len(img_paths))

        gt_depth_list = []
        for img_path in img_paths:
            print(img_path)
            tgt_img = imread(img_path).astype(np.float32) * 100 /(2**16-1)
            gt_depth_list.append(tgt_img)

        pred_depth = np.stack(gt_depth_list)

        np.savez_compressed("./test/c3vd_gt_depth/{}.npz".format(os.path.basename(test_dir)), data=pred_depth)

if __name__ == '__main__':
    main()
