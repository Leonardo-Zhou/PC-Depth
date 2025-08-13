import numpy as np
from tqdm import tqdm

# from losses.loss_functions import compute_errors
from visualization import *

import sys
import os



def compute_errors(gt, pred, dataset):
    # pred : b c h w
    # gt: b h w

    abs_diff = abs_rel = sq_rel = log10 = rmse = rmse_log = a1 = a2 = a3 = 0.0

    h, w = gt.shape
    # print("gt size:", h, w)
    # print("dataset_name: ", dataset)
    pred = cv2.resize(pred, (w, h))

    min_depth = 0.1

    if dataset == 'c3vd':
        crop_mask = gt != gt
        crop_mask[:, :] = 1
        max_depth = 100

    if dataset == 'SCARED':
        crop_mask = gt != gt
        crop_mask[:, :] = 1
        min_depth = 0.1
        max_depth = 150

    valid = (gt > min_depth) & (gt < max_depth)
    valid = valid & crop_mask

    valid_gt = gt[valid]
    valid_pred = pred[valid]

    # align scale
    # print("scale: ", torch.median(valid_gt)/torch.median(valid_pred))
    valid_pred = valid_pred * \
        np.median(valid_gt)/np.median(valid_pred)

    # valid_pred = valid_pred.clamp(min_depth, max_depth)
    valid_pred[valid_pred < min_depth] = min_depth
    valid_pred[valid_pred > max_depth] = max_depth


    pred = valid_pred
    gt = valid_gt
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def eval(pred_depths, gt_depths, dataset_name):
    print(pred_depths.shape, gt_depths.shape)

    length = pred_depths.shape[0]
    all_errs = []
    for i, _ in enumerate(tqdm(range(length))):
        pred_depth = pred_depths[i, :, :]
        # pred_depth = 1.0/(pred_depth)
        # pred_depth = 1.0 / (pred_depth + 10)
        # pred_depth = 1.0/(pred_depth + pred_depth.max())
        gt_depth = gt_depths[i, :, :]

        errs = compute_errors(gt_depth, pred_depth, dataset_name)

        all_errs.append(np.array(errs))

    all_errs = np.stack(all_errs)
    mean_errs = np.mean(all_errs, axis=0)


    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errs.tolist()) + "\\\\")
    print("\n-> Done!")

if __name__ == '__main__':
    pred_dir = sys.argv[1]
    dataset = sys.argv[2]
    if dataset == "c3vd":
        files = [
            "cecum_t2_a.npz",
            # "desc_t4_a_up.npz",
            "sigmoid_t3_a.npz",
            "trans_t2_a.npz",
        ]
        gt_dir = "test/c3vd_gt_depth"

        gt = []
        pred = []

        for file in files:
            print(file)
            pred_file = os.path.join(pred_dir, file)
            gt_file = os.path.join(gt_dir, file)
            pred_depths = np.load(pred_file, fix_imports=True, encoding='latin1')["data"]
            pred_depths = pred_depths.squeeze()
            gt_depths = np.load(gt_file, fix_imports=True, encoding='latin1')["data"]
            gt_depths = gt_depths.squeeze()

            pred.append(pred_depths)
            gt.append(gt_depths)

        gt = np.concatenate(gt, axis=0)
        pred = np.concatenate(pred, axis=0)

        print(gt.shape, pred.shape)


        eval(pred, gt, dataset)


    elif dataset == "SCARED":
        gt_path = "./splits/endovis/gt_depths.npz"
        pred_depths = np.load(pred_dir, fix_imports=True, encoding='latin1')["data"]
        pred_depths = pred_depths.squeeze()
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
        gt_depths = gt_depths.squeeze()

        eval(pred_depths, gt_depths, dataset)
            
