from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .inverse_warp import inverse_warp, pose_vec2mat
from .mask_ranking_loss import Mask_Ranking_Loss
from .normal_ranking_loss import EdgeguidedNormalRankingLoss

from .lightaligh import *

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        k = 7
        self.mu_x_pool = nn.AvgPool2d(k, 1)
        self.mu_y_pool = nn.AvgPool2d(k, 1)
        self.sig_x_pool = nn.AvgPool2d(k, 1)
        self.sig_y_pool = nn.AvgPool2d(k, 1)
        self.sig_xy_pool = nn.AvgPool2d(k, 1)

        self.refl = nn.ReflectionPad2d(k//2)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * \
            (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


compute_ssim_loss = SSIM().to(device)

normal_ranking_loss = EdgeguidedNormalRankingLoss().to(device)

mask_ranking_loss = Mask_Ranking_Loss().to(device)


def photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths, intrinsics, poses, poses_inv, hparams):

    diff_img_list = []
    diff_color_list = []
    diff_depth_list = []
    valid_mask_list = []

    for ref_img, ref_depth, pose, pose_inv in zip(ref_imgs, ref_depths, poses, poses_inv):
        diff_img_tmp1, diff_color_tmp1, diff_depth_tmp1, valid_mask_tmp1 = compute_pairwise_loss(
            tgt_img, ref_img, tgt_depth,
            ref_depth, pose, intrinsics,
            hparams
        )
        diff_img_tmp2, diff_color_tmp2, diff_depth_tmp2, valid_mask_tmp2 = compute_pairwise_loss(
            ref_img, tgt_img, ref_depth,
            tgt_depth, pose_inv, intrinsics,
            hparams
        )
        diff_img_list += [diff_img_tmp1, diff_img_tmp2]
        diff_color_list += [diff_color_tmp1, diff_color_tmp2]
        diff_depth_list += [diff_depth_tmp1, diff_depth_tmp2]
        valid_mask_list += [valid_mask_tmp1, valid_mask_tmp2]

    diff_img = torch.cat(diff_img_list, dim=1)
    diff_color = torch.cat(diff_color_list, dim=1)
    diff_depth = torch.cat(diff_depth_list, dim=1)
    valid_mask = torch.cat(valid_mask_list, dim=1)

    # using photo loss to select best match in multiple views
    if not hparams.no_min_optimize:
        indices = torch.argmin(diff_color, dim=1, keepdim=True)

        diff_img = torch.gather(diff_img, 1, indices)
        diff_depth = torch.gather(diff_depth, 1, indices)
        valid_mask = torch.gather(valid_mask, 1, indices)

    photo_loss = mean_on_mask(diff_img, valid_mask)
    geometry_loss = mean_on_mask(diff_depth, valid_mask)

    return photo_loss, geometry_loss


def compute_PC_loss(tgt_img, ref_imgs, tgt_depth, ref_depths, intrinsics, poses, poses_inv, 
                    tgt_specular, ref_speculars, lightAligh:LightAlign, hparams):
    diff_img_list = []
    diff_color_list = []
    diff_depth_list = []
    valid_mask_list = []

    unspecular_list = []

    for ref_img, ref_depth, pose, pose_inv, ref_specular in zip(ref_imgs, ref_depths, poses, poses_inv, ref_speculars):
        diff_img_tmp1, diff_color_tmp1, diff_depth_tmp1, valid_mask_tmp1, \
        ref_img_warped, k, specular_loss, sls_tar, unspecular_mask1 = compute_pairwise_loss_PC(
            tgt_img, ref_img, tgt_depth,
            ref_depth, pose, intrinsics,
            tgt_specular, ref_specular,
            lightAligh,
            hparams
        )
        diff_img_tmp2, diff_color_tmp2, diff_depth_tmp2, valid_mask_tmp2,\
              _, _, _, _, unspecular_mask2 = compute_pairwise_loss_PC(
            ref_img, tgt_img, ref_depth,
            tgt_depth, pose_inv, intrinsics,
            ref_specular, tgt_specular,
            lightAligh,
            hparams
        )

        # specular_loss is a tensor, not a dict
        point_mask = tgt_specular["point"] if isinstance(tgt_specular, dict) else tgt_specular

        diff_img_list += [diff_img_tmp1, diff_img_tmp2]
        diff_color_list += [diff_color_tmp1, diff_color_tmp2]
        diff_depth_list += [diff_depth_tmp1, diff_depth_tmp2]
        valid_mask_list += [valid_mask_tmp1, valid_mask_tmp2]


        unspecular_list += [unspecular_mask1, unspecular_mask2]


    diff_img = torch.cat(diff_img_list, dim=1)
    diff_color = torch.cat(diff_color_list, dim=1)
    diff_depth = torch.cat(diff_depth_list, dim=1)
    valid_mask = torch.cat(valid_mask_list, dim=1)

    unspecular_mask = torch.cat(unspecular_list, dim=1)

    # using photo loss to select best match in multiple views
    if not hparams.no_min_optimize:
        indices = torch.argmin(diff_color, dim=1, keepdim=True)

        diff_img = torch.gather(diff_img, 1, indices)
        diff_depth = torch.gather(diff_depth, 1, indices)
        valid_mask = torch.gather(valid_mask, 1, indices)
        unspecular_mask = torch.gather(unspecular_mask, 1, indices)

    LP_mask = valid_mask
    if not hparams.no_specular_mask:
        LP_mask = valid_mask * unspecular_mask
    photo_loss = mean_on_mask(diff_img, LP_mask)
    geometry_loss = mean_on_mask(diff_depth, valid_mask)
    specular_loss = mean_on_mask(specular_loss, point_mask)

    return photo_loss, geometry_loss, specular_loss, {"k": k, "warped": ref_img_warped, "sls": sls_tar, "mask":LP_mask }


def compute_pairwise_loss(tgt_img, ref_img, tgt_depth, ref_depth, pose, intrinsic, hparams):

    ref_img_warped, ref_img_warped2, projected_depth, computed_depth = inverse_warp(
        ref_img, tgt_depth, ref_depth, pose, intrinsic, padding_mode='zeros')

    diff_depth = (computed_depth-projected_depth).abs() / \
        (computed_depth+projected_depth)

    # masking zero values
    valid_mask_ref = (ref_img_warped.abs().mean(
        dim=1, keepdim=True) > 1e-3).float()
    valid_mask_tgt = (tgt_img.abs().mean(dim=1, keepdim=True) > 1e-3).float()
    valid_mask = valid_mask_tgt * valid_mask_ref

    ref_img_warped2 = valid_mask_ref * ref_img_warped2
    ref_img_warped = ref_img_warped2

    diff_color = (tgt_img-ref_img_warped).abs().mean(dim=1, keepdim=True)
    if not hparams.no_auto_mask:
        identity_warp_err = (tgt_img-ref_img).abs().mean(dim=1, keepdim=True)
        auto_mask = (diff_color < identity_warp_err).float()
        valid_mask = auto_mask * valid_mask

    diff_img = (tgt_img-ref_img_warped).abs().clamp(0, 1)
    if not hparams.no_ssim:
        ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)
    diff_img = torch.mean(diff_img, dim=1, keepdim=True)

    # reduce photometric loss weight for dynamic regions
    if not hparams.no_dynamic_mask:
        weight_mask = (1-diff_depth).detach()
        diff_img = diff_img * weight_mask

    return diff_img, diff_color, diff_depth, valid_mask


def mask_or(mask1, mask2):
    return torch.logical_or(mask1, mask2).to(torch.float32)

def mask_and(mask1, mask2):
    return torch.logical_and(mask1, mask2).to(torch.float32)


def mask_not(mask):
    return torch.logical_not(mask).to(torch.float32)

def get_unspecular_mask(tgt_mask, ref_mask):
    # 处理字典类型输入
    if isinstance(tgt_mask, dict):
        tgt_specular_mask = tgt_mask["point"]
    else:
        tgt_specular_mask = tgt_mask
        
    if isinstance(ref_mask, dict):
        ref_specular_mask = ref_mask["point"]
    else:
        ref_specular_mask = ref_mask
        
    specular_mask = mask_or(tgt_specular_mask, ref_specular_mask)
    unspecular_mask = mask_not(specular_mask)
    return unspecular_mask

def compute_pairwise_loss_PC(tgt_img, ref_img, tgt_depth, ref_depth, 
                             pose, intrinsic, tgt_mask, ref_mask, 
                             lightAligh:LightAlign, 
                             hparams):
    """used in PC-Depth"""
    T = pose_vec2mat(pose)
    
    # 从字典中提取张量
    tgt_highlight = tgt_mask["point"] if isinstance(tgt_mask, dict) else tgt_mask
    ref_highlight = ref_mask["point"] if isinstance(ref_mask, dict) else ref_mask
    
    refined, unrefined, projected_depth, projected_mask, computed_depth, sls_tar, k, specular_loss = lightAligh(
        ref_img, tgt_img, tgt_depth, tgt_highlight, ref_highlight, ref_depth, T, intrinsic, padding_mode='zeros')
    
    if not hparams.no_light_align:
        ref_img_warped = refined
    else:
        ref_img_warped = unrefined


    computed_depth = computed_depth

    # used in L_GC
    diff_depth = (computed_depth - projected_depth).abs() / (computed_depth + projected_depth)

    valid_mask_ref = (ref_img_warped.abs().mean(dim=1, keepdim=True) > 1e-3).float()
    valid_mask_tgt = (tgt_img.abs().mean(dim=1, keepdim=True) > 1e-3).float()
    valid_mask = valid_mask_tgt * valid_mask_ref

    unspecular_mask = get_unspecular_mask(tgt_mask, projected_mask).detach()

    diff_color = (tgt_img - unrefined).abs().mean(dim=1, keepdim=True)

    if not hparams.no_auto_mask:
        identity_warp_err = (tgt_img-ref_img).abs().mean(dim=1, keepdim=True)
        auto_mask = (diff_color < identity_warp_err).float()
        valid_mask = auto_mask * valid_mask

    diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)

    if not hparams.no_ssim:
        ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)

    diff_img = torch.mean(diff_img, dim=1, keepdim=True)

    # reduce photometric loss weight for dynamic regions
    if not hparams.no_dynamic_mask:
        weight_mask = (1-diff_depth).detach()
        diff_img = diff_img * weight_mask

    return diff_img, diff_color, diff_depth, valid_mask, ref_img_warped, k, specular_loss, sls_tar, unspecular_mask


# compute mean value on a binary mask
def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    if mask.sum() > 100:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0).float().to(device)
    return mean_value


def compute_smooth_loss(tgt_depth, tgt_img, isNorm=True):
    def get_smooth_loss(disp, img,isNorm):
        """
        Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """

        # normalize
        if isNorm:
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            disp = norm_disp

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(
            torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(
            torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    loss = get_smooth_loss(tgt_depth, tgt_img, isNorm)

    return loss


def compute_smooth_loss_mask(tgt_depth, tgt_img, mask):
    def get_smooth_loss(disp, img, mask):
        """
        Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """

        # normalize
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        disp = norm_disp

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(
            torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(
            torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        # print(mask.shape)
        mask_ = mask_not(mask)
        mask_x = mask_[:, :, :, :-1]
        mask_y = mask_[:, :, :-1, :]

        grad_disp_x = (grad_disp_x * mask_x).sum() / mask_x.sum()
        grad_disp_y = (grad_disp_y * mask_y).sum() / mask_y.sum()

        return grad_disp_x + grad_disp_y

    loss = get_smooth_loss(tgt_depth, tgt_img, mask)

    return loss

def compute_smooth_loss2(tgt_depth):
    def get_smooth_loss(disp):
        """
        Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """

        # normalize
        # mean_disp = disp.mean(2, True).mean(3, True)

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        return grad_disp_x.mean() + grad_disp_y.mean()

    loss = get_smooth_loss(tgt_depth)

    return loss


@torch.no_grad()
def compute_errors(gt, pred, dataset):
    # pred : b c h w
    # gt: b h w

    abs_diff = abs_rel = sq_rel = log10 = rmse = rmse_log = a1 = a2 = a3 = 0.0

    batch_size, h, w = gt.size()
    # print("gt size:", h, w)
    # print("dataset_name: ", dataset)
    if pred.nelement() != gt.nelement():
        pred = F.interpolate(
            pred, [h, w], mode='bilinear', align_corners=False)

    pred = pred.view(batch_size, h, w)
    min_depth = 0.1

    if dataset == 'kitti':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 80

    if dataset == 'nyu':
        crop_mask = gt[0] != gt[0]
        crop = np.array([45, 471, 41, 601]).astype(np.int32)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        max_depth = 10

    if dataset == 'ddad':
        crop_mask = gt[0] != gt[0]
        crop_mask[:, :] = 1
        max_depth = 200

    if dataset == 'bonn':
        crop_mask = gt[0] != gt[0]
        crop_mask[:, :] = 1
        max_depth = 10

    if dataset == 'tum':
        crop_mask = gt[0] != gt[0]
        crop_mask[:, :] = 1
        max_depth = 10
    
    if dataset == 'c3vd':
        crop_mask = gt[0] != gt[0]
        crop_mask[:, :] = 1
        max_depth = 100

    if dataset == 'SCARED':
        crop_mask = gt[0] != gt[0]
        crop_mask[:, :] = 1
        min_depth = 0.1
        max_depth = 150

    
    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > min_depth) & (current_gt < max_depth)
        valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid]


        # align scale
        # print("scale: ", torch.median(valid_gt)/torch.median(valid_pred))
        valid_pred = valid_pred * \
            torch.median(valid_gt)/torch.median(valid_pred)
    
        if dataset == 'SCARED':
            # print('SCARED min max')
            # valid_pred[valid_pred]
            # print(valid_pred.max())
            valid_pred[valid_pred < min_depth] = min_depth
            valid_pred[valid_pred > max_depth] = max_depth

        valid_pred = valid_pred.clamp(min_depth, max_depth)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        diff_i = valid_gt - valid_pred
        abs_diff += torch.mean(torch.abs(diff_i))
        abs_rel += torch.mean(torch.abs(diff_i) / valid_gt)
        sq_rel += torch.mean(((diff_i)**2) / valid_gt)
        rmse += torch.sqrt(torch.mean(diff_i ** 2))
        rmse_log += torch.sqrt(torch.mean((torch.log(valid_gt) -
                               torch.log(valid_pred)) ** 2))
        log10 += torch.mean(torch.abs((torch.log10(valid_gt) -
                            torch.log10(valid_pred))))

    return [metric.item() / batch_size for metric in [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]]
