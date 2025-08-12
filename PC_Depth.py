from os import wait4
import numpy as np
import torch
from pytorch_lightning import LightningModule

import losses.loss_functions as LossF
from models.DepthNet import DepthNet
from models.PoseNet import PoseNet
from visualization import *

from losses.lightaligh import *

from config import get_training_size


class PC_Depth(LightningModule):
    def __init__(self, hparams):
        super(PC_Depth, self).__init__()
        self.save_hyperparameters()

        # model
        self.depth_net = DepthNet(self.hparams.hparams.resnet_layers)
        self.pose_net = PoseNet()

        (H, W) = get_training_size(hparams.dataset_name)
        self.lightAligh = LightAlign(hparams.batch_size, H, W, hparams.light_mu, hparams.light_gamma)


    def configure_optimizers(self):
        optim_params = [
            {'params': self.depth_net.parameters(), 'lr': self.hparams.hparams.lr},
            {'params': self.pose_net.parameters(), 'lr': self.hparams.hparams.lr}
        ]
        optimizer = torch.optim.Adam(optim_params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=self.hparams.hparams.scheduler_step_size,
                                                    gamma=0.1)
        return [optimizer],[scheduler]

    def training_step(self, batch, batch_idx):
        tgt_img, ref_imgs, intrinsics, tgt_specular, ref_speculars = batch["color"]
        tgt_img_aug, ref_imgs_aug = batch["color_aug"]

        # network forward
        tgt_depth = self.depth_net(tgt_img_aug)
        ref_depths = [self.depth_net(im) for im in ref_imgs_aug]
        poses = [self.pose_net(tgt_img_aug, im) for im in ref_imgs_aug]
        poses_inv = [self.pose_net(im, tgt_img_aug) for im in ref_imgs_aug]

        # compute loss
        w1 = self.hparams.hparams.photo_weight
        w2 = self.hparams.hparams.geometry_weight
        w3 = self.hparams.hparams.highlight_weight
        w4 = self.hparams.hparams.smooth_weight

        loss_1, loss_2, loss_3, imgs = LossF.compute_PC_loss(tgt_img, ref_imgs, tgt_depth,
                                                             ref_depths,intrinsics, poses, poses_inv,
                                                             tgt_specular, ref_speculars,
                                                             self.lightAligh,
                                                             self.hparams.hparams)
        if self.hparams.hparams.no_inpaint_smooth:
            loss_4 = LossF.compute_smooth_loss(tgt_depth, tgt_img)
        else:
            loss_4 = LossF.compute_smooth_loss(tgt_depth, tgt_specular["inpaint"])

        if self.hparams.hparams.no_point_loss:
            w3 = 0

        loss = w1*loss_1 + w2*loss_2 +  w3*loss_3 + w4*loss_4


        # create logs
        self.log('train/total_loss', loss)

        self.log('train/photo_loss', loss_1)
        self.log('train/geometry_loss', loss_2)
        self.log('train/point_loss', loss_3)
        self.log('train/smooth_loss', loss_4)
        
        if batch_idx % 100 == 0:
            for i in range(min(1, self.hparams.hparams.batch_size)):
                vis_img = tgt_specular["inpaint"][i].cpu()  # (3, H, W)
                vis_depth = visualize_depth(tgt_depth[i, 0].detach())  # (3, H, W)
                stack = torch.cat([vis_img, vis_depth], dim=1).unsqueeze(0)  # (1, 3, 2*H, W)

                ref1 = imgs["mask"][i].detach().cpu()
                ref1 = torch.cat([ref1]*3, dim=0)

                ref2 = ref_imgs_aug[1][i].cpu()
                refs = torch.cat([ref1, ref2], dim=1).unsqueeze(0)

                stack = torch.cat([stack, refs], dim=3) # (1, 3, 2*H, 2*W)

                self.logger.experiment.add_images(
                    'train/img_depth/{}/{}'.format(self.current_epoch, i), stack, batch_idx)
                

                vis_img = tgt_img[i].cpu()  # (3, H, W)
                vis_img2 = visualize_k(imgs["k"][i][0])
                ref1 = imgs["warped"][i].cpu().unsqueeze(0)
                ref2 = imgs["sls"][i].detach().cpu().unsqueeze(0)
                ref2 = ref2 / ref2.max()
                ref2 = torch.cat([ref2]*3, dim=1)
                refs = torch.cat([ref1, ref2], dim=2)
                stack = torch.cat([vis_img, vis_img2], dim=1).unsqueeze(0)  # (1, 3, 2*H, W)
                stack = torch.cat([stack, refs], dim=3) # (1, 3, 2*H, 2*W)

                self.logger.experiment.add_images(
                    'train/img_warp/{}/{}'.format(self.current_epoch, i), stack, batch_idx)
                
        return loss

    def validation_step(self, batch, batch_idx):
        tgt_img, ref_imgs, intrinsics, tgt_specular, ref_speculars = batch["color"]

        tgt_depth = self.depth_net(tgt_img)
        ref_depths = [self.depth_net(im) for im in ref_imgs]
        poses = [self.pose_net(tgt_img, im) for im in ref_imgs]
        poses_inv = [self.pose_net(im, tgt_img) for im in ref_imgs]

        loss_1, loss_2, loss_3, imgs = LossF.compute_PC_loss(tgt_img, ref_imgs, tgt_depth, 
                                                ref_depths,intrinsics, poses, poses_inv,
                                                tgt_specular, ref_speculars, self.lightAligh,
                                                self.hparams.hparams,
                                                self.current_epoch)
        errs = {'photo_loss': loss_1.item()}

        if self.global_step < 10:
            return errs

        # plot
        for i in range(min(1, self.hparams.hparams.batch_size)):
            vis_img = tgt_img[i].cpu()  # (3, H, W)
            vis_depth = visualize_depth(tgt_depth[i, 0])  # (3, H, W)

            stack = torch.cat([vis_img, vis_depth], dim=1).unsqueeze(0)  # (1, 3, 2*H, W)

            self.logger.experiment.add_images(
                'val/img_depth/{}/{}'.format(self.current_epoch, i), stack, batch_idx)

        return errs

    def on_validation_epoch_end(self):
        # 在PyTorch Lightning 2.0+中，验证输出可以通过回调指标获取
        # 这里我们简化处理，直接记录验证损失
        val_loss = self.trainer.callback_metrics.get('val_loss', 0.0)
        self.log('val_loss', val_loss, prog_bar=True)
