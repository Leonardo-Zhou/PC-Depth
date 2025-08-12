import numpy as np
import torch
from pytorch_lightning import LightningModule

import losses.loss_functions as LossF
from models.DepthNet import DepthNet
from models.PoseNet import PoseNet
from visualization import *


def disp_to_depth(disp, min_depth=1, max_depth=100):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return depth

class SC_Depth(LightningModule):
    def __init__(self, hparams):
        super(SC_Depth, self).__init__()
        self.save_hyperparameters()

        # model
        self.depth_net = DepthNet(self.hparams.hparams.resnet_layers)
        self.pose_net = PoseNet()

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
        tgt_img, ref_imgs, intrinsics = batch["color"]
        tgt_img_aug, ref_imgs_aug = batch["color_aug"]

        # network forward
        tgt_depth = self.depth_net(tgt_img_aug)
        ref_depths = [self.depth_net(im) for im in ref_imgs_aug]

        poses = [self.pose_net(tgt_img_aug, im) for im in ref_imgs_aug]
        poses_inv = [self.pose_net(im, tgt_img_aug) for im in ref_imgs_aug]

        # compute loss
        w1 = self.hparams.hparams.photo_weight
        w2 = self.hparams.hparams.geometry_weight
        w3 = self.hparams.hparams.smooth_weight

        loss_1, loss_2 = LossF.photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths,
                                                       intrinsics, poses, poses_inv, self.hparams.hparams)
        loss_3 = LossF.compute_smooth_loss(tgt_depth, tgt_img)

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3

        # plot
        if batch_idx % 50 == 0:
            for i in range(min(1, self.hparams.hparams.batch_size)):
                vis_img = tgt_img_aug[i].cpu()  # (3, H, W)
                vis_depth = visualize_depth(tgt_depth[i, 0].detach())  # (3, H, W)


                stack = torch.cat([vis_img, vis_depth], dim=1).unsqueeze(0)  # (1, 3, 2*H, W)

                self.logger.experiment.add_images(
                    'train/img_depth/{}/{}'.format(self.current_epoch, i), stack, batch_idx)

        # create logs
        self.log('train/total_loss', loss)
        self.log('train/photo_loss', loss_1)
        self.log('train/geometry_loss', loss_2)
        self.log('train/smooth_loss', loss_3)

        return loss

    def validation_step(self, batch, batch_idx):
        tgt_img, ref_imgs, intrinsics = batch["color"]
        tgt_img_aug, ref_imgs_aug = batch["color_aug"]

        tgt_depth = self.depth_net(tgt_img)
        ref_depths = [self.depth_net(im) for im in ref_imgs]
        poses = [self.pose_net(tgt_img, im) for im in ref_imgs]
        poses_inv = [self.pose_net(im, tgt_img) for im in ref_imgs]

        loss_1, loss_2 = LossF.photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths,
                                                        intrinsics, poses, poses_inv, self.hparams.hparams)
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
