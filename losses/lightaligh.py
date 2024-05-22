import torch
import torch.nn as nn
import torch.nn.functional as F

from visualization import visualize_k

from .inverse_warp import depth2world

def mask_or(mask1, mask2):
    return torch.logical_or(mask1, mask2).to(torch.float32)

class Depth2Normal(nn.Module):
    """ Layer to convert depth image to noraml image
    """
    def __init__(self, batch_size, height, width) -> None:
        super(Depth2Normal, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width

        vectors = [torch.arange(0, s) for s in [height, width]]
        grid_y, grid_x = torch.meshgrid(vectors)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).float()
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).float()

        self.grid_x = nn.Parameter(grid_x, requires_grad=False)
        self.grid_y = nn.Parameter(grid_y, requires_grad=False)
    
    def getNormalFrom3D(self, points):
        p = points
        eye = -p

        # vector of surface normal in west, south, east, north direction
        p_cur = p[:, :, 1:-1, 1:-1]
        vw = p_cur - p[:, :, 1:-1, 2:]
        vs = p[:, :, 2:, 1:-1] - p_cur
        ve = p_cur - p[:, :, 1:-1, :-2]
        vn = p[:, :, :-2, 1:-1] - p_cur

        normal_1 = torch.cross(vw, vs, dim=1)
        normal_2 = torch.cross(ve, vn, dim=1)
        # normal_1 = F.normalize(normal_1, p=2, dim=1)
        # normal_2 = F.normalize(normal_2)

        # do add then normalize here, because of the weighted method
        # more information https://github.com/Microsoft/DirectXMesh/wiki/ComputeNormals
        normal = normal_1 + normal_2
        normal = F.normalize(normal, p=2, dim=1)
        paddings = (1, 1, 1, 1)
        normal = F.pad(normal, paddings, 'constant')
        
        mask = (torch.sum(eye * normal, dim=1, keepdim=True) < 0)
        mask = torch.cat([mask, mask, mask], dim=1)
        normal[mask] = -normal[mask]

        return normal  # (B, 3, H, W)        

    
    def forward(self, depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        p = depth2world(depth, K)
        return self.getNormalFrom3D(p)


def cal_highlight_loss(points_world, normals):
    p3d = points_world
    ni = normals # B, 3, H, W
    eye = -F.normalize(p3d, p=2, dim=1)
    theta = torch.sum(eye*ni, dim=1, keepdim=True)

    highlight_loss = (1 - theta) ** 2
    
    return highlight_loss

class LightAlign(nn.Module):
    """
    Layer of aligning the lightness between two images
    Inputs:
        src image, target image and corresponding depth map
    Outputs:
        refined image
    """
    def __init__(self, batch_size, height, width, mu, gamma) -> None:
        super(LightAlign, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        self.depth2normal = Depth2Normal(batch_size, height, width)
        # used in SLS
        self.mu = mu
        self.gamma = gamma
        # light vector [0, 0, -1]
        self.light_vec_ = torch.zeros((batch_size, 3, height, width))
        self.light_vec_[:, 2, :, :] = -1
        self.light_vec_ = nn.Parameter(self.light_vec_, requires_grad=False)

        vectors = [torch.arange(0, s) for s in [height, width]]
        grid_y, grid_x = torch.meshgrid(vectors)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).float()
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).float()

        grid_x = grid_x.repeat(self.batch_size, 1, 1, 1)
        grid_y = grid_y.repeat(self.batch_size, 1, 1, 1)

        self.grid_x = nn.Parameter(grid_x, requires_grad=False)
        self.grid_y = nn.Parameter(grid_y, requires_grad=False)

    def getNoraml(self, depth, K):
        normal = self.depth2normal(depth, K)
        return normal
    
    def toNormalCoord(self, x_coord: torch.tensor, y_coord: torch.tensor, K):
        """
        x_coord: B*3*H*W
        """
        # B * 1
        B, _, H, W = x_coord.size()
        fx, fy, cx, cy = (K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2])
        print(fx.reshape(B,1).shape)
        x = (x_coord.view(B, -1) - cx.view(B,1)) / fx.view(B,1)
        y = (y_coord.view(B, -1) - cy.view(B,1)) / fy.view(B,1)

        x_ = x.view(B, 1, H, W)
        y_ = y.view(B, 1, H, W)
        z = torch.ones_like(x_)

        p = torch.cat([x_, y_, z], dim=1)
        return p

    def forward(self, ref_img, tgt_img, tgt_depth,tgt_highlight_mask, ref_highlight_mask, ref_depth, T, K, padding_mode="zeros") -> torch.Tensor:
        B, _, H, W = ref_img.size()
        eps = 1e-7

        # target photometry
        point_tar = depth2world(tgt_depth, K)
        p2c_tar_ = F.normalize(-point_tar, p=2, dim=1, eps=eps)
        normal_tar_ = self.depth2normal(tgt_depth, K)
        theta_tar = torch.sum(p2c_tar_ * normal_tar_, dim=1, keepdim=True)
        theta_tar[theta_tar < eps] = eps
        r2_tar = torch.sum(point_tar**2, dim=1, keepdim=True)
        phi_tar = torch.sum(self.light_vec_ * p2c_tar_, dim=1, keepdim=True)
        R_tar = torch.exp(-self.mu * (1 - phi_tar)) * theta_tar / (r2_tar + eps)
        R_tar[R_tar < eps] = eps

        highlight_loss = cal_highlight_loss(point_tar, normal_tar_)

        # src photometry
        # depth and normal in src view generated from target
        world_points = torch.cat([point_tar, torch.ones(B,1,H,W).type_as(point_tar)], 1)
        point_src = torch.matmul(T[:, :3, :], world_points.view(B, 4, -1))
        point_src = point_src.view(B, 3, H, W)
        p2c_src_ = F.normalize(-point_src, p=2, dim=1, eps=eps)
        R = T[:, :3, :3]
        normal_src_ = torch.matmul(R, normal_tar_.view(B, 3, -1))
        normal_src_ = normal_src_.view(B, 3, H, W)
        theta_src = torch.sum(p2c_src_ * normal_src_, dim=1, keepdim=True)
        theta_src[theta_src < eps] = eps
        r2_src = torch.sum(point_src**2, dim=1, keepdim=True)
        phi_src = torch.sum(self.light_vec_ * p2c_src_, dim=1, keepdim=True)
        R_src = torch.exp(-self.mu * (1 - phi_src)) * theta_src / (r2_src + eps)
        R_src[R_src < eps] = eps

        # warp
        # B * H * W * 2
        cam_points = torch.matmul(K, point_src.view(B, 3, -1))
        computed_depth = cam_points[:, 2, :].unsqueeze(1).view(B, 1, H, W)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + eps)
        pix_coords = pix_coords.view(B, 2, H, W)

        coord_X = pix_coords[:, 0, :, :]
        coord_Y = pix_coords[:, 1, :, :]

        valid_mask = torch.ones_like(coord_X)
        valid_mask[coord_X < 1] = 0
        valid_mask[coord_X > W - 2] = 0
        valid_mask[coord_Y < 1] = 0
        valid_mask[coord_Y > H - 2] = 0
        valid_mask = valid_mask.unsqueeze(dim=1)

        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= W - 1
        pix_coords[..., 1] /= H - 1
        pix_coords = (pix_coords - 0.5) * 2

        src_sample = F.grid_sample(ref_img, pix_coords, padding_mode=padding_mode, align_corners=False)
        projected_mask = F.grid_sample(ref_highlight_mask, pix_coords, padding_mode=padding_mode, align_corners=False)
        projected_depth = F.grid_sample(ref_depth, pix_coords, padding_mode=padding_mode, align_corners=False)
        
        
        # calculate ratio of photometry
        highlight_mask = mask_or(projected_mask, tgt_highlight_mask)

        delt_min = 0.2
        mask1 = theta_tar < delt_min
        mask2 = theta_src < delt_min

        kR = R_tar / (R_src + eps)
        mask3 = mask1 | mask2 

        # The photometric ratio calculation at mask3 is unstable, typically occurring at the edge of the target
        kR[mask3] = 1
        
        unvalid_mask = (1.0 - valid_mask).to(torch.bool)
        src_sample = src_sample * valid_mask
        unrefined_mask = mask3 | unvalid_mask | highlight_mask.bool()

        img_src_lambda = torch.pow(src_sample, self.gamma)
        img_tar_lambda = torch.pow(tgt_img, self.gamma)
        img_src_lambda = img_src_lambda * (~unrefined_mask).type(torch.float32)
        img_tar_lambda = img_tar_lambda * (~unrefined_mask).type(torch.float32)

        src_image_k1 = img_src_lambda * kR
        sum_src = torch.sum(src_image_k1.view(B, -1), dim=1, keepdim=True)
        sum_tar = torch.sum(img_tar_lambda.contiguous().view(B, -1), dim=1, keepdim=True)
        k_g = sum_tar / (sum_src + eps)
        k_g = k_g.detach()

        k = (kR * (k_g).view(B, 1, 1, 1))
        k = torch.pow(k + eps, 1.0/self.gamma)

        k[unvalid_mask] = 0
        k = k.detach()

        refined = k * src_sample
        refined = refined.clamp(0, 1)

        return refined, src_sample, projected_depth, projected_mask, computed_depth, R_tar, k, highlight_loss