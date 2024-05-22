import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if torch.is_tensor(depth):
        x = depth.cpu().numpy()
    else:
        x = depth
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8)  # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_

def visualize_k(k, cmap="viridis", is_image=False):
    if torch.torch.is_tensor(k):
        x = k.detach().cpu().numpy()
    else:
        x = k.detach()
    x = np.nan_to_num(x)  # change nan to 0
    x[x < 1e-6] = 1.0
    x = np.abs(x-1)
    vmax = np.percentile(x, 95)
    x[x > vmax] = vmax
    x = x/vmax
    x = (255*x).astype(np.uint8)
    mapper = cm.ScalarMappable(cmap=cmap) # colormap
    x_ = mapper.to_rgba(x)[:, :, :3]
    if is_image:
        return x_
    x_ = torch.from_numpy(x_).float().permute(2, 0, 1)
    return x_