from typing import Any
import numpy as np
import cv2
import torch

def dilateByELLIPSE(img, radius):
    disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(img, disk_kernel)

def closeByELLIPSE(img, radius):
    disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, disk_kernel)

"""Reference: https://github.com/Be997398715/matalb-remove-specular-highlights-
"""
class SpecularDetection:
    def __init__(self, T1, T2_abs, T2_rel, T3, N_min, decay_win_size = 10, decay_cof = 20) -> None:
        self.T1 = T1
        self.T2_abs = T2_abs
        self.T2_rel = T2_rel
        self.T3 = T3
        self.N_min = N_min

        # inpaintting
        self.decay_win_size = decay_win_size
        self.decay_cof = decay_cof

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.detection(*args, **kwds)

    def detection(self, img):
        if torch.torch.is_tensor(img):
            img = img.permute(1, 2, 0).squeeze().numpy()

        if img.max() <= 1 + 1e3:
            img = (img * 255).astype(np.uint8)


        cR = img[:, :, 0]
        cG = img[:, :, 1]
        cB = img[:, :, 2]
        cE = 0.2989*cR + 0.5870*cG + 0.1140*cB

        mask1 = self.module1(cE, cG, cB, self.T1)

        mask2 = self.module1(cE, cG, cB, self.T2_abs)
        fill_img = self.fillImage(mask2, img)
        mask3 = self.module2(fill_img, self.T2_rel, cR, cG, cB)

        final_mask = np.logical_or(mask1, mask3).astype(np.uint8) * 255
        final_mask = np.logical_and(final_mask, mask2).astype(np.uint8) * 255
        final_mask = dilateByELLIPSE(final_mask, 2)

        mask_highlight = self.classify(final_mask, self.N_min)
        inpaint_img = self.inpaintting(mask_highlight, img)
    
        inpaint_img = torch.from_numpy(inpaint_img/255.0).permute(2, 0, 1)

        mask_highlight = np.expand_dims(mask_highlight, axis=0)


        return {"point": torch.from_numpy(mask_highlight),
                "inpaint": inpaint_img,}

    def module1(self, cE, cG, cB, T):
        p95_cG = np.percentile(cG, 95)
        p95_cB = np.percentile(cB, 95)
        p95_cE = np.percentile(cG, 95)

        rGE = p95_cG/p95_cE
        rBE = p95_cB/p95_cE
        
        mask1 = cG > rGE*T
        mask2 = cB > rBE*T
        mask3 = cE > T

        module1_mask = np.logical_or(np.logical_or(mask1, mask2), mask3)
        # print(module1_mask)
        return module1_mask.astype(np.uint8)*255
    
    def module2(self, img, T_rel, cR, cG, cB):
        fR = cv2.medianBlur(img[:, :, 0], 31).astype(float)
        fG = cv2.medianBlur(img[:, :, 1], 31).astype(float)
        fB = cv2.medianBlur(img[: ,:, 2], 31).astype(float)

        tR = self.contrast_coeffcient(fR)
        tG = self.contrast_coeffcient(fG)
        tB = self.contrast_coeffcient(fB)

        max_img = np.stack([tR*np.divide(cR, fR), tG*np.divide(cG, fG), tB*np.divide(cB, fB)], axis=2)
        e_max = np.max(max_img, axis=2)
        module2_mask = e_max > T_rel
        
        return module2_mask.astype(np.uint8) * 255


    def fillImage(self, mask: np.ndarray, img: np.ndarray):
        img2 = np.copy(img)
        
        dilated_mask_1 = dilateByELLIPSE(mask, 2)
        dilated_mask_2 = dilateByELLIPSE(mask, 4)
        dilated_area = dilated_mask_2 - dilated_mask_1

        nums1, labels1, _, centroids1 = cv2.connectedComponentsWithStats(dilated_area)
        nums2, labels2, _, centroids2 = cv2.connectedComponentsWithStats(mask)
        mean_color = []

        for label in range(1, nums1):
            mean_color.append(np.median(img2[labels1==label], axis=0))
        
        for label in range(1, nums2):
            centroids = centroids2[label]
            nearest_idx = 1
            nearest_dist = np.linalg.norm(centroids - centroids1[nearest_idx])
            for idx in range(2, nums1):
                dist = np.linalg.norm(centroids - centroids1[idx])
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_idx = idx

            img2[labels2==label] = mean_color[nearest_idx-1]

        return img2

    def contrast_coeffcient(self, c):
        mean = np.mean(c)
        std = np.std(c)
        t = 1/((mean + std)/mean)
        return t

    def inpaintting(self, specular_mask, img):
        filled_img = self.fillImage(specular_mask.astype(np.uint8)*255, img)
        sig = 8
        gaussian_filtered_img = cv2.GaussianBlur(filled_img, (0, 0), sig)
        
        filter_kernel = np.ones((self.decay_win_size, self.decay_win_size)) / self.decay_cof
        mx = cv2.filter2D(specular_mask, -1, filter_kernel)
        mx = mx + specular_mask
        mx[mx > 1] = 1.0
        
        mx = np.stack([mx] * 3, axis=2)

        inpainted_img = mx * gaussian_filtered_img + (1 - mx) * img
        inpainted_img = cv2.medianBlur(inpainted_img.astype(np.uint8), 3).astype(float)

        return inpainted_img
    
    def classify(self, mask, N_min):
        nums, labels = cv2.connectedComponents(mask)
        mask_highlight = np.zeros_like(mask).astype(np.float32)

        for label in range(1, nums):
            N = np.sum(labels == label)
            if N < N_min:
                mask_highlight[labels == label] = 1.0

        return mask_highlight