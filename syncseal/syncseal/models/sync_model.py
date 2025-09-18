# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Test with:
    python -m syncseal.models.syncmodel
"""

import torch
from torch import nn
from torch.nn import functional as F

from ..augmentation.augmenter import Augmenter
from ..augmentation.geometricunified import GeometricAugmenter
from ..data.transforms import RGB2YUV
from ..modules.jnd import JND
from .blender import Blender
from .embedder import Embedder
from .extractor import Extractor
try:
    from .wam_sync import SyncManager
except:
    import inspect
    frame = inspect.currentframe()
    print(f"File: {frame.f_code.co_filename}, line: {frame.f_lineno}")
    print("Warning: WAMSyncModel not available, please read the instructions in README.md to use it.")

import os
import numpy as np
import cv2

class SyncModel(nn.Module):

    @property
    def device(self):
        """Return the device of the model."""
        return next(self.parameters()).device

    def __init__(
        self,
        embedder: Embedder,
        detector: Extractor,
        geom_augmenter: GeometricAugmenter,
        valuem_augmenter: Augmenter,
        attenuation: JND = None,
        scaling_w: float = 1.0,
        scaling_i: float = 1.0,
        clamp: bool = True,
        rounding: bool = True,
        img_size: int = 256
    ) -> None:
        """
        WAM (watermark-anything models) model that combines an embedder, a detector, and an augmenter.
        Embeds a message into an image and detects it as a mask.

        Arguments:
            embedder: The watermark embedder
            detector: The watermark detector
            augmenter: The image augmenter
            attenuation: The JND model to attenuate the watermark distortion
            scaling_w: The scaling factor for the watermark
            scaling_i: The scaling factor for the image
            img_size: The size at which the images are processed
            clamp: Whether to clamp the output images to [0, 1]
            rounding: Whether to apply rounding to the output images
        """
        super().__init__()
        # modules
        self.embedder = embedder
        self.detector = detector
        self.geom_augmenter = geom_augmenter
        self.valuem_augmenter = valuem_augmenter
        # image format
        self.img_size = img_size
        self.rgb2yuv = RGB2YUV()
        # blending
        self.blender = Blender(scaling_i, scaling_w)
        self.attenuation = attenuation
        self.clamp = clamp
        self.rounding = rounding

    def forward(
        self,
        imgs: torch.Tensor,
        masks: torch.Tensor,
        interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": True},
        per_img_aug: bool = False,
    ) -> dict:
        """
        Does the full forward pass of the model (used for training).
        (1) Generates watermarked images from the input images.
        (2) Augments the watermarked images.
        (3) Detects the watermark in the augmented images and augmentation parameters.
        """
        # interpolate
        imgs_res = imgs.clone()
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs_res = F.interpolate(imgs, size=(self.img_size, self.img_size),
                                     **interpolation)

        # generate watermarked images
        if self.embedder.yuv:  # take y channel only
            preds_w = self.embedder(self.rgb2yuv(imgs_res)[:, 0:1])
        else:
            preds_w = self.embedder(imgs_res)

        # interpolate back
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            preds_w = F.interpolate(preds_w, size=imgs.shape[-2:],
                                    **interpolation)
        preds_w = preds_w.to(imgs.device)
        imgs_w = self.blender(imgs, preds_w)

        masks = masks.to(imgs.device)
        imgs_w = imgs_w * masks + imgs * (1 - masks)

        # apply attenuation and clamp
        if self.attenuation is not None:
            self.attenuation.to(imgs.device)
            imgs_w = self.attenuation(imgs, imgs_w)
        if self.clamp:
            imgs_w = torch.clamp(imgs_w, 0, 1)
        if self.rounding:
            imgs_w = (torch.round(imgs_w * 255) / 255 - imgs_w).detach() + imgs_w

        # augment
        if per_img_aug:
            # Per-image augmentation: each image gets a different augmentation
            bsz = imgs_w.shape[0]
            imgs_aug_list = []
            masks_augmented_list = []
            target_pts_list = []
            for ii in range(bsz):
                img_w_i = imgs_w[ii:ii+1]
                img_i = imgs[ii:ii+1]
                mask_i = masks[ii:ii+1]
                img_aug_i, mask_aug_i, aug_selected_i = self.valuem_augmenter(img_w_i, img_i, mask_i, do_resize=False)
                img_aug_i, mask_aug_i, geom_info_i = self.geom_augmenter(img_aug_i, mask_aug_i)
                imgs_aug_list.append(img_aug_i)
                masks_augmented_list.append(mask_aug_i)
                pts_i = geom_info_i["startpoints"]
                pts_i = torch.tensor(pts_i, device=self.device).flatten()
                pts_i = (pts_i - self.img_size / 2) / (self.img_size / 2)  # normalize to [-1, 1]
                target_pts_list.append(pts_i)
            imgs_aug = torch.cat(imgs_aug_list, dim=0)
            masks_augmented = torch.cat(masks_augmented_list, dim=0)
            target_pts = torch.stack(target_pts_list, dim=0)  # shape (bsz, 8)
        else:
            # Batch augmentation: same augmentation for all images
            imgs_aug, masks_augmented, aug_selected = self.valuem_augmenter(imgs_w, imgs, masks, do_resize=False)
            imgs_aug, masks_augmented, geom_info = self.geom_augmenter(imgs_aug, masks_augmented)
            target_pts = geom_info["startpoints"]
            target_pts = torch.tensor(target_pts, device=self.device).flatten()
            target_pts = (target_pts - self.img_size / 2) / (self.img_size / 2)  # normalize to [-1, 1]
            target_pts = target_pts.unsqueeze(0).repeat(imgs_aug.shape[0], 1)  # shape (bsz, 8)            

        # interpolate back
        if imgs_aug.shape[-2:] != (self.img_size, self.img_size):
            imgs_aug = F.interpolate(imgs_aug, size=(self.img_size, self.img_size),
                                        **interpolation)
            
        # detect watermark
        preds = self.detector(imgs_aug)

        # create and return outputs
        outputs = {
            "preds_w": preds_w,  # predicted watermarks distortions: b c h w
            "imgs_w": imgs_w,  # watermarked images: b c h w
            "imgs_aug": imgs_aug,  # augmented images: b c h w
            "preds": preds,  # predicted outputs: b (1+nparams)
            "target_pts": target_pts,  # target points for the augmentation: 8
        }
        return outputs

    def embed(
        self,
        imgs: torch.Tensor,
        interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": True},
        lowres_attenuation: bool = False,
    ) -> dict:
        """
        Generates watermarked images from the input images and messages (used for inference).
        Images may be arbitrarily sized.
        Args:
            imgs (torch.Tensor): Batched images with shape BxCxHxW.
            interpolation (dict): Interpolation parameters.
            lowres_attenuation (bool): Whether to attenuate the watermark at low resolution,
                which is more memory efficient for high-resolution images.
        Returns:
            dict: A dictionary with the following keys:
                - preds_w (torch.Tensor): Predicted watermarks with shape BxCxHxW.
                - imgs_w (torch.Tensor): Watermarked images with shape BxCxHxW.
        """
        # interpolate
        imgs_res = imgs.clone()
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs_res = F.interpolate(imgs, size=(self.img_size, self.img_size),
                                     **interpolation)
        imgs_res = imgs_res.to(self.device)

        # generate watermarked images
        if self.embedder.yuv:  # take y channel only
            preds_w = self.embedder(
                self.rgb2yuv(imgs_res)[:, 0:1]
            )
        else:
            preds_w = self.embedder(imgs_res)

        # attenuate at low resolution if needed
        if self.attenuation is not None and lowres_attenuation:
            self.attenuation.to(imgs_res.device)
            hmaps = self.attenuation.heatmaps(imgs_res)
            preds_w = hmaps * preds_w

        # interpolate back
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            preds_w = F.interpolate(preds_w, size=imgs.shape[-2:],
                                    **interpolation)
        preds_w = preds_w.to(imgs.device)
        
        # apply attenuation
        if self.attenuation is not None and not lowres_attenuation:
            self.attenuation.to(imgs.device)
            hmaps = self.attenuation.heatmaps(imgs)
            preds_w = hmaps * preds_w

        # blend and clamp
        imgs_w = self.blender(imgs, preds_w)
        if self.clamp:
            imgs_w = torch.clamp(imgs_w, 0, 1)
        if self.rounding:
            imgs_w = (torch.round(imgs_w * 255) / 255 - imgs_w).detach() + imgs_w

        outputs = {
            "preds_w": preds_w,  # predicted watermarks: b c h w
            "imgs_w": imgs_w,  # watermarked images: b c h w
        }
        return outputs

    def detect(
        self,
        imgs: torch.Tensor,
        interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": True},
    ) -> dict:
        """
        Performs the forward pass of the detector only (used at inference).
        Rescales the input images to 256x256 pixels and then computes the mask and the message.
        Args:
            imgs (torch.Tensor): Batched images with shape BxCxHxW.
        Returns:
            dict: A dictionary with the following keys:
                - preds (torch.Tensor): Predicted masks and/or messages with shape Bx(1+nparams)xHxW.
        """
        
        # interpolate
        imgs_res = imgs.clone()
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs_res = F.interpolate(imgs, size=(self.img_size, self.img_size),
                                        **interpolation)
        imgs_res = imgs_res.to(self.device)

        # detect watermark
        preds = self.detector(imgs_res).to(imgs.device)

        outputs = {
            "preds": preds,  # predicted masks and/or messages: b (1+nparams)
        }
        return outputs


class SIFTSyncModel(nn.Module):
    """
    Baseline model using SIFT+Lowe for sync point detection.
    The embed function is a no-op, and detect uses SIFT+Lowe to estimate the 4 corners.
    """

    def __init__(self, img_size=256, device="cpu"):
        super().__init__()
        self.img_size = img_size
        self.device = device

    def to(self, device):
        self.device = device
        return self

    @property
    def device(self):
        return self._device if hasattr(self, "_device") else "cpu"

    @device.setter
    def device(self, value):
        self._device = value

    def embed(self, imgs, **kwargs):
        # Return input image and zeros for preds_w
        imgs = imgs.to(self.device)
        preds_w = torch.zeros_like(imgs)
        return {"imgs_w": imgs, "preds_w": preds_w}

    def detect(self, imgs_transformed, imgs_original, **kwargs):
        # imgs_transformed: BxCxHxW, imgs_original: BxCxHxW, expects B=1
        imgs_transformed = imgs_transformed.to(self.device)
        imgs_original = imgs_original.to(self.device)
        B, C, H, W = imgs_original.shape
        assert B == 1, "SIFTSyncModel only supports batch size 1"
        img_t = imgs_transformed[0].detach().cpu()
        img_o = imgs_original[0].detach().cpu()
        # Convert to numpy
        def to_np(img):
            if img.shape[0] == 1:
                arr = img.squeeze().numpy()
                arr = np.stack([arr]*3, axis=-1)
            else:
                arr = img.permute(1, 2, 0).numpy()
            arr = (arr * 255).astype(np.uint8)
            return arr
        img_np_t = to_np(img_t)
        img_np_o = to_np(img_o)
        gray_t = cv2.cvtColor(img_np_t, cv2.COLOR_RGB2GRAY)
        gray_o = cv2.cvtColor(img_np_o, cv2.COLOR_RGB2GRAY)

        # SIFT keypoints and descriptors
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray_o, None)
        kp2, des2 = sift.detectAndCompute(gray_t, None)

        # Match descriptors using BFMatcher + Lowe's ratio test
        bf = cv2.BFMatcher()
        try:
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)
        except:
            good_matches = []
            print("SIFT matching failed, no matches found")
            print(f"des1: {des1}, des2: {des2}")

        # If not enough matches, fallback to corners
        if len(good_matches) < 4:
            corners_aug = np.float32([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]]).reshape(-1, 2)
            detected_points = corners_aug
        else:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H_mat, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            corners_aug = np.float32([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]]).reshape(-1, 1, 2)
            corners_orig = cv2.perspectiveTransform(corners_aug, H_mat) if H_mat is not None else corners_aug
            detected_points = corners_orig.reshape(-1, 2)

        # Normalize to [-1, 1] as in SyncModel
        norm_pts = (detected_points - np.array([(W-1)/2, (H-1)/2])) / ((W-1)/2)
        norm_pts = norm_pts.flatten()
        torch_pts = torch.tensor(norm_pts, dtype=torch.float32, device=imgs_transformed.device).unsqueeze(0)  # shape (1, 8)

        # Output dict as in SyncModel
        return {"preds": torch.cat([torch.zeros((1,1), device=imgs_transformed.device), torch_pts], dim=1)}


class WAMSyncModel(nn.Module):
    """
    Wrapper for WAM watermark-anything model using SyncManager.
    """

    def __init__(self, img_size=256, device="cpu"):
        model_path = "checkpoints/wam_mit.pth"  # path to WAM model checkpoint
        if not os.path.exists(model_path):
            import inspect
            frame = inspect.currentframe()
            print(f"File: {frame.f_code.co_filename}, line: {frame.f_lineno}")
            print(f"Warning: Baseline with WAM not available.")
            print("Please download WAM with `wget https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth -P checkpoints/`")
            raise FileNotFoundError(f"WAM model checkpoint not found at {model_path}")

        super().__init__()
        self.img_size = img_size
        self.device = device
        self.sync_manager = SyncManager(model_path, device)

    def to(self, device):
        self.device = device
        self.sync_manager.device = device
        return self

    def embed(self, imgs, **kwargs):
        # imgs: BxCxHxW, expects [-1, 1] range, returns dict with imgs_w and preds_w
        imgs = imgs.to(self.device)
        imgs_w = self.sync_manager.add_wam(imgs)
        preds_w = torch.zeros_like(imgs)  
        return {"imgs_w": imgs_w, "preds_w": preds_w}

    def detect(self, imgs, **kwargs):
        # imgs: BxCxHxW, expects [-1, 1] range, returns dict with "preds" (B, 9)
        imgs = imgs.to(self.device)
        B, C, H, W = imgs.shape
        # resize to 256x256 if needed
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs = F.interpolate(imgs, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        imgs_wam, aug_info, wam_info = self.sync_manager.remove_wam(imgs, return_info=True)
        angle, cuti, cutj, is_flipped = aug_info
        cuti = min(max(cuti,0), (self.img_size-1))
        cutj = min(max(cutj,0), (self.img_size-1))
        crop_applied = (cuti != (self.img_size-1) // 2 or cutj != (self.img_size-1) // 2) and not is_flipped

        # Start with full image corners
        corners = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype=np.float32)

        # Apply crop if detected (centered at cuti, cutj)
        cuti = int((H-1) * cuti / (self.img_size-1))
        cutj = int((W-1) * cutj / (self.img_size-1))
        if crop_applied:
            top = 0
            left = 0
            pad_i = 2 * cuti - (H-1)
            pad_j = 2 * cutj - (W-1)
            bottom = (H-1) - pad_i
            right = (W-1) - pad_j
            corners = np.array([
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom]
            ], dtype=np.float32)

        # Apply rotation if detected
        if abs(angle) > 1e-2:
            # Rotate around image center
            center = np.array([W/2, H/2])
            theta = -np.deg2rad(angle)  # negative for inverse
            rot_mat = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            corners = np.dot(corners - center, rot_mat.T) + center

        # Apply horizontal flip if detected
        if is_flipped:
            corners[:, 0] = W - 1 - corners[:, 0]

        # Normalize to [-1, 1]
        norm_pts = (corners - np.array([W/2, H/2])) / np.array([W/2, H/2])
        norm_pts = norm_pts.flatten()
        norm_pts = torch.tensor(norm_pts, dtype=torch.float32, device=imgs.device).unsqueeze(0)  # (1, 8)
        # Output dict as in SyncModel
        return {"preds": torch.cat([torch.zeros((1,1), device=imgs.device), norm_pts], dim=1)}
