# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from loguru import logger
from scipy import ndimage

from deps.watermark_anything.augmentation.geometric import HorizontalFlip, Rotate
from deps.watermark_anything.utils.inference_utils import (
    load_model_from_checkpoint,
    normalize_img,
    unnormalize_img,
)

logger.remove()


class SyncManager:
    def __init__(self, wampath, device):
        self.device = device

        # NOTE: Make sure to download the model to wampath
        json_path = os.path.join("deps", "watermark_anything", "params.json")
        ckpt_path = os.path.join(wampath)
        self.wam = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()
        self.epsilon = 1
        self.min_samples = 500
        self.wm_msgs = torch.tensor(
            [
                [0 for _ in range(32)],
                [0 for _ in range(16)] + [1 for _ in range(16)],
                [1 for _ in range(16)] + [0 for _ in range(16)],
                [1 for _ in range(32)],
            ]
        ).to(device)
        self.nb_msgs = 4
        # Define a color map for each unique value for multiple wm viz
        self.color_map = {
            -1: [0, 0, 0],  # Black for -1
            0: [255, 0, 255],  # Magenta for 0
            1: [255, 0, 0],  # Red for 1
            2: [0, 255, 0],  # Green for 2
            3: [0, 0, 255],  # Blue for 3
            4: [255, 255, 0],  # Yellow for 4
            5: [255, 165, 0],  # Orange for 5
            6: [128, 0, 128],  # Purple for 6
            7: [128, 128, 0],  # Olive for 7
            8: [0, 128, 128],  # Teal for 8
        }

    # transfer to WAM space, [-1, 1] -> [0, 1] + normalized
    def normalize(self, imgs):
        return normalize_img((imgs + 1.0) / 2.0)

    # transfer from WAM space, [0, 1] + normalized -> [-1, 1]
    def unnormalize(self, imgs):
        imgs = unnormalize_img(imgs) * 2.0 - 1.0
        return imgs.clamp(-1, 1)

    # imgs: [b, 3, 256, 256] in [-1, 1] -> return same
    def add_wam(self, imgs, return_masks=False):
        orig_device = imgs.device
        imgs = self.normalize(imgs).to(self.device)
        masks = self.create_grid_mask(imgs[-1], num_masks=len(self.wm_msgs))
        multi_wm_imgs = []
        for img in imgs:
            multi_wm_img = img.clone()
            for ii in range(len(self.wm_msgs)):
                wm_msg, mask = self.wm_msgs[ii].unsqueeze(0), masks[ii]
                outputs = self.wam.embed(img.unsqueeze(0), wm_msg)
                multi_wm_img = outputs["imgs_w"] * mask + multi_wm_img * (1 - mask)  # [1, 3, H, W]
            multi_wm_imgs.append(multi_wm_img)
        multi_wm_imgs = torch.cat(multi_wm_imgs, dim=0)
        ret = self.unnormalize(multi_wm_imgs).to(orig_device)
        if return_masks:
            return ret, masks
        else:
            return ret

    # imgs: [b, 3, 256, 256] in [-1, 1] -> return same
    def remove_wam(self, imgs, return_info=False):
        orig_device = imgs.device
        imgs = self.normalize(imgs).to(self.device)
        preds = self.wam.detect(imgs)["preds"]  # [B, 33, 256, 256]
        reverteds = []
        for i in range(imgs.shape[0]):
            aug_info, wam_info = self.estimate_augmentation_with_wam(
                imgs[i].unsqueeze(0), self.wm_msgs, preds[i].unsqueeze(0), self.epsilon, self.min_samples, idx=i
            )
            reverted = self.revert_augmentation(imgs[i].unsqueeze(0), aug_info).detach()
            reverteds.append(self.unnormalize(reverted).to(orig_device))

            angle, cuti, cutj, is_flipped = aug_info
            mask_preds_res, positions, centroids = wam_info
            logger.info(
                f"{i}: {len(centroids)} messages found | Rot: {angle}, cuts: {cuti}, {cutj} (flip={is_flipped})"
            )

        if return_info:
            return torch.cat(reverteds, dim=0), aug_info, wam_info
        else:
            return torch.cat(reverteds, dim=0)

    def create_grid_mask(self, img_pt, num_masks):
        masks = torch.zeros((num_masks, 1, img_pt.shape[-2], img_pt.shape[-1]))
        sqrt_num_masks = int(np.sqrt(self.nb_msgs))
        sqrt_img_size = img_pt.shape[-1] // sqrt_num_masks
        for i in range(sqrt_num_masks):
            for j in range(sqrt_num_masks):
                # take a square of size sqrt_num_masks x sqrt_num_mask in i-th row and j-th col
                masks[
                    i * sqrt_num_masks + j,
                    0,
                    i * sqrt_img_size : (i + 1) * sqrt_img_size,
                    j * sqrt_img_size : (j + 1) * sqrt_img_size,
                ] = 1

        # With a buffer in the middle there's some leeway
        midpoint = img_pt.shape[-1] // 2
        SZ = img_pt.shape[-1]
        leeway_width = 18 if SZ == 256 else 36
        start = midpoint - leeway_width // 2
        end = midpoint + leeway_width // 2 + 1
        for i in range(start, end):
            masks[:, :, :, start:end] = 0
            masks[:, :, start:end, :] = 0
        return masks.to(img_pt.device)

    def rotate_wm(self, wm, rotation):
        res = np.zeros_like(wm)
        # Rotate each ID separately to avoid interpolation problems
        for i in range(1, self.nb_msgs + 1):
            mask = (wm == i) * 255
            mask_rot = ndimage.rotate(mask, rotation, reshape=False)
            res[mask_rot >= 0.5] = i
        return res

    def find_cut(self, cumsums, pairs, dim, SZ):
        error = 0
        cut = 0
        cut_weight = 0
        is_flipped = 0
        for l, r in pairs:
            errors_normal = cumsums[dim][r] + (cumsums[dim][l][-1] - cumsums[dim][l])
            minerror_normal = np.min(errors_normal)
            minerror_indices_normal = np.where(errors_normal == minerror_normal)[0]
            score_normal = minerror_normal - len(minerror_indices_normal) * 1e-3

            errors_flipped = cumsums[dim][l] + (cumsums[dim][r][-1] - cumsums[dim][r])
            minerror_flipped = np.min(errors_flipped)
            minerror_indices_flipped = np.where(errors_flipped == minerror_flipped)[0]
            score_flipped = minerror_flipped - len(minerror_indices_flipped) * 1e-3

            # First we decide if Hflipped based on min error but also size of valid range
            if score_normal < score_flipped or dim == 1:
                curr_is_flipped = False
                errors = errors_normal
                minerror_indices = minerror_indices_normal
                is_flipped -= 1
            else:
                is_flipped += 1
                curr_is_flipped = True
                errors = errors_flipped
                # There's a shift in indices for hflip apparently. 
                # No idea where it comes from, but adding 2 fixes it
                minerror_indices = [el for el in minerror_indices_flipped] 

            if cumsums[dim][r][-1] != 0 and cumsums[dim][l][-1] == 0:
                # only R exists
                idx = 0 if (curr_is_flipped == 1) else -1
                idx_minerror = minerror_indices[idx]
            elif cumsums[dim][l][-1] != 0 and cumsums[dim][r][-1] == 0:
                # only L exists
                idx = -1 if (curr_is_flipped == 1) else 0
                idx_minerror = minerror_indices[idx]
            else:
                # both exist, take the middle of the valid range
                idx_minerror = (minerror_indices[0] + minerror_indices[-1]) // 2

            # Each pair is as worth as its avg component size
            w = cumsums[dim][l][-1] + cumsums[dim][r][-1]
            error += errors[idx_minerror] * w
            cut += idx_minerror * w
            cut_weight += w
        
        # If there was absolutely no signal for a cut return the default cut
        if cut_weight == 0:
            return 1e9, SZ // 2, False

        cut = round(cut / cut_weight)
        is_flipped = ((is_flipped / cut_weight) > 0).item()
        error /= cut_weight

        # Recompute error for the average
        error = 0
        for l, r in pairs:
            if not is_flipped:
                errors = cumsums[dim][r] + (cumsums[dim][l][-1] - cumsums[dim][l])
            else:
                errors = cumsums[dim][l] + (cumsums[dim][r][-1] - cumsums[dim][r])
            error += errors[cut]

        return error, cut, is_flipped

    def fit_best_aug(self, positions):
        SZ = positions.shape[-1]
        # fix the image
        # NOTE: inside this function labels are 1 2 3 4 for visualization
        wm = np.zeros_like(positions)
        for i in [4, 3, 2, 1, 0]:
            wm[positions == i - 1] = i
        min_errors = float("inf")
        best_rotations = [0]
        best_cuti = SZ // 2
        best_cutj = SZ // 2
        best_isflipped = False
        # for rotation in [0]:
        for rotation in np.arange(-30, 31, 1):
            # use PIL to rotate
            wm_rot = self.rotate_wm(wm, rotation)

            # show discrete colors 0 1 2 3 4 with discrete fixed color for each
            # find min number of errors vertically (01 and 23) and horizontally (02 and 13)
            THRESH = 40 if SZ == 256 else 80
            cumsums = [[None], [None]]
            for dim in range(2):
                for i in range(1, 5):
                    sums = np.sum(wm_rot == i, axis=dim)
                    sums[sums < THRESH] = 0  # naive, better to dilate
                    cumsums[dim].append(np.cumsum(sums))

            errori, cuti, _ = self.find_cut(cumsums, [(1, 3), (2, 4)], dim=1, SZ=SZ)
            errorj, cutj, is_flipped = self.find_cut(cumsums, [(1, 2), (3, 4)], dim=0, SZ=SZ)
            if errori + errorj < min_errors:
                min_errors = errori + errorj
                best_rotations = [rotation]
                best_cuti = cuti
                best_cutj = cutj
                best_isflipped = is_flipped
            elif errori + errorj == min_errors:
                best_rotations.append(rotation)
        best_rotation = round((np.max(best_rotations) + np.min(best_rotations)) / 2)
        return (best_rotation, best_cuti, best_cutj, best_isflipped)

    def estimate_augmentation_with_wam(self, multi_wm_img_aug, wm_msgs, wam_preds, epsilon, min_samples, idx):
        origH, origW = multi_wm_img_aug.shape[-2], multi_wm_img_aug.shape[-1]
        mask_preds = F.sigmoid(wam_preds[:, 0, :, :]).unsqueeze(1)  # [1, 1, 256, 256], predicted mask
        bit_preds = wam_preds[:, 1:, :, :]  # [1, 32, 256, 256], predicted bits
        if mask_preds.shape[-1] != multi_wm_img_aug.shape[-1]:
            mask_preds_res = F.interpolate(
                mask_preds,
                size=(multi_wm_img_aug.shape[-2], multi_wm_img_aug.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )  # [1, 1, H, W]
            bit_preds_res = F.interpolate(
                bit_preds,
                size=(multi_wm_img_aug.shape[-2], multi_wm_img_aug.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )  # [1, 32, H, W]
        else:
            mask_preds_res = mask_preds
            bit_preds_res = bit_preds

        # Manual, no DBscan
        clip_dist = 6
        centroids = {k: msg.to(multi_wm_img_aug.device) for k, msg in enumerate(wm_msgs)}
        H, W = bit_preds_res.shape[-2], bit_preds_res.shape[-1]
        positions = torch.zeros((H, W)).to(multi_wm_img_aug.device)
        bit_preds_res_flat = (bit_preds_res[0] > 0).long().view(32, -1).transpose(0, 1)  # [256*256, 32]
        dists = torch.round(
            torch.cdist(bit_preds_res_flat.float(), wm_msgs.float(), p=1).view(H, W, 4), decimals=0
        ).long()  # [256, 256, 4]

        # For each min index in dists
        mask = mask_preds_res.squeeze(0).squeeze(0)  # [256, 256]
        min_idxs = torch.argmin(dists, dim=2)  # [256, 256]
        min_dists = torch.gather(dists, 2, min_idxs.unsqueeze(2)).squeeze(2)  # [256, 256]

        # Need to have a close one AND be in the mask, otherwise -1
        positions = ((min_dists <= clip_dist).float() * ((mask > 0.5).float()) * (min_idxs.float() + 1) - 1).long()

        # Filtering out
        sizes = [torch.sum(positions == k).item() for k in centroids.keys()]
        if centroids is None or len(centroids) < 4:
            nb_centroids = 0 if centroids is None else len(centroids)
            logger.warning(
                f"idx={idx}: Found {nb_centroids} centroids after postprocessing, expected 4, returning dummy values"
            )
            return (0, origH // 2, origW // 2, False), (mask_preds_res, positions, centroids)
        FACTOR = 0.7 if origH == 256 else 0.75
        sum_size_thresh = round((origH * origW) * FACTOR)
        #print(f"Sum sizes: {sum(sizes)} vs {sum_size_thresh}")
        if sum(sizes) < sum_size_thresh:
            logger.warning(
                f"idx={idx}: Total size is {sum(sizes)}<{sum_size_thresh}; I am not confident enough in WAM, returning dummy values"
            )
            return (0, origH // 2, origW // 2, False), (mask_preds_res, positions, centroids)

        centroids = {k: v.detach().cpu().numpy() for k, v in centroids.items()}
        positions = positions.detach().cpu().numpy()

        # Try to revert the transformation
        aug_info = self.fit_best_aug(positions)

        wam_info = (mask_preds_res, positions, centroids)
        return aug_info, wam_info

    def revert_augmentation(self, multi_wm_img_aug, aug_info):
        origH, origW = multi_wm_img_aug.shape[-2], multi_wm_img_aug.shape[-1]
        #print(f"OrigH: {origH}, OrigW: {origW}")

        angle, cuti, cutj, is_flipped = aug_info
        reverted = multi_wm_img_aug

        if is_flipped:
            reverted, _ = HorizontalFlip()(reverted, reverted)
            return reverted

        if abs(angle) >= 3:
            reverted, _ = Rotate()(multi_wm_img_aug, multi_wm_img_aug, angle)
            return reverted

        # What's left is to revert the crop
        PAD_THRESH = 10 if origH == 256 else 25
        pad_i = 2 * cuti - multi_wm_img_aug.shape[-2]
        if pad_i < PAD_THRESH:
            pad_i = 0
        pad_j = max(0, 2 * cutj - multi_wm_img_aug.shape[-1])
        if pad_j < PAD_THRESH:
            pad_j = 0
        if pad_i > 0 or pad_j > 0:
            reverted = F.pad(reverted, (0, pad_j, 0, pad_i))

        # Now resize back to (256, 256)
        reverted = TVF.resize(reverted, (origH, origW))
        return reverted
