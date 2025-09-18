# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import random

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as maskUtils

from torchvision.datasets import CocoDetection
from torchvision.datasets.folder import is_image_file
from torchvision.transforms import ToTensor

from ..utils import suppress_output

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_image_paths(path):
    cache_dir = '.cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = path.replace('/', '_') + '.json'
    cache_file = os.path.join(cache_dir, cache_file)
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            paths = json.load(f)
    else:
        print(f"Creating cache file {cache_file} for image paths...")
        paths = []
        for root, _, files in os.walk(path):
            for filename in files:
                if is_image_file(filename):
                    paths.append(os.path.join(root, filename))
        paths = sorted(paths)
        with open(cache_file, 'w') as f:
            json.dump(paths, f)
    return paths


def create_random_mask(height, width, num_masks=1, mask_percentage=0.1, max_attempts=100):
    mask_area = int(height * width * mask_percentage)
    masks = torch.zeros((num_masks, 1, height, width), dtype=torch.float32)

    if mask_percentage >= 0.999:
        # Full mask for entire image
        return torch.ones((num_masks, 1, height, width), dtype=torch.float32)

    for ii in range(num_masks):
        placed = False
        attempts = 0
        while not placed and attempts < max_attempts:
            attempts += 1

            max_dim = int(mask_area ** 0.5)
            mask_width = random.randint(1, max_dim)
            mask_height = mask_area // mask_width
            if random.random() < 0.5:  # 50% chance to allow overlap
                mask_height, mask_width = mask_width, mask_height

            # Allow broader aspect ratios for larger masks
            aspect_ratio = mask_width / mask_height if mask_height != 0 else 0
            if 0.25 <= aspect_ratio <= 4:  # Looser ratio constraint
                if mask_height <= height and mask_width <= width:
                    x_start = random.randint(0, width - mask_width)
                    y_start = random.randint(0, height - mask_height)
                    masks[ii, :, y_start:y_start + mask_height, x_start:x_start + mask_width] = 1
                    placed = True

        if not placed:
            # Fallback: just fill a central region if all attempts fail
            print(f"Warning: Failed to place mask {ii}, using fallback.")
            center_h = height // 2
            center_w = width // 2
            half_area = int((mask_area // 2) ** 0.5)
            h_half = min(center_h, half_area)
            w_half = min(center_w, half_area)
            masks[ii, :, center_h - h_half:center_h + h_half, center_w - w_half:center_w + w_half] = 1

    return masks.sum(dim=0).clamp(0, 1)


class ImageFolder:
    """An image folder dataset intended for self-supervised learning."""

    def __init__(self, path, transform=None, mask_transform=None, random_mask=False):
        # assuming 'path' is a folder of image files path and
        # 'annotation_path' is the base path for corresponding annotation json files
        self.samples = get_image_paths(path)
        self.transform = transform
        self.mask_transform = mask_transform
        self.random_mask = random_mask

    def __getitem__(self, idx: int):
        assert 0 <= idx < len(self)
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = ToTensor()(img)

        if self.transform:
            img = self.transform(img)

        # Get MASKS
        if self.random_mask:
            _, H, W = img.shape
            mask = create_random_mask(H, W, num_masks=random.randint(2,4), mask_percentage=random.random() * 0.2 + 0.2, max_attempts=100)
        else:
            mask = torch.ones_like(img[0:1, ...])
        assert mask.shape[1:] == img.shape[1:] and mask.shape[0] == 1

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        
        return img, mask

    def __len__(self):
        return len(self.samples)

class CocoImageIDWrapper(CocoDetection):
    def __init__(
        self, root, annFile, transform=None, mask_transform=None,
        random_nb_object=True, max_nb_masks=4, multi_w=False
    ) -> None:
        """
        Args:
            root (str): Root directory where images are saved.
            annFile (str): Path to json annotation file.
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
            mask_transform (callable, optional): The same as transform but for the mask.
            random_nb_object (bool, optional): If True, randomly sample the number of objects in the image. Defaults to True.
            max_nb_masks (int, optional): Maximum number of masks to return. Defaults to 4.
            multi_w (bool, optional): If True, return multiple masks as a single tensor. Defaults to False.
        """
        with suppress_output():
            super().__init__(root, annFile, transform=transform, target_transform=mask_transform)
        self.random_nb_object = random_nb_object
        self.max_nb_masks = max_nb_masks
        self.multi_w = multi_w

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(index, int):
            raise ValueError(
                f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        img = self._load_image(id)
        mask = self._load_mask(id)
        if mask is None:
            return None  # Skip this image if no valid mask is available

        # convert PIL to tensor
        img = ToTensor()(img)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def _load_mask(self, id):
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        if not anns:
            return None  # Return None if there are no annotations

        img_info = self.coco.loadImgs(id)[0]
        original_height = img_info['height']
        original_width = img_info['width']

        # Initialize a list to hold all masks
        masks = []
        if self.random_nb_object and np.random.rand() < 0.5:
            random.shuffle(anns)
            anns = anns[:np.random.randint(1, len(anns)+1)]
        if not (self.multi_w):
            mask = np.zeros((original_height, original_width),
                            dtype=np.float32)
            # one mask for all objects
            for ann in anns:
                rle = self.coco.annToRLE(ann)
                m = maskUtils.decode(rle)
                mask = np.maximum(mask, m)
            mask = torch.tensor(mask, dtype=torch.float32)
            return mask[None, ...]  # Add channel dimension
        else:
            anns = anns[:self.max_nb_masks]
            for ann in anns:
                rle = self.coco.annToRLE(ann)
                m = maskUtils.decode(rle)
                masks.append(m)
            # Stack all masks along a new dimension to create a multi-channel mask tensor
            if masks:
                masks = np.stack(masks, axis=0)
                masks = torch.tensor(masks, dtype=torch.bool)
                # Check if the number of masks is less than max_nb_masks
                if masks.shape[0] < self.max_nb_masks:
                    # Calculate the number of additional zero masks needed
                    additional_masks_count = self.max_nb_masks - masks.shape[0]
                    # Create additional zero masks
                    additional_masks = torch.zeros(
                        (additional_masks_count, original_height, original_width), dtype=torch.bool)
                    # Concatenate the original masks with the additional zero masks
                    masks = torch.cat([masks, additional_masks], dim=0)
            else:
                # Return a tensor of shape (max_nb_masks, height, width) filled with zeros if there are no masks
                masks = torch.zeros(
                    (self.max_nb_masks, original_height, original_width), dtype=torch.bool)
            return masks

if __name__ == "__main__":
    import time

    dataset = ImageFolder(path="/large_experiments/meres/sa-1b/anonymized_resized/valid/", annotations_folder="/datasets01/segment_anything/annotations/release_040523/")
    print(dataset[0][1])


    # Specify the path to the folder containing the MP4 files
    video_folder_path = "./assets/videos"

    from .transforms import get_resize_transform

    train_transform, train_mask_transform = get_resize_transform(img_size=256)
    val_transform, val_mask_transform = get_resize_transform(img_size=256)

    # Load and print stats for 3 videos for demonstration
    num_videos_to_print_stats = 3
    for i in range(min(num_videos_to_print_stats, len(dataset))):
        start_time = time.time()
        video_data, masks, frames_positions = dataset[i]
        end_time = time.time()
        print(f"Stats for video {i+1}/{num_videos_to_print_stats}:")
        print(
            f"  Time taken to load video: {end_time - start_time:.2f} seconds")
        print(f"  frames positions in returned clip: {frames_positions}")
        print(f"  Shape of video data: {video_data.shape}")
        print(f"  Data type of video data: {video_data.dtype}")
        print(f"Finished processing video {i+1}/{num_videos_to_print_stats}")

    print("Completed video stats test.")
