# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from ..utils.dist import is_dist_avail_and_initialized
from .datasets import CocoImageIDWrapper, ImageFolder
from .transforms import default_transform


def get_dataloader(
    data_dir: str,
    transform: callable = default_transform,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 8
) -> DataLoader:
    """ Get dataloader for the images in the data_dir. The data_dir must be of the form: input/0/... """
    dataset = ImageFolder(data_dir, transform=transform)
    if is_dist_avail_and_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                sampler=sampler, num_workers=num_workers,
                                pin_memory=True, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers,
                                pin_memory=True, drop_last=True)
    return dataloader


def custom_collate(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])

    images, masks = zip(*batch)
    images = torch.stack(images)

    # Find the maximum number of masks in any single image
    max_masks = max(mask.shape[0] for mask in masks)
    if max_masks == 1:
        masks = torch.stack(masks)
        return images, masks

    # Pad each mask tensor to have 'max_masks' masks and add the inverse mask
    padded_masks = []
    for mask in masks:
        # Calculate the union of all masks in this image
        # Assuming mask is of shape [num_masks, H, W]
        union_mask = torch.max(mask, dim=0).values

        # Calculate the inverse of the union mask
        inverse_mask = ~union_mask

        # Pad the mask tensor to have 'max_masks' masks
        pad_size = max_masks - mask.shape[0]
        if pad_size > 0:
            padded_mask = F.pad(mask, pad=(
                0, 0, 0, 0, 0, pad_size), mode='constant', value=0)
        else:
            padded_mask = mask

        # Append the inverse mask to the padded mask tensor
        # padded_mask = torch.cat([padded_mask, inverse_mask.unsqueeze(0)], dim=0)

        padded_masks.append(padded_mask)

    # Stack the padded masks
    masks = torch.stack(padded_masks)

    return images, masks


def get_dataloader_segmentation(
    data_dir: str,
    ann_file: str,
    transform: callable,
    mask_transform: callable,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 8,
    random_nb_object=True,
    multi_w=False,
    max_nb_masks=4,
    random_mask=False,
) -> DataLoader:
    """ Get dataloader for COCO dataset. """
    # Initialize the CocoDetection dataset
    if "coco" in data_dir:
        dataset = CocoImageIDWrapper(root=data_dir, annFile=ann_file, transform=transform, mask_transform=mask_transform,
                                     random_nb_object=random_nb_object, multi_w=multi_w, max_nb_masks=max_nb_masks)
    else:
        dataset = ImageFolder(path=data_dir, transform=transform, mask_transform=mask_transform, random_mask=random_mask)

    if is_dist_avail_and_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=custom_collate)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=custom_collate)

    return dataloader

