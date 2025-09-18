# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Test with:
    python -m syncseal.augmentation.augmenter
"""

import os

import torch
from PIL import Image
from torch import nn
from torchvision.utils import save_image

from ..data.transforms import default_transform
from .masks import get_mask_embedder
from .valuemetric import (Identity, JPEG, Brightness, Contrast, GaussianBlur, Hue,
                          MedianFilter, Saturation, Pad, InsertMemeText,
                          InsertLogo)

name2aug = {
    'identity': Identity,
    'jpeg': JPEG,
    'gaussian_blur': GaussianBlur,
    'pad': Pad,
    "meme": InsertMemeText,
    "logo": InsertLogo,
    'median_filter': MedianFilter,
    'brightness': Brightness,
    'contrast': Contrast,
    'saturation': Saturation,
    'hue': Hue,
}
geometric_augs = ['rotate', 'crop', 'perspective', 'hflip', 'identity']
aes_augs = ['stable_diffusion_vae', 'stable_diffusion_xl_vae']

def get_dummy_augmenter():
    """
    An augmenter that does nothing.
    """
    
    return Augmenter(
        augs = {'identity': 1},
        augs_params = {},
        masks = {'kind': None}
    )

class Augmenter(nn.Module):
    """
    Augments the watermarked image.
    """

    def __init__(
        self,
        masks: dict,
        augs: dict,
        augs_params: dict,
        num_augs: int = 1,
        **kwargs: dict
    ) -> None:
        """
        Args:
            masks: (dict) The parameters for the mask embedder. \
                E.g. {'kind': None}
            augs: (dict) The augmentations to apply. \
                E.g. {'identity': 4, 'resize': 1, 'crop': 1}
            augs_params: (dict) The parameters for each augmentation. \
                E.g. {'resize': {'min_size': 0.7, 'max_size': 1.5}, 'crop': {'min_size': 0.7, 'max_size': 1.0}}
            num_augs: (int) The number of augmentations to apply.
            **kwargs: (dict) Additional arguments.
        """
        super(Augmenter, self).__init__()

        # create mask embedder
        self.mask_embedder = get_mask_embedder(
            **masks  # contains, e.g., invert_proba
        )

        # create augs
        self.augs, self.aug_probs = self.parse_augmentations(
            augs=augs,
            augs_params=augs_params
        )
        self.num_augs = num_augs
        # Put as module list to allow for to(device).
        self.augs = nn.ModuleList(self.augs)  

    def parse_augmentations(
        self,
        augs: dict[str, float],
        augs_params: dict[str, dict[str, float]]
    ):
        """
        Parse the post augmentations into a list of augmentations.
        Args:
            augs: (dict) The augmentations to apply.
                e.g. {'identity': 4, 'resize': 1, 'crop': 1}
            augs_params: (dict) The parameters for each augmentation.
                e.g. {'resize': {'min_size': 0.7, 'max_size': 1.5}, 'crop': {'min_size': 0.7, 'max_size': 1.0}}
        """
        augmentations = []
        probs = []
        # parse each augmentation
        for aug_name in augs.keys():
            aug_prob = float(augs[aug_name])
            if aug_prob > 0:
                aug_params = augs_params[aug_name] if aug_name in augs_params else {}
                try:
                    selected_aug = name2aug[aug_name](**aug_params)
                except KeyError:
                    raise ValueError(
                        f"Augmentation {aug_name} not found. Add it in name2aug.")
                augmentations.append(selected_aug)
                probs.append(aug_prob)
        # normalize probabilities
        total_prob = sum(probs)
        probs = [prob / total_prob for prob in probs]
        return augmentations, torch.tensor(probs)

    def augment(self, image, mask, do_resize=True):
        augs = self.augs
        aug_probs = self.aug_probs
        index = torch.multinomial(aug_probs, 1).item()
        selected_aug = augs[index]
        if not do_resize:
            image, mask = selected_aug(image, mask)
        else:
            h, w = image.shape[-2:]
            image, mask = selected_aug(image, mask)
            if image.shape[-2:] != (h, w):
                image = nn.functional.interpolate(image, size=(
                    h, w), mode='bilinear', align_corners=False, antialias=True)
                mask = nn.functional.interpolate(mask, size=(
                    h, w), mode='bilinear', align_corners=False, antialias=True)
        return image, mask, selected_aug.__class__.__name__

    def forward(
        self,
        imgs_w: torch.Tensor,
        imgs: torch.Tensor,
        masks: torch.Tensor,
        do_resize=True
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            imgs_w: (torch.Tensor) Batched watermarked images with shape BxCxHxW
            masks: (torch.Tensor) Batched masks with shape Bx1xHxW
        Returns:
            imgs_aug: (torch.Tensor) The augmented watermarked images.
            mask_targets: (torch.Tensor) The mask targets, at ones where the watermark is present.
        """
        if self.training:
            # create mask targets
            mask_targets = self.mask_embedder(imgs_w, masks=masks).to(imgs_w.device)
            # watermark masking
            imgs_aug = imgs_w * mask_targets + imgs * (1 - mask_targets)
            # image augmentations
            selected_augs = []
            for ii in range(self.num_augs):
                imgs_aug, mask_targets, selected_aug_ = self.augment(imgs_aug, mask_targets, do_resize)
                selected_augs.append(selected_aug_)
            selected_aug = "+".join(selected_augs)
            return imgs_aug, mask_targets, selected_aug
        else:
            # no mask
            mask_targets = torch.ones_like(imgs_w)[:, 0:1, :, :]
            imgs_aug = imgs_w * mask_targets + imgs * (1 - mask_targets)
            # image augmentations
            selected_augs = []
            for _ in range(self.num_augs):
                imgs_aug, mask_targets, selected_aug_ = self.augment(imgs_aug, mask_targets, do_resize)
                selected_augs.append(selected_aug_)
            selected_aug = "+".join(selected_augs)
            return imgs_aug, mask_targets, selected_aug

    def __repr__(self) -> str:
        # print the augmentations and their probabilities
        augs = [aug.__class__.__name__ for aug in self.augs]
        return f"Augmenter(augs={augs}, probs={self.aug_probs})"


if __name__ == "__main__":
    # Define the augmentations and their parameters
    augs = {
        'identity': 1,
        'jpeg': 1,
        'gaussian_blur': 1,
        'median_filter': 1,
    }
    augs_params = {
        'jpeg': {'min_quality': 40, 'max_quality': 80},
        'gaussian_blur': {'min_kernel_size': 3, 'max_kernel_size': 17},
        'median_filter': {'min_kernel_size': 3, 'max_kernel_size': 9},
    }

    # Create a batch of images
    imgs = [
        Image.open(f"images/squirrel.png"),
        Image.open(f"images/squirrel.png")
    ]
    imgs = torch.stack([default_transform(img) for img in imgs])
    imgs_w = imgs.clone()

    # Create an instance of the Augmenter class
    augmenter = Augmenter(
        augs=augs,
        augs_params=augs_params,
        kind='mixed',
        invert_proba=0.5
    )
    print("Augmenter:", augmenter)

    # Apply the augmentations to the images and save
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    for ii in range(10):
        imgs_aug, mask_targets, selected_aug = augmenter(imgs_w, imgs)
        save_image(imgs_aug.clamp(0, 1),
                   os.path.join(output_dir, f"imgs_aug_{ii}.png"), nrow=2)
        save_image(mask_targets, os.path.join(
            output_dir, f"mask_targets_{ii}.png"), nrow=2)
