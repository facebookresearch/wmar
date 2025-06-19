# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.​
#
# Adapted from https://github.com/facebookresearch/videoseal

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, image, *args, **kwargs):
        return image

    def __repr__(self):
        return "Identity"


class Rotate(nn.Module):
    def __init__(self, min_angle=None, max_angle=None, do90=False):
        super(Rotate, self).__init__()
        self.min_angle = min_angle
        self.max_angle = max_angle
        if do90:
            self.base_angles = torch.tensor([-90, 0, 0, 90])
        else:
            self.base_angles = torch.tensor([0])

    def get_random_angle(self):
        if self.min_angle is None or self.max_angle is None:
            raise ValueError("min_angle and max_angle must be provided")
        base_angle = self.base_angles[torch.randint(0, len(self.base_angles), size=(1,))].item()
        return base_angle + torch.randint(self.min_angle, self.max_angle + 1, size=(1,)).item()

    def forward(self, image, angle=None):
        if angle is None:
            angle = self.get_random_angle()
        base_angle = angle // 90 * 90
        angle = angle - base_angle
        # rotate base_angle first with expand=True to avoid cropping
        image = F.rotate(image, base_angle, expand=True)
        # rotate the rest with expand=False
        image = F.rotate(image, angle)
        return image

    def __repr__(self):
        return "Rotate"


class UpperLeftCrop(nn.Module):
    def __init__(self, min_size=None, max_size=None):
        super(UpperLeftCrop, self).__init__()
        self.min_size = min_size
        self.max_size = max_size

    def get_random_size(self, h, w):
        if self.min_size is None or self.max_size is None:
            raise ValueError("min_size and max_size must be provided")
        output_size = (
            torch.randint(int(self.min_size * h), int(self.max_size * h) + 1, size=(1,)).item(),
            torch.randint(int(self.min_size * w), int(self.max_size * w) + 1, size=(1,)).item(),
        )
        return output_size

    def forward(self, image, size=None):
        h, w = image.shape[-2:]
        if size is None:
            output_size = self.get_random_size(h, w)
        else:
            output_size = (int(size * h), int(size * w))
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=output_size)
        image = F.crop(image, 0, 0, h, w)
        return image


class UpperLeftCropWithResizeBack(nn.Module):
    def __init__(self):
        super(UpperLeftCropWithResizeBack, self).__init__()
        self.crop = UpperLeftCrop()

    def forward(self, image, crop_size=None):
        output_size = (image.shape[-2], image.shape[-1])
        image = self.crop(image, crop_size)
        image = F.resize(image, output_size, antialias=True)
        return image


class UpperLeftCropWithPadBack(nn.Module):
    def __init__(self):
        super(UpperLeftCropWithPadBack, self).__init__()
        self.crop = UpperLeftCrop()

    def forward(self, image, crop_size=None):
        output_size = (image.shape[-2], image.shape[-1])
        image = self.crop(image, crop_size)
        pad = output_size[0] - image.shape[-2]
        image = F.pad(image, (0, 0, pad, pad), padding_mode="constant")
        return image


class HorizontalFlip(nn.Module):
    def __init__(self):
        super(HorizontalFlip, self).__init__()

    def forward(self, image, *args, **kwargs):
        image = F.hflip(image)
        return image

    def __repr__(self):
        return "HorizontalFlip"
