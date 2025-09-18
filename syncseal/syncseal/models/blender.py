# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Blender(nn.Module):

    def __init__(self,  scaling_i, scaling_w):
        """
        Initializes the Blender class with a specific blending method and optional post-processing.

        Parameters:
            method (str): The blending method to use. 
            scaling_i (float): Scaling factor for the original image.
            scaling_w (float): Scaling factor for the watermark.
        """
        super(Blender, self).__init__()
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w

    def forward(self, imgs, preds_w):
        """
        Blends the original images with the predicted watermarks.
        E.g., if method is additive
            If scaling_i = 0.0 and scaling_w = 1.0, the watermarked image is predicted directly.
            If scaling_i = 1.0 and scaling_w = 0.2, the watermark is additive.
        Parameters:
            imgs (torch.Tensor): The original image batch tensor.
            preds_w (torch.Tensor): The watermark batch tensor.

        Returns:
            torch.Tensor: Blended and attenuated image batch.
        """
        return self.scaling_i * imgs + self.scaling_w * preds_w
