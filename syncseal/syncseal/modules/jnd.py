# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_jnd(imgs: torch.Tensor, imgs_w: torch.Tensor, hmaps: torch.Tensor, mode: str, alpha: float = 1.0) -> torch.Tensor:
    """ 
    Apply the JND model to the images.
    Args:
        imgs (torch.Tensor): The original images.
        imgs_w (torch.Tensor): The watermarked images.
        hmaps (torch.Tensor): The JND heatmaps.
        mode (str): The mode of applying the JND model.
            If 'multiply', the JND model is applied by multiplying the heatmaps with the difference between the watermarked and original images.
            If 'clamp', the JND model is applied by clamping the difference between the -jnd and +jnd values.
        alpha (float): The alpha value.
    """
    deltas = alpha * (imgs_w - imgs)
    if mode == 'multiply':
        deltas = hmaps * deltas
    elif mode == 'clamp':
        deltas = torch.clamp(deltas, -hmaps, hmaps)
    return imgs + deltas


class JND(nn.Module):
    """ https://ieeexplore.ieee.org/document/7885108 """
    
    def __init__(self,
            in_channels: int = 1,
            out_channels: int = 3,
            blue: bool = False,
            apply_mode: str = "multiply"
    ) -> None:
        super(JND, self).__init__()

        # setup input and output methods
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blue = blue
        groups = self.in_channels

        # create kernels
        kernel_x = torch.tensor(
            [[-1., 0., 1.], 
            [-2., 0., 2.], 
            [-1., 0., 1.]]
        ).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor(
            [[1., 2., 1.], 
            [0., 0., 0.], 
            [-1., -2., -1.]]
        ).unsqueeze(0).unsqueeze(0)
        kernel_lum = torch.tensor(
            [[1., 1., 1., 1., 1.], 
             [1., 2., 2., 2., 1.], 
             [1., 2., 0., 2., 1.], 
             [1., 2., 2., 2., 1.], 
             [1., 1., 1., 1., 1.]]
        ).unsqueeze(0).unsqueeze(0)

        # Expand kernels for 3 input channels and 3 output channels, apply the same filter to each channel
        kernel_x = kernel_x.repeat(groups, 1, 1, 1)
        kernel_y = kernel_y.repeat(groups, 1, 1, 1)
        kernel_lum = kernel_lum.repeat(groups, 1, 1, 1)

        self.conv_x = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, bias=False, groups=groups)
        self.conv_y = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, bias=False, groups=groups)
        self.conv_lum = nn.Conv2d(3, 3, kernel_size=(5, 5), padding=2, bias=False, groups=groups)

        self.conv_x.weight = nn.Parameter(kernel_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(kernel_y, requires_grad=False)
        self.conv_lum.weight = nn.Parameter(kernel_lum, requires_grad=False)

        # setup apply mode
        self.apply_mode = apply_mode

    def jnd_la(self, x: torch.Tensor, alpha: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
        """ Luminance masking: x must be in [0,255] """
        la = self.conv_lum(x) / 32
        mask_lum = la <= 127
        la[mask_lum] = 17 * (1 - torch.sqrt(la[mask_lum]/127 + eps))
        la[~mask_lum] = 3/128 * (la[~mask_lum] - 127) + 3
        return alpha * la

    def jnd_cm(self, x: torch.Tensor, beta: float = 0.117, eps: float = 1e-5) -> torch.Tensor:
        """ Contrast masking: x must be in [0,255] """
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        cm = torch.sqrt(grad_x**2 + grad_y**2)
        cm = 16 * cm**2.4 / (cm**2 + 26**2)
        return beta * cm

    # @torch.no_grad()
    def heatmaps(
        self, 
        imgs: torch.Tensor, 
        clc: float = 0.3
    ) -> torch.Tensor:
        """ imgs must be in [0,1] """
        imgs = 255 * imgs
        rgbs = torch.tensor([0.299, 0.587, 0.114])
        if self.in_channels == 1:
            imgs = rgbs[0] * imgs[...,0:1,:,:] + rgbs[1] * imgs[...,1:2,:,:] + rgbs[2] * imgs[...,2:3,:,:]  # luminance: b 1 h w
        la = self.jnd_la(imgs)
        cm = self.jnd_cm(imgs)
        hmaps = torch.clamp_min(la + cm - clc * torch.minimum(la, cm), 0)   # b 1or3 h w
        if self.out_channels == 3 and self.in_channels == 1:
            # rgbs = (1-rgbs).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            hmaps = hmaps.repeat(1, 3, 1, 1)  # b 3 h w
            if self.blue:
                hmaps[:, 0] = hmaps[:, 0] * 0.5
                hmaps[:, 1] = hmaps[:, 1] * 0.5
                hmaps[:, 2] = hmaps[:, 2] * 1.0
            # return  hmaps * rgbs.to(hmaps.device)  # b 3 h w
        elif self.out_channels == 1 and self.in_channels == 3:
            # rgbs = (1-rgbs).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            # return torch.sum(
            #     hmaps * rgbs.to(hmaps.device), 
            #     dim=1, keepdim=True
            # )  # b c h w * 1 c -> b 1 h w
            hmaps = torch.sum(hmaps / 3, dim=1, keepdim=True)  # b 1 h w
        return hmaps / 255

    def forward(self, imgs: torch.Tensor, imgs_w: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """ imgs and deltas must be in [0,1] """
        hmaps = self.heatmaps(imgs, clc=0.3)
        imgs_w = apply_jnd(imgs, imgs_w, hmaps, self.apply_mode, alpha)
        return imgs_w
