# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):

    def __init__(
        self, 
        upscale_type: str, 
        in_channels: int, 
        out_channels: int, 
        up_factor: int, 
        activation: nn.Module, 
        bias: bool = False
    ) -> None:
        """
        Build an upscaling block.
        Args:
            upscale_type (str): the type of upscaling to use
            in_channels (int): the input channel dimension
            out_channels (int): the output channel dimension
            up_factor (int): the upscaling factor
            activation (nn.Module): the type of activation to use
            bias (bool): whether to use bias in the convolution
        Returns:
            nn.Module: the upscaling block
        """
        super(Upsample, self).__init__()
        if upscale_type == 'nearest':
            upsample_block = nn.Sequential(
                nn.Upsample(scale_factor=up_factor, mode='nearest'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=bias),
                LayerNorm(out_channels, data_format="channels_first"),
                activation(),
            )
        elif upscale_type == 'bilinear':
            upsample_block = nn.Sequential(
                nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=bias),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=bias),
                LayerNorm(out_channels, data_format="channels_first"),
                activation(),
            )
        elif upscale_type == 'conv':
            upsample_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=up_factor, stride=up_factor),
                LayerNorm(out_channels, data_format="channels_first"),
                activation(),
            )
        elif upscale_type == 'pixelshuffle':
            conv = nn.Conv2d(in_channels, out_channels * up_factor ** 2, kernel_size=1, bias=False)
            upsample_block = nn.Sequential(
                conv,
                LayerNorm(out_channels * up_factor ** 2, data_format="channels_first"),
                activation(),
                nn.PixelShuffle(up_factor),
            )
            self.init_shuffle_conv_(conv, up_factor)
        else:
            raise ValueError(f"Invalid upscaling type: {upscale_type}")
        
        self.upsample_block = upsample_block

    def init_shuffle_conv_(self, conv, up_factor):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // (up_factor ** 2), i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = einops.repeat(conv_weight, f'o ... -> (o {up_factor ** 2}) ...')

        conv.weight.data.copy_(conv_weight)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample_block(x)
    

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, act_layer, do_init=True):
        super().__init__()
        conv = nn.Conv2d(in_channels * 4, out_channels, 1)
        self.net = nn.Sequential(
            nn.PixelUnshuffle(2),
            conv,
            act_layer()
        )
        if do_init:
            self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o, i // 4, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = einops.repeat(conv_weight, 'o i ... -> o (i 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)
    

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/  # noqa

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ChanRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.scale * self.gamma


def get_normalization(normalization: str) -> nn.Module:
    """ Set the normalization layer """
    if normalization.startswith("batch"):
        norm_layer = nn.BatchNorm2d
    elif normalization.startswith("group"):
        norm_layer = lambda num_channels: nn.GroupNorm(num_groups=8, num_channels=num_channels)
    elif normalization.startswith("layer"):
        norm_layer = LayerNorm
    elif normalization.startswith("rms"):
        norm_layer = ChanRMSNorm
    else:
        raise NotImplementedError
    return norm_layer

def get_activation(activation: str) -> nn.Module:
    """ Set the activation layer """
    if activation == "relu":
        act_layer = nn.ReLU
    elif activation == "leakyrelu":
        act_layer = partial(nn.LeakyReLU, negative_slope=0.2)
    elif activation == "gelu":
        act_layer = nn.GELU
    elif activation == "silu":
        act_layer = nn.SiLU
    else:
        raise NotImplementedError
    return act_layer


class Conv3dWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Wrapper for 3D convolution to handle 4D input tensors.
        Args:
            *args: Arguments for nn.Conv3d.
            **kwargs: Keyword arguments for nn.Conv3d.
        """
        super().__init__()
        self.conv = nn.Conv3d(*args, **kwargs)

    def forward(self, x):
        assert len(x.shape) == 4
        x = x.unsqueeze(0).permute(0, 2, 1, 3, 4) # change [B, C, H, W] to [1, C, T, H, W]
        x = self.conv(x)
        x = x.permute(0, 2, 1, 3, 4).squeeze(0)
        return x


class Conv2p1dWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Wrapper for 2D convolution then optional temporal convolution to handle 4D input tensors.
        Allows to keep 2D convolution unchanged, then add a temporal convolution.
        Args:
            *args: Arguments for nn.Conv2d.
            **kwargs: Keyword arguments for nn.Conv2d.
        """
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.temp_conv = None
        if kwargs["kernel_size"] != 1:
            assert isinstance(kwargs["kernel_size"], int)
            self.temp_conv = nn.Conv3d(
                args[1], args[1], # in_channels, out_channels
                kernel_size=(kwargs["kernel_size"], 1, 1), 
                padding=(kwargs["kernel_size"] // 2, 0, 0), 
                bias=False
            )

    def forward(self, x):
        assert len(x.shape) == 4
        x = self.conv(x)
        if self.temp_conv is not None:
            x = x.unsqueeze(0).permute(0, 2, 1, 3, 4) # change [B, C, H, W] to [1, C, T, H, W]
            x = self.temp_conv(x)
            x = x.permute(0, 2, 1, 3, 4).squeeze(0)
        return x

def get_conv_layer(name: str) -> nn.Module:
    """ Set the convolution layer """
    if name == "conv2d":
        return nn.Conv2d
    if name == "conv3d":
        return Conv3dWrapper
    if name == "conv2p1d":
        return Conv2p1dWrapper
    else:
        raise NotImplementedError


class AvgPool3dWrapper(nn.Module):
    def __init__(self, kernel_size=3, stride=None, padding=0, ceil_mode=True, count_include_pad=False):
        """
        Wrapper class for 3D average pooling to handle 4D input tensors. Almost reimplementation of AvgPool3d.
        Args:
            kernel_size: Size of pooling kernel
            stride: Stride of pooling operation 
            padding: Padding size
            ceil_mode: Whether to use ceil or floor for computing output size
            count_include_pad: Whether to include padding in averaging calculation
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x):
        assert len(x.shape) == 4
        x = x.unsqueeze(0).permute(0, 2, 1, 3, 4) # change [B, C, H, W] to [1, C, T, H, W]
        x = F.avg_pool3d(
            x, 
            (self.kernel_size, 1, 1),
            (self.stride, 1, 1),
            self.padding, 
            ceil_mode=self.ceil_mode, count_include_pad=self.count_include_pad
        )
        x = x.permute(0, 2, 1, 3, 4).squeeze(0)
        return x
