# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from ..modules.unet import UNet
from ..modules.vae import VAEDecoder, VAEEncoder


class Embedder(nn.Module):
    """
    Abstract class for watermark embedding.
    """

    def __init__(self) -> None:
        super(Embedder, self).__init__()
        self.yuv = False  # used by WAM module to know if the model should take YUV images

    def preprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        return imgs * 2 - 1

    def forward(
        self,
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        return None


class VAEEmbedder(Embedder):
    """
    Inserts a watermark into an image.
    """

    def __init__(
        self,
        encoder: VAEEncoder,
        decoder: VAEDecoder,
    ) -> None:
        super(VAEEmbedder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        imgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        imgs = self.preprocess(imgs)  # put in [-1, 1]
        latents = self.encoder(imgs)
        imgs_w = self.decoder(latents)
        return imgs_w


class UnetEmbedder(Embedder):
    """
    Inserts a watermark into an image.
    """

    def __init__(
        self,
        unet: nn.Module,
    ) -> None:
        super(UnetEmbedder, self).__init__()
        self.unet = unet

    def forward(
        self,
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        imgs = self.preprocess(imgs)  # put in [-1, 1]
        imgs_w = self.unet(imgs)
        return imgs_w


def build_embedder(name, cfg):
    if name.startswith('vae'):
        # build the encoder, decoder and msg processor
        encoder = VAEEncoder(**cfg.encoder)
        decoder = VAEDecoder(**cfg.decoder)
        embedder = VAEEmbedder(encoder, decoder)
    elif name.startswith('unet'):
        # updates some cfg
        unet = UNet(**cfg)
        embedder = UnetEmbedder(unet)
    else:
        raise NotImplementedError(f"Model {name} not implemented")
    embedder.yuv = True if 'yuv' in name else False
    return embedder
