# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.discriminator import NLayerDiscriminator
from ..utils.optim import freeze_grads
from .perceptual import PerceptualLoss


def hinge_d_loss(logits_real, logits_fake):
    """
    https://paperswithcode.com/method/gan-hinge-loss
    """
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.):
    """
    Adopt weight if global step is less than threshold
    """
    if global_step < threshold:
        weight = value
    return weight


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class SyncLoss(nn.Module):
    def __init__(self,
                 disc_weight=1.0, percep_weight=1.0, detect_weight=1.0, sync_weight=1.0,
                 disc_start=0, disc_num_layers=3, disc_in_channels=3, disc_loss="hinge", use_actnorm=False,
                 percep_loss="lpips", transform_loss="mse",
                 ):
        super().__init__()

        self.percep_weight = percep_weight
        self.detect_weight = detect_weight
        self.disc_weight = disc_weight
        self.sync_weight = sync_weight
        self.transform_loss = transform_loss

        self.perceptual_loss = PerceptualLoss(percep_loss=percep_loss)

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss

        self.detection_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.mse = torch.nn.MSELoss(reduction="none")
        self.mae = torch.nn.L1Loss(reduction="none")

    def forward(self,
        inputs: torch.Tensor, reconstructions: torch.Tensor,
        preds: torch.Tensor, target_pts: torch.Tensor,
        optimizer_idx: int, global_step: int,
        last_layer=None, cond=None,
    ) -> tuple:

        if optimizer_idx == 0:  # embedder update
            weights = {}
            losses = {}

            # perceptual loss
            if self.percep_weight > 0:
                losses["percep"] = self.perceptual_loss(
                    imgs=inputs.contiguous(),
                    imgs_w=reconstructions.contiguous(),
                ).mean()
                weights["percep"] = self.percep_weight

            # discriminator loss
            if self.disc_weight > 0:
                with freeze_grads(self.discriminator):
                    disc_factor = adopt_weight(1.0, global_step, threshold=self.discriminator_iter_start)
                    logits_fake = self.discriminator(reconstructions.contiguous())
                    losses["disc"] = - logits_fake.mean()
                    weights["disc"] = disc_factor * self.disc_weight

            # detection loss
            if self.detect_weight > 0:
                detection_loss = self.detection_loss(
                    preds[:, 0:1].contiguous(),
                    torch.ones_like(preds[:, 0:1]).contiguous(),
                ).mean()
                losses["detect"] = detection_loss
                weights["detect"] = self.detect_weight

            # transformation prediction loss
            if self.sync_weight > 0:
                transform_loss = self.compute_geometric(
                    preds[:, 1:].contiguous(),
                    target_pts
                ).mean()
                losses["transform"] = transform_loss
                weights["transform"] = self.sync_weight

            scales = weights
            total_loss = sum(scales[key] * losses[key] for key in losses)
            # log
            log = {
                "total_loss": total_loss.clone().detach().mean(),
                **{f"loss_{k}": v.clone().detach().mean() for k, v in losses.items()},
                **{f"scale_{k}": v for k, v in scales.items()}
            }
            return total_loss, log

        if optimizer_idx == 1:  # discriminator update
            if cond is None:
                logits_real = self.discriminator(
                    inputs.contiguous().detach())
                logits_fake = self.discriminator(
                    reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(
                1.0, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"disc_loss": d_loss.clone().detach().mean(),
                   "disc_factor": disc_factor,
                   "logits_real": logits_real.detach().mean(),
                   "logits_fake": logits_fake.detach().mean()
                   }
            return d_loss, log

    def compute_geometric(self, preds, target_pts):
        """
        Compute the transformation prediction loss.
        
        Args:
            preds: Model predictions [B, 8]
            target_pts: Target points for the augmentation [8] - normalized to [-1, 1] range
            
        Returns:
            Loss tensor
        """
        assert preds.shape[1] == 8, "Predictions must have shape [B, 8] for transformation loss."
        assert target_pts.shape == preds.shape, "Target points must have the same shape as predictions [B, 8]."

        if self.transform_loss == "mse":
            loss = self.mse(preds, target_pts)
        elif self.transform_loss == "mae":
            loss = self.mae(preds, target_pts)
        else:
            raise ValueError(f"Unknown transform loss type: {self.transform_loss}. Supported types: 'mse', 'mae'")
        
        return loss

    def to(self, device, *args, **kwargs):
        super().to(device)
        self.perceptual_loss = self.perceptual_loss.to(device)
        return self
