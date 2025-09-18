# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
To run
    python -m syncseal.evals.metrics
"""

import math
from scipy import stats

import torch
import pytorch_msssim

def psnr(x, y, is_video=False):
    """ 
    Return PSNR 
    Args:
        x: Image tensor with normalized values (≈ [0,1])
        y: Image tensor with normalized values (≈ [0,1]), ex: original image
        is_video: If True, the PSNR is computed over the entire batch, not on each image separately
    """
    delta = 255 * (x - y)
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])  # BxCxHxW
    peak = 20 * math.log10(255.0)
    avg_on_dims = (0,1,2,3) if is_video else (1,2,3)
    noise = torch.mean(delta**2, dim=avg_on_dims)
    psnr = peak - 10*torch.log10(noise)
    return psnr

def ssim(x, y, data_range=1.0):
    """
    Return SSIM
    Args:
        x: Image tensor with normalized values (≈ [0,1])
        y: Image tensor with normalized values (≈ [0,1]), ex: original image
    """
    return pytorch_msssim.ssim(x, y, data_range=data_range, size_average=False)

def msssim(x, y, data_range=1.0):
    """
    Return MSSSIM
    Args:
        x: Image tensor with normalized values (≈ [0,1])
        y: Image tensor with normalized values (≈ [0,1]), ex: original image
    """
    return pytorch_msssim.ms_ssim(x, y, data_range=data_range, size_average=False)

def linf(x, y, data_range=1.0):
    """
    Return L_inf in pixel space (integer between 0 and 255)
    Args:
        x: Image tensor with normalized values (≈ [0,1])
        y: Image tensor with normalized values (≈ [0,1]), ex: original image
    """
    multiplier = 255.0 / data_range
    return torch.max(torch.abs(x - y)) * multiplier

def pvalue(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    mask: torch.Tensor = None,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Return p values
    Args:
        preds (torch.Tensor): Predicted bits with shape BxKxHxW
        targets (torch.Tensor): Target bits with shape BxK
        mask (torch.Tensor): Mask with shape Bx1xHxW (optional)
            Used to compute bit accuracy only on non masked pixels.
    """
    nbits = targets.shape[-1]
    bit_accs = bit_accuracy(preds, targets, mask, threshold)  # b
    pvalues = [stats.binomtest(int(p*nbits), nbits, 0.5, alternative='greater').pvalue for p in bit_accs]
    return torch.tensor(pvalues)  # b

def plogp(p: torch.Tensor) -> torch.Tensor:
    """
    Return p log p
    Args:
        p (torch.Tensor): Probability tensor with shape BxK
    """
    plogp = p * torch.log2(p)
    plogp[p == 0] = 0
    return plogp

def capacity(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor = None,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Return normalized bit accuracy, defined as the capacity of the nbits channels,
    in the case of a binary symmetric channel of error probability being the bit. acc.
    """
    nbits = targets.shape[-1]
    bit_accs = bit_accuracy(preds, targets, mask, threshold)  # b
    entropy = - plogp(bit_accs) - plogp(1-bit_accs)
    capacity = 1 - entropy
    capacity = nbits * capacity
    return capacity

def bit_accuracy(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    mask: torch.Tensor = None,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Return bit accuracy
    Args:
        preds (torch.Tensor): Predicted bits with shape BxKxHxW
        targets (torch.Tensor): Target bits with shape BxK
        mask (torch.Tensor): Mask with shape Bx1xHxW (optional)
            Used to compute bit accuracy only on non masked pixels.
            Bit accuracy will be NaN if all pixels are masked.
    """
    preds = preds > threshold  # b k ...
    if preds.dim() == 4:  # bit preds are pixelwise
        bsz, nbits, h, w = preds.size()
        if mask is not None:
            mask = mask.expand_as(preds).bool()
            preds = preds.masked_select(mask).view(bsz, nbits, -1)  # b k n
            preds = preds.mean(dim=-1, dtype=float)  # b k
        else:
            preds = preds.mean(dim=(-2, -1), dtype=float) # b k
    preds = preds > 0.5  # b k
    targets = targets > 0.5  # b k
    correct = (preds == targets).float()  # b k
    bit_acc = torch.mean(correct, dim=-1)  # b
    return bit_acc
