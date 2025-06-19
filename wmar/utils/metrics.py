# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.​


import numpy as np
import torch

try:
    pass
except ImportError:
    pass

from wmar.utils.utils import pillow_to_chw


# compute psnr between two PIL images
def compute_psnr(a, b, M=255.0):
    mse = np.mean((np.array(a) * 1.0 - np.array(b) * 1.0) ** 2)
    return 10 * np.log10(M**2 / mse)


def compute_metric(metric_name, code, orig_code, img, orig_img, watermarker, transform, param, compressors=None):
    if metric_name == "bpp":
        if transform == "neural-compress":
            # Find the compressor
            img_tensor = pillow_to_chw(img).clamp(-1, 1) / 2.0 + 0.5  # [0, 1]
            bpp = compressors[param](img_tensor.unsqueeze(0), return_bpp=True)[1]
            return bpp
        else:
            return None
    elif metric_name == "l0":
        return (orig_code != code).sum().item() / orig_code.shape[0]
    elif metric_name == "psnr":
        return compute_psnr(img, orig_img)
    else:
        if watermarker is None:
            return None

        if metric_name == "pvalue":
            return watermarker.detect(torch.LongTensor(code.reshape(1, -1)).to(watermarker.device)).item()
        else:
            raise ValueError(f"Metric {metric_name} not found")
