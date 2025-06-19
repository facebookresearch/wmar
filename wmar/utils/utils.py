# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.​

import io
import math
import random
from typing import Union
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageEnhance
from torch.nn import functional as F
from torchvision.transforms import PILToTensor
from transformers.generation.utils import TopKLogitsWarper, TopPLogitsWarper
from wmar.augmentations.geometric import Identity
from wmar.augmentations.valuemetric import JPEG


# Used in finetuning
# NOTE: images are in [-1, 1] but aug classes expect [0, 1]
def apply_random_augmentation(x, augmentations, p=0.5):
    if len(augmentations) == 0:
        return x, None
    if random.random() < p:
        cls, params = random.choice(augmentations)
        if cls is Identity:
            return x, None
        param = random.choice(params)
        transform_instance = cls()
        x_unit = x / 2.0 + 0.5
        x_unit_t, _ = transform_instance(x_unit, None, param)
        x_t = x_unit_t * 2.0 - 1.0

        # If it is not JPEG there should be gradient flow x_t -> x
        # If it is JPEG we need to do a straight-through here
        if cls is JPEG:
            x_t = x + (x_t - x).detach()

        return x_t, (cls, param)
    return x, None


def update_weights(model, ckpt_path, delta=True):  # Deltas!
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if delta:
        state_dict_to_apply = model.state_dict().copy()
        for key in state_dict:
            if key in state_dict_to_apply:
                state_dict_to_apply[key] = state_dict_to_apply[key] + state_dict[key].to(
                    state_dict_to_apply[key].device
                )
            else:
                state_dict_to_apply[key] = state_dict[key]
    else:
        state_dict_to_apply = state_dict

    missing, unexpected = model.load_state_dict(state_dict_to_apply, strict=False)
    logger.debug(f"Missing: {missing}")
    logger.debug(f"Unexpected: {unexpected}")


def simple_rescale(x):
    return (x + 1.0) / 2.0


# Rescale to [0, 1], clip (!), transpose, rescale to [0, 255], convert to uint8 -> PIL image
def chw_to_pillow(x: Union[torch.Tensor, np.ndarray]) -> Image.Image:
    # if torch go to numpy
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = (255 * simple_rescale(x.transpose(1, 2, 0))).clip(0, 255)
    x = np.round(x).astype(np.uint8)
    return Image.fromarray(x)


def pillow_to_chw(x: Image.Image) -> torch.Tensor:
    t = PILToTensor()(x.copy()) / 255.0 * 2 - 1  # [0,255] -> [-1,1], now no clipping needed
    t = t.cuda()
    return t


def add_code_to_plot(code_orig, code, watermarker, img_size=256):
    root = int(code.shape[-1] ** 0.5)
    code_orig = code_orig.reshape(root, root)
    code = code.reshape(root, root)
    multiplier = img_size // code.shape[-1]
    offset = multiplier // 2

    _, masks = watermarker.detect(code.ravel().unsqueeze(0), return_masks=True)
    mask = masks[0]

    it = 0
    for i in range(code.shape[0]):
        for j in range(code.shape[1]):
            plt.gca().add_patch(
                plt.Rectangle(
                    (i * multiplier, j * multiplier), multiplier, multiplier, fill=False, color="black", linewidth=1
                )
            )

            if code_orig[i, j].item() == code[i, j].item():
                char = "■"  # match
            else:
                char = "⨯"  # mismatch

            if mask[it] == -1:
                color = "white"
            elif mask[it] == 1:
                color = "green"
            elif mask[it] == 0:
                color = "red"
            else:
                raise RuntimeError(f"This should not be in the mask: {mask[it]}")
            it += 1

            # Invert (i, j) for matplotlib
            plt.text(j * multiplier + offset - 3, i * multiplier + offset + 2, char, color=color, fontsize=10)

def patch_chameleon(modelpath):
    # Chameleon (Anole): Patch the loss for compatibility with finetuning code -- Taming needed for this
    ckpt_path = os.path.join(modelpath, "tokenizer", "vqgan.ckpt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_old_path = "checkpoints/2021-04-03T19-39-50_cin_transformer/checkpoints/vqgan.ckpt"
    ckpt_old = torch.load(ckpt_old_path, map_location="cpu", weights_only=False)
    ks_to_delete = [] 
    for k, v in ckpt["state_dict"].items():
        if "loss.discriminator" in k:
            ks_to_delete.append(k)
    for k in ks_to_delete:
        del ckpt["state_dict"][k]
    for k, v in ckpt_old["state_dict"].items():
        if "loss.discriminator" in k:
            ckpt["state_dict"][k] = v 
    torch.save(ckpt, ckpt_path.replace(".ckpt", "_patched.ckpt"))