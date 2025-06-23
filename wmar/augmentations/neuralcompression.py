# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.​

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from compressai.zoo import models as compressai_models

    COMPRESSAI_AVAILABLE = True
except ImportError:
    COMPRESSAI_AVAILABLE = False
    print("CompressAI package not found. Install with pip install compressai")

try:
    from diffusers import AutoencoderDC, AutoencoderKL, FluxPipeline

    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Diffusers package not found. Install with pip install diffusers")


def get_model(model_name, quality):
    if model_name in compressai_models:
        return compressai_models[model_name](quality=quality, pretrained=True)
    else:
        avail_models = list(compressai_models.keys())
        raise ValueError(f"Model {model_name} not found. Available models: {avail_models}")


def get_diffusers_model(model_id):
    """Load a model from the Diffusers library"""
    if "dc-ae" in model_id.lower():
        model = AutoencoderDC.from_pretrained(model_id, device_map="cuda")
    elif "flux-vae" in model_id.lower():
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")
        model = AutoencoderKL.from_pretrained(pipe.vae.config._name_or_path, device_map="cuda")
        # spatial=8
    else:
        if "fp16" in model_id.lower():
            model = AutoencoderKL.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")
        else:
            model = AutoencoderKL.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cuda")
    return model


class NeuralCompression(nn.Module):
    def __init__(self, model_name, quality):
        super(NeuralCompression, self).__init__()
        self.model_name = model_name
        self.quality = quality
        self.model = get_model(model_name, quality)
        self.device = "cuda"
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def compute_bpp(self, out):
        size = out["x_hat"].size()
        num_pixels = size[0] * size[2] * size[3]
        return sum(
            torch.log(likelihoods).sum() / (-math.log(2) * num_pixels) for likelihoods in out["likelihoods"].values()
        ).item()

    def forward(self, image: torch.Tensor, return_bpp: bool = False, *args, **kwargs):
        orig_device = image.device
        if self.model_name not in ["bmshj2018-factorized"]:
            # resize to closest multiple of 64
            h, w = image.shape[-2:]
            h = max((h // 64) * 64, 64)
            w = max((w // 64) * 64, 64)
            if image.shape[-2:] != (h, w):
                image = F.interpolate(image, size=(h, w), mode="bilinear", align_corners=False)
        out = self.model(image.to(self.device))
        x_hat = out["x_hat"].to(orig_device)
        if return_bpp:
            bpp = self.compute_bpp(out)
            return x_hat, bpp
        else:
            return x_hat

    def __repr__(self):
        return f"{self.model_name}-q={self.quality}".replace("_", "-")

    @staticmethod
    def from_name(name):
        if "flux" in name.lower():
            return FluxVAE()
        elif "deep-compression" in name.lower():
            return DeepCompressionAE()
        elif name.lower() == "diffusers-sd-vae-fp16":
            return StableDiffusionXLVAEHalfPrecision()
        elif name.lower() == "diffusers-sd-vae-ft-ema":
            return StableDiffusionVAE()
        elif "bmshj2018-factorized" in name.lower():
            return BMSHJ2018Factorized(int(name.split("-q=")[-1]))
        elif "bmshj2018-hyperprior" in name.lower():
            return BMSHJ2018Hyperprior(int(name.split("-q=")[-1]))
        elif "mbt2018-mean" in name.lower():
            return MBT2018Mean(int(name.split("-q=")[-1]))
        elif "mbt2018" in name.lower():
            return MBT2018(int(name.split("-q=")[-1]))
        elif "cheng2020-anchor" in name.lower():
            return Cheng2020Anchor(int(name.split("-q=")[-1]))
        elif "cheng2020-attn" in name.lower():
            return Cheng2020Attn(int(name.split("-q=")[-1]))
        else:
            raise RuntimeError(f"Unknown diffuser model: {name}")


class DiffusersCompression(nn.Module):
    """Base class for models from the Diffusers library"""

    def __init__(self, model_id):
        super(DiffusersCompression, self).__init__()
        self.model_id = model_id
        self.model = get_diffusers_model(model_id)
        self.device = "cuda"
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def preprocess(self, x):
        """Preprocess image to DC input format [-1, 1]"""
        return 2.0 * x - 1.0

    def postprocess(self, x):
        """Convert DC output back to [0, 1]"""
        return (x + 1.0) / 2.0

    def forward(self, image: torch.Tensor, return_bpp: bool = False, *args, **kwargs):
        orig_device = image.device
        # cast to fp16 if model is like that
        orig_dtype = image.dtype
        if self.model.dtype == torch.float16:
            image = image.to(torch.float16)

        # Handle input size requirements if any
        h, w = image.shape[-2:]
        original_size = (h, w)

        # Some diffusers models require dimensions to be multiples of 16
        if h % 16 != 0 or w % 16 != 0:
            h = ((h // 16) + (1 if h % 16 != 0 else 0)) * 16
            w = ((w // 16) + (1 if w % 16 != 0 else 0)) * 16
            image = F.interpolate(image, size=(h, w), mode="bilinear", align_corners=False)

        image = image.to(self.model.device)

        if isinstance(self.model, AutoencoderDC):
            x = self.preprocess(image)
            latent = self.model.encode(x).latent
            y = self.model.decode(latent).sample
            x_hat = self.postprocess(y)
        else:
            # For AutoencoderKL
            x_hat = self.model.decode(self.model.encode(image).latent_dist.sample()).sample

        # Resize back to original if needed
        if original_size != (h, w):
            x_hat = F.interpolate(x_hat, size=original_size, mode="bilinear", align_corners=False)

        # Go back to orig type too and device
        x_hat = x_hat.to(orig_dtype)
        x_hat = x_hat.to(orig_device)

        if return_bpp:
            return x_hat, self.bpp
        else:
            return x_hat

    def __repr__(self):
        return f"diffusers-{self.model_id.split('/')[-1]}".replace("_", "-")


class StableDiffusionVAE(DiffusersCompression):
    def __init__(self):
        super(StableDiffusionVAE, self).__init__("stabilityai/sd-vae-ft-ema")
        # f=8, latents:(W/f, H/f, c=4), fp32
        # source: Table 8 of LDM paper
        # bpp = 4 * 32 / (8 * 8) = 2
        self.bpp = 2


class StableDiffusionXLVAEHalfPrecision(DiffusersCompression):
    def __init__(self):
        super(StableDiffusionXLVAEHalfPrecision, self).__init__("madebyollin/sdxl-vae-fp16-fix")
        # same as SD just trained differently but in fp16
        # f=8, latents:(W/f, H/f, c=4), fp16
        # source: SDXL paper 2.4
        self.bpp = 1

    def __repr__(self):
        return "diffusers-sdxl-vae-fp16"


class DeepCompressionAE(DiffusersCompression):
    def __init__(self):
        super(DeepCompressionAE, self).__init__("mit-han-lab/dc-ae-f64c128-in-1.0-diffusers")
        # f=64, c=128, fp32
        # bpp = 128 * 32 / (64 * 64) = 1
        self.bpp = 1

    def __repr__(self):
        return "diffusers-deep-compression"


class FluxVAE(DiffusersCompression):
    def __init__(self):
        super(FluxVAE, self).__init__("flux-vae")
        # f=16, c=16, fp32
        # bpp = 16 * 32 / (16 * 16) = 2
        self.bpp = 2

    def __repr__(self):
        return "diffusers-flux"


class BMSHJ2018Hyperprior(NeuralCompression):
    def __init__(self, quality):
        super(BMSHJ2018Hyperprior, self).__init__("bmshj2018-hyperprior", quality)


class BMSHJ2018Factorized(NeuralCompression):
    def __init__(self, quality):
        super(BMSHJ2018Factorized, self).__init__("bmshj2018-factorized", quality)


class MBT2018Mean(NeuralCompression):
    def __init__(self, quality):
        super(MBT2018Mean, self).__init__("mbt2018-mean", quality)


class MBT2018(NeuralCompression):
    def __init__(self, quality):
        super(MBT2018, self).__init__("mbt2018", quality)


class Cheng2020Anchor(NeuralCompression):
    def __init__(self, quality):
        super(Cheng2020Anchor, self).__init__("cheng2020-anchor", quality)


class Cheng2020Attn(NeuralCompression):
    def __init__(self, quality):
        super(Cheng2020Attn, self).__init__("cheng2020-attn", quality)
