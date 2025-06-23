# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.​


import os

import torch
from omegaconf import OmegaConf
from deps.taming.modules.transformer.mingpt import (
    sample_with_past as taming_sample_with_past,
)
from deps.taming.util import (
    instantiate_from_config as instantiate_taming_or_vqgan_from_config,
)
from wmar.models.armm_wrapper import AutoregressiveMultimodalModelWrapper
from wmar.watermarking.gentime_watermark import GentimeWatermark


class TamingARMMWrapper(AutoregressiveMultimodalModelWrapper):
    def __init__(self, modelpath):
        super().__init__()
        # NOTE: make sure you download the models first (see README.md)
        config_path = os.path.join(modelpath, "configs/net2net.yaml")
        ckpt_path = os.path.join(modelpath, "checkpoints/net2net.ckpt")
        config = OmegaConf.load(config_path)
        model = instantiate_taming_or_vqgan_from_config(config.model)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
        model.to("cuda")
        model.eval()
        self.model = model

        self.init_alivecodes("assets/vqgan_alive_ids.txt")
        self.codes_size = 16
        self.image_size = 256
        self.dim_z = 256

    def __repr__(self):
        return "TamingARMMWrapper"

    def set_watermarker(self, watermarker: GentimeWatermark | None = None):
        self.watermarker = watermarker

    def get_image_tokenizer(self):
        return self.model.first_stage_model

    def get_vq(self):
        return self.get_image_tokenizer().quantize

    def get_total_vocab_size(self):
        return self.get_vq().n_e

    # conditioning: list of size [b]
    # gen_params: dict
    # Returns: detached codes [b, self.codes_size * self.codes_size]
    def sample(self, conditioning, gen_params, apply_watermark=False):
        conditioning = torch.tensor(conditioning, device=self.model.device).view(-1, 1)
        watermark_logit_processor = self.watermarker.spawn_logit_processor() if apply_watermark else None
        codes = taming_sample_with_past(
            conditioning,
            self.model.transformer,
            steps=self.codes_size * self.codes_size,
            sample_logits=True,
            temperature=gen_params["temperature"],
            top_k=gen_params["top_k"],
            top_p=gen_params["top_p"],
            logit_processor=watermark_logit_processor,
        ).detach()
        assert self.is_codes_shaped(codes), f"Codes shape: {codes.shape}"
        return codes

    # codes: [b, self.codes_size * self.codes_size], tokens
    # returns: [b, 3, self.image_size, self.image_size], pixels in [-1, 1]
    def codes_to_images(self, codes):
        assert self.is_codes_shaped(codes), f"Codes shape: {codes.shape}"
        qzshape = (codes.shape[0], self.dim_z, self.codes_size, self.codes_size)
        images = self.model.decode_to_img(codes, qzshape).clamp(-1, 1)
        assert self.is_images_shaped(images), f"Images shape: {images.shape}"
        return images

    # images: [b, 3, self.image_size, self.image_size], pixels in [-1, 1]
    # returns: [b, self.codes_size * self.codes_size], tokens
    def images_to_codes(self, images):
        assert self.is_images_shaped(images), f"Images shape: {images.shape}"
        codes = self.model.encode_to_z(images)[1]
        assert self.is_codes_shaped(codes), f"Codes shape: {codes.shape}"
        return codes
