# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.​

import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from omegaconf import OmegaConf
from deps.rar.modeling.modules.maskgit_vqgan import Decoder as Pixel_Decoder
from deps.rar.modeling.modules.maskgit_vqgan import Encoder as Pixel_Encoder
from deps.rar.utils.train_utils import create_pretrained_tokenizer as create_rar_tokenizer
from deps.taming.modules.diffusionmodules.model import Decoder as TamingDecoder
from deps.taming.modules.diffusionmodules.model import Encoder as TamingEncoder
from deps.taming.util import (
    instantiate_from_config as instantiate_taming_or_vqgan_from_config,
)
from wmar.watermarking.gentime_watermark import GentimeWatermark


class AutoregressiveMultimodalModelWrapper:
    def __init__(self):
        pass

    def set_watermarker(self, watermarker: GentimeWatermark | None = None):
        raise NotImplementedError("Subclass should implement this, after init")

    def get_image_tokenizer(self):
        raise NotImplementedError("Subclass should implement this")

    def get_vq(self):
        raise NotImplementedError("Subclass should implement this")

    def get_total_vocab_size(self):
        raise NotImplementedError("Subclass should implement this")

    @property
    def device(self):
        return self.model.device

    def init_alivecodes(self, alive_ids_path):
        # Load alive IDs for this particular model
        vq = self.get_image_tokenizer().quantize
        vocab_sz = vq.n_e if hasattr(vq, "n_e") else vq.num_embeddings

        # Compute alive and dead
        alive_ids = []
        with open(alive_ids_path, "r") as f:
            for line in f:
                alive_ids.extend(list(map(int, line.split(","))))
        print(f"Loaded alive ids: {len(alive_ids)}")
        dead_ids = list(set(range(vocab_sz)) - set(alive_ids))
        vq.alive_ids = torch.tensor(alive_ids, dtype=torch.long)
        vq.dead_ids = torch.tensor(dead_ids, dtype=torch.long)

    # conditioning: list of size [b]
    # returned detached codes: [b, self.codes_size * self.codes_size]
    def sample(self, conditioning, gen_params, apply_watermark=False):
        raise NotImplementedError("Subclass should implement this")

    def sample_gimmick(self, conditioning, gen_params, apply_watermark=False):
        raise NotImplementedError("Subclass should implement this")

    # codes: [b, self.codes_size * self.codes_size], tokens
    # returns: [b, 3, self.image_size, self.image_size], pixels in [-1, 1]
    def codes_to_images(self, codes):
        raise NotImplementedError("Subclass should implement this")

    # images: [b, 3, self.image_size, self.image_size], pixels in [-1, 1]
    # returns: [b, self.codes_size * self.codes_size], tokens
    def images_to_codes(self, images):
        raise NotImplementedError("Subclass should implement this")

    # Shape checkers

    def is_codes_shaped(self, codes):
        return (
            isinstance(codes, torch.Tensor) and codes.ndim == 2 and codes.shape[1] == self.codes_size * self.codes_size
        )

    def is_images_shaped(self, images):
        return (
            isinstance(images, torch.Tensor)
            and images.ndim == 4
            and images.shape[1] == 3
            and images.shape[2] == self.image_size
            and images.shape[3] == self.image_size
        )


# Load any kind of VQGAN, might be Taming (for taming and cham) or titok for RAR
def load_model(
    config_path, ckpt_path, clone_encoder=False, backup_orig_decoder=False, device="cuda", vqgan_codebase="taming"
):
    config = OmegaConf.load(config_path)
    if vqgan_codebase == "rar":
        hf_hub_download(repo_id="fun-research/TiTok", filename="maskgit-vqgan-imagenet-f16-256.bin", local_dir="./")
        model = create_rar_tokenizer(config)  # also loads the weights
    else:
        model = instantiate_taming_or_vqgan_from_config(config.model)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)

    if clone_encoder:
        if vqgan_codebase == "rar":
            model.watermark_encoder = Pixel_Encoder(model.encoder.config)
        else:
            model.watermark_encoder = TamingEncoder(**config.model.params.ddconfig)
        model.watermark_encoder.load_state_dict(model.encoder.state_dict())
        for param in list(model.watermark_encoder.parameters()):
            param.requires_grad = False
        logger.info("Cloned encoder to watermark_encoder!")

    if backup_orig_decoder:
        if vqgan_codebase == "rar":
            model.orig_decoder = Pixel_Decoder(model.decoder.config)
        else:
            model.orig_decoder = TamingDecoder(**config.model.params.ddconfig)
        model.orig_decoder.load_state_dict(model.decoder.state_dict())
        for param in list(model.orig_decoder.parameters()):
            param.requires_grad = False
        logger.info("Cloned decoder to orig_decoder!")

    # Move
    model.to(device)
    model.eval()

    # Disable gradients
    for param in (
        list(model.quantize.parameters()) + list(model.encoder.parameters()) + list(model.decoder.parameters())
    ):
        param.requires_grad = False

    if vqgan_codebase == "taming":
        for param in list(model.quant_conv.parameters()) + list(model.post_quant_conv.parameters()):
            param.requires_grad = False

    return model
