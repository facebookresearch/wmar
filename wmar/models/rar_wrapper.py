# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.​

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from deps.rar.modeling.rar import RAR
from deps.rar.utils.train_utils import create_pretrained_tokenizer
from wmar.models.armm_wrapper import AutoregressiveMultimodalModelWrapper
from wmar.watermarking.gentime_watermark import GentimeWatermark


class RarARMMWrapper(AutoregressiveMultimodalModelWrapper):
    def __init__(self, modelpath, rar_size="rar_xl"):
        """
        Initializes the RAR ARMM wrapper.
        Args: 
            modelpath: Path to the directory where the model files will be downloaded.
            rar_size: Size of the RAR model to use, e.g., "rar_b", "rar_l", "rar_xl", "rar_xxl".
        """
        super().__init__()
        device = "cuda"
        hf_hub_download(
            repo_id="fun-research/TiTok",
            filename="maskgit-vqgan-imagenet-f16-256.bin",
            local_dir=modelpath,
        )
        self.rar_size = rar_size
        hf_hub_download(repo_id="yucornetto/RAR", filename=f"{self.rar_size}.bin", local_dir=modelpath)

        config = OmegaConf.load("deps/rar/configs/training/generator/rar.yaml")
        config.model.vq_model.pretrained_tokenizer_weight = f"{modelpath}/maskgit-vqgan-imagenet-f16-256.bin"

        self.tokenizer = create_pretrained_tokenizer(config)
        self.tokenizer.to(device)
        self.tokenizer.eval()

        config.experiment.generator_checkpoint = f"{modelpath}/{self.rar_size}.bin"
        config.model.generator.hidden_size = {"rar_b": 768, "rar_l": 1024, "rar_xl": 1280, "rar_xxl": 1408}[
            self.rar_size
        ]
        config.model.generator.num_hidden_layers = {"rar_b": 24, "rar_l": 24, "rar_xl": 32, "rar_xxl": 40}[
            self.rar_size
        ]
        config.model.generator.num_attention_heads = 16
        config.model.generator.intermediate_size = {"rar_b": 3072, "rar_l": 4096, "rar_xl": 5120, "rar_xxl": 6144}[
            self.rar_size
        ]
        generator = RAR(config)
        generator.load_state_dict(torch.load(config.experiment.generator_checkpoint, map_location="cpu"))
        generator.eval()
        generator.requires_grad_(False)
        generator.set_random_ratio(0)
        self.model = generator
        self.model.to(device)
        self.model.eval()
        self.model.device = device

        self.init_alivecodes("assets/rar_all_ids.txt")

        self.codes_size = int(np.sqrt(config.model.generator.image_seq_len))
        assert "f16" in config.model.vq_model.pretrained_tokenizer_weight
        self.image_size = self.codes_size * 16
        self.dim_z = config.model.vq_model.token_size
        print(f"Codes size: {self.codes_size}, Image size: {self.image_size}, Dim z: {self.dim_z}")

    def __repr__(self):
        return "RarARMMWrapper"

    def set_watermarker(self, watermarker: GentimeWatermark | None = None):
        self.watermarker = watermarker

    def get_image_tokenizer(self):
        return self.tokenizer

    def get_vq(self):
        return self.tokenizer.quantize

    def get_total_vocab_size(self):
        return self.get_vq().num_embeddings

    # conditioning: list of size [b], class indices or (index, prompt) coco tuples
    # gen_params: dict
    # Returns: detached codes [b, self.codes_size * self.codes_size]
    def sample(self, conditioning, gen_params, apply_watermark=False):
        conditioning = torch.tensor(conditioning, device=self.model.device).view(-1, 1)
        watermark_logit_processor = self.watermarker.spawn_logit_processor() if apply_watermark else None

        codes = self.model.generate(
            condition=conditioning,
            guidance_scale=4.0,
            guidance_decay="constant",
            guidance_scale_pow=0.0,
            randomize_temperature=1.0,
            softmax_temperature_annealing=False,
            num_sample_steps=8,
            logit_processor=watermark_logit_processor,
        ).detach()

        assert self.is_codes_shaped(codes), f"Codes shape: {codes.shape}"
        return codes

    # codes: [b, self.codes_size * self.codes_size], tokens
    # returns: [b, 3, self.image_size, self.image_size], pixels in [-1, 1]
    def codes_to_images(self, codes):
        assert self.is_codes_shaped(codes), f"Codes shape: {codes.shape}"

        images = self.tokenizer.decode_tokens(codes)
        images = images * 2.0 - 1.0
        images = torch.clamp(images, -1.0, 1.0)

        assert self.is_images_shaped(images), f"Images shape: {images.shape}"
        return images

    # images: [b, 3, self.image_size, self.image_size], pixels in [-1, 1]
    # returns: [b, self.codes_size * self.codes_size], tokens
    def images_to_codes(self, images):
        assert self.is_images_shaped(images), f"Images shape: {images.shape}"

        images = (images + 1.0) / 2.0
        codes = self.tokenizer.encode(images)

        assert self.is_codes_shaped(codes), f"Codes shape: {codes.shape}"
        return codes
