# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.​

import json
import os 

import torch
from deps.chameleon.inference.chameleon import ChameleonInferenceModel, Options, DistributedMode
from wmar.models.armm_wrapper import AutoregressiveMultimodalModelWrapper
from wmar.watermarking.gentime_watermark import GentimeWatermark


class ChameleonARMMWrapper(AutoregressiveMultimodalModelWrapper):
    def __init__(self, modelpath, seed=0):
        super().__init__()
        weights_path = os.path.join(modelpath, "models", "7b")
        text_tokenizer_path = os.path.join(modelpath, "tokenizer", "text_tokenizer.json")
        image_tokenizer_cfg_path = os.path.join(modelpath, "tokenizer", "vqgan.yaml")
        image_tokenizer_path = os.path.join(modelpath, "tokenizer", "vqgan_patched.ckpt") 
        # NOTE: make sure to patch it first!
    
        self.model = ChameleonInferenceModel(
            weights_path,
            text_tokenizer_path, 
            image_tokenizer_cfg_path,
            image_tokenizer_path,
            distributed_mode=DistributedMode.THREAD
        )
        self.init_alivecodes("assets/chameleon_all_ids.txt")

        self.seed = seed
        self.codes_size = 32
        self.image_size = 512
        self.dim_z = 256

    def set_watermarker(self, watermarker: GentimeWatermark | None = None, watermarker_text: GentimeWatermark | None = None):
        self.watermarker = watermarker
        self.watermarker_text = watermarker_text
        self.model.set_watermarker_and_init_workers(watermarker, watermarker_text)

    def get_image_tokenizer(self):
        return self.model.token_manager.image_tokenizer._vq_model

    def get_vq(self):
        return self.get_image_tokenizer().quantize

    def get_total_vocab_size(self):
        return len(self.model.token_manager.vocab.all_tokens)

    def split_token_sequence(self, tokens: torch.LongTensor, boi: int, eoi: int) -> list[tuple[str, torch.LongTensor]]:
        """
        Split a sequence of tokens into text and image segments.

        Args:
            tokens (torch.LongTensor): The token sequence.
            boi (int): Begin of image token.
            eoi (int): End of image token.

        Returns:
            list[tuple[str, torch.LongTensor]]: List of tuples indicating segment type and tokens.
        """
        batch_size, _ = tokens.shape
        assert batch_size == 1, "Batch size must be 1"

        device = tokens.device
        tokens = tokens[0]
        tokens = tokens.to(device)
        segments = []
        current_segment = []
        in_image_seg = False

        for token in tokens:
            if token == boi:
                # if entering an image segment, save the current text segment (if any)
                if current_segment:
                    segments.append(
                        ("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1))
                    )
                    current_segment = []
                in_image_seg = True
            elif token == eoi and in_image_seg:
                # if exiting an image segment, save the current image segment
                segments.append(
                    ("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1))
                )
                current_segment = []
                in_image_seg = False
            else:
                current_segment.append(token)
        # save any remaining tokens
        if current_segment:
            if in_image_seg:
                segments.append(
                    ("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1))
                )
            else:
                segments.append(
                    ("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1))
                )
        return segments

    # conditioning: list of size [b], (index, prompt) coco tuples
    # gen_params: dict
    # Returns: detached codes [b, self.codes_size * self.codes_size]
    def sample_interleaved(self, conditioning, gen_params, apply_watermark=False):
        img_options = Options.Image()
        img_options.top_p = gen_params["top_p"]
        img_options.temp = gen_params["temperature"]
        options = Options(
            txt=True,
            img=img_options,
            seed=self.seed,
        )
        # Prepare batch prompts
        batch_prompt_ui = []
        for idx, prompt in conditioning:
            batch_prompt_ui += [
                [{"type": "text", "value": prompt}, {"type": "sentinel", "value": "<END-OF-TURN>"}],
            ]

        # Generate images
        codes: torch.LongTensor = self.model.generate(
            batch_prompt_ui=batch_prompt_ui,
            options=options,
            apply_watermark=apply_watermark,
        )
        codes = codes.contiguous()

        boi, eoi = self.model.vocab.begin_image, self.model.vocab.end_image
        segments = self.split_token_sequence(codes, boi, eoi)
        return segments

    # conditioning: list of size [b], class indices or (index, prompt) coco tuples
    # gen_params: dict
    # Returns: detached codes [b, self.codes_size * self.codes_size]
    def sample(self, conditioning, gen_params, apply_watermark=False):
        img_options = Options.Image()
        img_options.top_p = gen_params["top_p"]
        img_options.temp = gen_params["temperature"]
        options = Options(
            txt=False,
            img=img_options,
            seed=self.seed,
        )
        # Prepare batch prompts
        batch_prompt_ui = []
        for idx, prompt in conditioning:
            batch_prompt_ui += [
                [{"type": "text", "value": prompt}, {"type": "sentinel", "value": "<END-OF-TURN>"}],
            ]

        # Generate images
        # TODO: add watermark logit processor
        codes: torch.LongTensor = self.model.generate(
            batch_prompt_ui=batch_prompt_ui,
            options=options,
            apply_watermark=apply_watermark,
        )
        codes = codes.contiguous()
        assert self.is_codes_shaped(codes), f"Codes shape: {codes.shape}"
        return codes

    # codes: [b, self.codes_size * self.codes_size], tokens
    # returns: [b, 3, self.image_size, self.image_size], pixels in [-1, 1]
    def codes_to_images(self, codes):
        assert self.is_codes_shaped(codes), f"Codes shape: {codes.shape}"
        images_pil, images = self.model.decode_image(codes, return_imagetensor=True)
        images = torch.stack(images, dim=0).clamp(-1, 1)
        assert self.is_images_shaped(images), f"Images shape: {images.shape}"
        return images

    # images: [b, 3, self.image_size, self.image_size], pixels in [-1, 1]
    # returns: [b, self.codes_size * self.codes_size], tokens
    def images_to_codes(self, images):
        assert self.is_images_shaped(images), f"Images shape: {images.shape}"
        codes = []
        for image in images:
            image_pil = self.model.token_manager.image_tokenizer._pil_from_chw_tensor(image)
            tokens = self.model.token_manager.tokenize_image(image_pil)[1:-1]  # remove BOI and EOI
            codes.append(tokens)
        codes = torch.tensor(codes).to(images[0].device)
        assert self.is_codes_shaped(codes), f"Codes shape: {codes.shape}"
        return codes
