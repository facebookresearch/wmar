"""This file contains the model definition of TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf
from deps.rar.modeling.modules.base_model import BaseModel
from deps.rar.modeling.modules.blocks import TiTokDecoder, TiTokEncoder
from deps.rar.modeling.modules.maskgit_vqgan import Decoder as Pixel_Decoder
from deps.rar.modeling.modules.maskgit_vqgan import Encoder as Pixel_Eecoder
from deps.rar.modeling.modules.maskgit_vqgan import VectorQuantizer as Pixel_Quantizer
from deps.rar.modeling.quantizer.quantizer import (
    DiagonalGaussianDistribution,
    VectorQuantizer,
)
from deps.taming.modules.losses.lpips import LPIPS
from wmar.augmentations.geometric import Identity, Rotate, UpperLeftCropWithPadBack
from wmar.utils.utils import apply_random_augmentation


# TODO: this is "VQGAN", change it to support finetuning 
class PretrainedTokenizer(nn.Module):
    def __init__(self, pretrained_weight):
        super().__init__()
        conf = OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
            "num_resolutions": 5,
            "dropout": 0.0,
            "hidden_channels": 128,
            "num_channels": 3,
            "num_res_blocks": 2,
            "resolution": 256,
            "z_channels": 256})
        self.encoder = Pixel_Eecoder(conf)
        self.decoder = Pixel_Decoder(conf)
        self.quantize = Pixel_Quantizer(
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
        # Load pretrained weights
        self.load_state_dict(torch.load(pretrained_weight, map_location=torch.device("cpu")), strict=True)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

        # Finetuning, will be set from outside
        self.use_watermark_encoder = False
        self.loss_name = "" 
        self.loss_weight = 0.0

        self.perceptual_loss = LPIPS().eval()
    
    
    # Taming was: z_q, loss, (perplexity, min_encodings, min_encoding_indices)
    # Now: z_q, min_encoding_indices, loss
    @torch.no_grad()
    def encode(self, x):
        hidden_states = self.encoder(x)
        quantized_states, codebook_indices, codebook_loss = self.quantize(hidden_states)
        return codebook_indices.detach()
    
    @torch.no_grad()
    def decode(self, codes):
        quantized_states = self.quantize.get_codebook_entry(codes)
        rec_images = self.decoder(quantized_states)
        rec_images = torch.clamp(rec_images, 0.0, 1.0)
        return rec_images.detach()
    
    @torch.no_grad()
    def decode_tokens(self, codes):
        return self.decode(codes)

    ### Versions with grad and taming api for finetuning, [-1, 1] images etc 

    def quantize_like_taming(self, h):
        z_q, min_encoding_indices, loss = self.quantize(h)
        return z_q, loss, (None, None, min_encoding_indices)

    # BHWC in [-1, 1]
    def encode_like_taming(self, x):
        x = (x+1.0)/2.0 # [-1, 1] -> [0, 1]
        h = self.encoder(x)
        return self.quantize_like_taming(h)
    
    # B,C,16/32,16/32
    # embeddings, not indices! do the lookup before 
    # returns: BHWC in [-1, 1]
    def decode_like_taming(self, quant, decoder):
        dec = decoder(quant)
        dec = torch.clamp(dec, 0.0, 1.0)
        return dec*2.0 - 1.0 
    
    # all in [-1, 1]
    def loss(self, codebook_loss, inputs, reconstructions):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
        nll_loss = torch.mean(rec_loss + p_loss)
        loss = nll_loss + codebook_loss.mean()
        log = {
            "vqgan_loss": loss.clone().detach().mean().item(),
            "vqgan_rec_loss": rec_loss.detach().mean().item(),
            "vqgan_rec_loss_perceptual_component": p_loss.detach().mean().item(),
            "vqgan_codebook_loss": codebook_loss.detach().mean().item()
        }
        return loss, log 

    def forward(self, x, split, batch_idx, augmentations, augmentations_p):
        if x.ndim==4 and x.shape[2] == x.shape[3] and x.shape[3] in [256, 512]: # BHWC 
            # ImageNet training mode -> just go to codes
            z_q, qloss, (_, _, z_indices) = self.encode_like_taming(x)
            z_indices = z_indices.reshape(x.shape[0], -1) # add batch dimension
        else:
            # Transformer training mode, we are already in quantized codes 
            z_indices = x #[b, 256/512]
            z_q = self.quantize.embedding(z_indices) # [b, 256/512]
            # sqrt
            n_spatial = torch.sqrt(torch.tensor(z_q.shape[1])).int()
            assert n_spatial*n_spatial == z_q.shape[1]
            z_q = z_q.view(z_q.shape[0], n_spatial, n_spatial, z_q.shape[2])
            z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
            
            qloss = torch.tensor(0.0).to(x.device, non_blocking=True)

            # There is no original x for this case!
            x = None

        # Now proper latent training mode
        assert z_q.shape[2] == z_q.shape[3] and z_q.shape[3] in [16, 32], f"{z_q.shape} is not (b,c,16/32,16/32); are you sure you are using the right model?"

        # Decode to image 
        xrec = self.decode_like_taming(z_q, decoder=self.decoder)

        # Get loss from VQGAN and original decoder 
        xrec_orig_decoder = self.decode_like_taming(z_q, decoder=self.orig_decoder)

        vqgan_loss, log_dict = self.loss(qloss, xrec_orig_decoder, xrec)

        # Possibly augment and encode again for idempotence loss
        if augmentations is not None and len(augmentations) > 0:
            # NOTE: this flows grads back to xrec when possible, and does ST otherwise
            xrec_maybe_augmented, status = apply_random_augmentation(xrec, augmentations, p=augmentations_p)
            if status is None:
                was_augmented = False
            else:
                was_augmented = True
                applied_aug, applied_aug_param = status
        else:
            xrec_maybe_augmented = xrec
            was_augmented = False
        
        if self.use_watermark_encoder:
            zrec = self.watermark_encoder((xrec_maybe_augmented+1.0)/2.0) # not quantized 
        else:
            zrec = self.encoder((xrec_maybe_augmented+1.0)/2.0) # not quantized 
        zrec_q, _, (_, _, zrec_indices) = self.quantize_like_taming(zrec)
        zrec_indices = zrec_indices.reshape(xrec_maybe_augmented.shape[0], -1) # add batch dimension

        assert zrec.shape == z_q.shape, f"zrec shape {zrec.shape} != zq shape {z_q.shape}"
        
        # Get loss 
        if self.loss_name == "hard-to-soft-with-ae":
            if was_augmented and applied_aug is Rotate:
                # Skip first and last 1/8 of rows and columns for rotation
                skip = z_q.shape[2]//8
                idem_loss = torch.mean((z_q[:,:,skip:-skip, skip:-skip]-zrec[:,:,skip:-skip, skip:-skip])**2)
            elif was_augmented and applied_aug is UpperLeftCropWithPadBack:
                # Skip the cropped part for crop 
                cutoff = torch.floor(torch.tensor(z_q.shape[2])*applied_aug_param).int()
                idem_loss = torch.mean((z_q[:,:,:cutoff,:cutoff]-zrec[:,:,:cutoff,:cutoff])**2)
            else:
                idem_loss = torch.mean((z_q-zrec)**2) # standard idempotence

            loss = vqgan_loss + self.loss_weight*idem_loss
        else:
            raise ValueError(f"Loss {self.loss_name} not supported")
        log_dict[f"idem_loss"] = idem_loss.detach().mean().item()
        log_dict[f"loss"] = loss.detach().mean().item()
        log_dict[f"loss_weight"] = self.loss_weight # to track its annealing

        res_dict = {
            "orig_z_q": z_q,
            "orig_z_indices": z_indices,
            "rec_x": xrec,
            "rec_x_maybe_augmented": xrec_maybe_augmented,
            "rec_x_orig_decoder": xrec_orig_decoder,
            "rec_z": zrec,
            "rec_z_q": zrec_q,
            "rec_z_indices": zrec_indices,
        }
        return loss, res_dict, log_dict, was_augmented


class TiTok(BaseModel, PyTorchModelHubMixin, tags=["arxiv:2406.07550", "image-tokenization"], repo_url="https://github.com/bytedance/1d-tokenizer", license="apache-2.0"):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        # This should be False for stage1 and True for stage2.
        self.finetune_decoder = config.model.vq_model.get("finetune_decoder", True)

        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
        if self.quantize_mode not in ["vq", "vae"]:
            raise ValueError(f"Unsupported quantize mode {self.quantize_mode}.")
        
        if self.finetune_decoder and self.quantize_mode not in ["vq"]:
            raise ValueError("Only supprot finetune_decoder with vq quantization for now.")

        self.encoder = TiTokEncoder(config)
        self.decoder = TiTokDecoder(config)
        
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.apply(self._init_weights)

        if self.quantize_mode == "vq":
            self.quantize = VectorQuantizer(
                codebook_size=config.model.vq_model.codebook_size,
                token_size=config.model.vq_model.token_size,
                commitment_cost=config.model.vq_model.commitment_cost,
                use_l2_norm=config.model.vq_model.use_l2_norm,)
        elif self.quantize_mode == "vae":
            self.quantize = DiagonalGaussianDistribution
        else:
            raise NotImplementedError
        
        if self.finetune_decoder:
            # Freeze encoder/quantizer/latent tokens
            self.latent_tokens.requires_grad_(False)
            self.encoder.eval()
            self.encoder.requires_grad_(False)
            self.quantize.eval()
            self.quantize.requires_grad_(False)

            # Include MaskGiT-VQGAN's quantizer and decoder
            self.pixel_quantize = Pixel_Quantizer(
                num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
            self.pixel_decoder = Pixel_Decoder(OmegaConf.create(
                {"channel_mult": [1, 1, 2, 2, 4],
                "num_resolutions": 5,
                "dropout": 0.0,
                "hidden_channels": 128,
                "num_channels": 3,
                "num_res_blocks": 2,
                "resolution": 256,
                "z_channels": 256}))
        
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        if self.finetune_decoder:
            with torch.no_grad():
                self.encoder.eval()
                self.quantize.eval()
                z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
                z_quantized, result_dict = self.quantize(z)
                result_dict["quantizer_loss"] *= 0
                result_dict["commitment_loss"] *= 0
                result_dict["codebook_loss"] *= 0
        else:
            z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
            if self.quantize_mode == "vq":
                z_quantized, result_dict = self.quantize(z)
            elif self.quantize_mode == "vae":
                posteriors = self.quantize(z)
                z_quantized = posteriors.sample()
                result_dict = posteriors

        return z_quantized, result_dict
    
    def decode(self, z_quantized):
        decoded = self.decoder(z_quantized)
        if self.finetune_decoder:
            quantized_states = torch.einsum(
                'nchw,cd->ndhw', decoded.softmax(1),
                self.pixel_quantize.embedding.weight)
            decoded = self.pixel_decoder(quantized_states)
        return decoded
    
    def decode_tokens(self, tokens):
        if self.quantize_mode == "vq":
            tokens = tokens.squeeze(1)
            batch, seq_len = tokens.shape # B x N
            z_quantized = self.quantize.get_codebook_entry(
                tokens.reshape(-1)).reshape(batch, 1, seq_len, -1)
            z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        elif self.quantize_mode == "vae":
            z_quantized = tokens
        decoded = self.decode(z_quantized)
        return decoded
    
    def forward(self, x):
        z_quantized, result_dict = self.encode(x)
        decoded = self.decode(z_quantized)
        return decoded, result_dict