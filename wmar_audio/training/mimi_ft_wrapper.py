# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from moshi.models import MimiModel

class MimiFTWrapper(torch.nn.Module):
    
    def __init__(
        self,
        model: MimiModel,
        model_replica: MimiModel,
        augmenter=None,  # add augmenter
        augmentation_start: int = -1,  # add augmentation start epoch
        *args, 
        **kwargs
    ):
        """
        Wrapper for Mimi model to handle training with a replica model using a quantized audio representation pipeline.

        Pipeline:
        Input Audio → [Replica Encoder 🔒]  →  Quantizer (codes + embeddings) → Decoder  → Reconstructed audio → Encoder → Quantizer (codes + embeddings) \\
                                                                → [ Replica Decoder 🔒 ] → Reconstructed audio (Target)

        Quantization Submodule:
            Input Projection → Quantizer → Output Projection

        Description:
            - The replica model is frozen and used to generate ground-truth embeddings and reconstructions.
            - The main model is trained to mimic the latent and audio outputs of the replica.
            - Quantized codes are used to enforce consistency in the latent space.

        Args:
            model (MimiModel): The primary Mimi model used for prediction.
            model_replica (MimiModel): The frozen (locked) replica model used to generate targets for training.
            augmenter (optional): An optional augmenter to apply to the predicted audio reconstruction.
            augmentation_start (int): Epoch at which to start applying augmentations. -1 means never apply.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.model_replica = model_replica
        self.augmenter = augmenter
        self.augmentation_start = augmentation_start

    def forward(self, audio: torch.Tensor, epoch: int = -1):
        # encode with model replica.
        embs_pre_q = self.model_replica._encode_to_unquantized_latent(audio) # b 1 s -> b d s/f
        codes, embs_post_q, all_pre_q, all_post_q = self.model_replica.quantizer.encode_decode(embs_pre_q) # b d s/f -> b d s/f
        # decode from model replica. Will be used as audio target.
        audio_recon = self.model_replica._decode_from_unquantized_latent(embs_post_q) # b d s/f -> b 1 s
        # decode from model. Will be used as prediction.
        audio_recon_pred = self.model._decode_from_unquantized_latent(embs_post_q) # b d s/f -> b 1 s
        # apply augmentation if available and after start epoch
        if self.augmenter is not None and epoch >= self.augmentation_start:
            audio_recon_pred_aug, _, selected_aug = self.augmenter(audio_recon_pred)
        else:
            audio_recon_pred_aug, selected_aug = audio_recon_pred, "identity"
        # encode from model. Will be used as code target.
        recons_embs_pre_q_pred = self.model._encode_to_unquantized_latent(audio_recon_pred_aug) # b 1 s -> b d s/f
        recons_codes, recons_embs_post_q_pred, recons_all_pre_q, recons_all_post_q = self.model.quantizer.encode_decode(recons_embs_pre_q_pred) # b d s/f -> b d s/f
        return {
            "audio_recon": audio_recon,
            "audio_recon_pred": audio_recon_pred,
            "embs_pre_q": embs_pre_q,
            "recons_embs_pre_q_pred": recons_embs_pre_q_pred,
            "all_pre_q": all_pre_q,
            "recons_all_pre_q": recons_all_pre_q,
            "all_post_q": all_post_q,
            "recons_all_post_q": recons_all_post_q,
            "embs_post_q": embs_post_q,
            "recons_embs_post_q_pred": recons_embs_post_q_pred,
            "codes": codes,
            "recons_codes": recons_codes,
            "audio_recon_pred_aug": audio_recon_pred_aug,
            "selected_aug": selected_aug,
        }
