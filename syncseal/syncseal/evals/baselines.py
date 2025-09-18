# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from ..modules.jnd import JND


class BaselineHiddenEmbedder(nn.Module):
    def __init__(
        self,
        encoder_path: str,
        nbits: int = 48,
    ) -> None:
        super(BaselineHiddenEmbedder, self).__init__()
        self.encoder = torch.jit.load(encoder_path).eval()
        self.nbits = nbits
        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.postprocess = transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    def get_random_msg(self, bsz: int = 1) -> torch.Tensor:
        return torch.randint(0, 2, (bsz, self.nbits))

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        msgs = 2 * msgs.float() - 1
        imgs = self.preprocess(imgs)
        imgs_w = self.encoder(imgs, msgs)
        imgs_w = self.postprocess(imgs_w)
        return imgs_w


class BaselineHiddenExtractor(nn.Module):
    def __init__(
        self,
        decoder_path: str
    ) -> None:
        super(BaselineHiddenExtractor, self).__init__()
        self.decoder = torch.jit.load(decoder_path).eval()
        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(
        self,
        imgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            The extracted messages.
        """
        imgs = self.preprocess(imgs)
        msgs = self.decoder(imgs)  # b k
        # add +1 to the last dimension to make it b k+1 (WAM compatible)
        msgs = torch.cat([torch.zeros(msgs.size(0), 1).to(msgs.device), msgs], dim=1)
        return msgs


class BaselineMBRSEmbedder(nn.Module):
    def __init__(
        self,
        encoder_path: str
    ) -> None:
        super(BaselineMBRSEmbedder, self).__init__()
        self.encoder = torch.jit.load(encoder_path).eval()
        self.nbits = 256
        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.postprocess = transforms.Normalize(mean=[-1, -1, -1],std=[2, 2, 2])

    def get_random_msg(self, bsz: int = 1) -> torch.Tensor:
        return torch.randint(0, 2, (bsz, self.nbits))

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        msgs = msgs.float()
        imgs_w = self.encoder(self.preprocess(imgs), msgs)
        imgs_w = self.postprocess(imgs_w)
        return imgs_w - imgs


class BaselineMBRSExtractor(nn.Module):
    def __init__(
        self,
        decoder_path: str,
    ) -> None:
        super(BaselineMBRSExtractor, self).__init__()
        self.decoder = torch.jit.load(decoder_path).eval()
        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def forward(
        self,
        imgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            The extracted messages.
        """
        imgs = self.preprocess(imgs)
        msgs = 2* self.decoder(imgs) -1  # b k
        # add +1 to the last dimension to make it b k+1 (WAM compatible)
        msgs = torch.cat([torch.zeros(msgs.size(0), 1).to(msgs.device), msgs], dim=1)
        return msgs


class BaselineCINEmbedder(nn.Module):
    def __init__(
        self,
        encoder_path: str
    ) -> None:
        super(BaselineCINEmbedder, self).__init__()
        self.encoder = torch.jit.load(encoder_path).eval()
        self.nbits = 30
        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.postprocess = transforms.Normalize(mean=[-1, -1, -1],std=[2, 2, 2])

    def get_random_msg(self, bsz: int = 1) -> torch.Tensor:
        return torch.randint(0, 2, (bsz, self.nbits))

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        msgs = msgs.float()
        imgs_w = self.encoder(self.preprocess(imgs), msgs)
        imgs_w = self.postprocess(imgs_w)
        return imgs_w - imgs


class BaselineCINExtractor(nn.Module):
    def __init__(
        self,
        decoder_path: str,
    ) -> None:
        """
        CIN decoder:
        - works at resolution 128x128
        - outputs msgs ≈ between 0,1
        """
        super(BaselineCINExtractor, self).__init__()
        self.decoder = torch.jit.load(decoder_path).eval()
        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def forward(
        self,
        imgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            The extracted messages.
        """
        imgs = self.preprocess(imgs)
        # scale the output to be between -1,1
        msgs = 2 * self.decoder(imgs) - 1 # b k
        # add +1 to the last dimension to make it b k+1 (WAM compatible)
        msgs = torch.cat([torch.zeros(msgs.size(0), 1).to(msgs.device), msgs], dim=1)
        return msgs


class BaselineWAMEmbedder(nn.Module):
    def __init__(
        self,
        encoder_path: str,
        nbits: int = 32,
    ) -> None:
        super(BaselineWAMEmbedder, self).__init__()
        self.encoder = torch.jit.load(encoder_path).eval()
        self.nbits = nbits
        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.postprocess = transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    def get_random_msg(self, bsz: int = 1) -> torch.Tensor:
        return torch.randint(0, 2, (bsz, self.nbits))

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        imgs = self.preprocess(imgs)
        imgs_w = self.encoder(imgs, msgs)
        imgs_w = self.postprocess(imgs_w)
        return imgs_w


class BaselineWAMExtractor(nn.Module):
    def __init__(
        self,
        decoder_path: str
    ) -> None:
        super(BaselineWAMExtractor, self).__init__()
        self.decoder = torch.jit.load(decoder_path).eval()
        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(
        self,
        imgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            The extracted messages.
        """
        imgs = self.preprocess(imgs)
        msgs = self.decoder(imgs)  # b 1+k h w
        msgs = msgs.mean(dim=[-2, -1])  # b 1+k
        return msgs


class BaselineTrustmarkEmbedder(nn.Module):
    def __init__(
        self,
        encoder_path: str,
        nbits: int = 100,
    ) -> None:
        super(BaselineTrustmarkEmbedder, self).__init__()
        self.encoder = torch.jit.load(encoder_path).eval()
        self.nbits = nbits
        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.postprocess = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])

    def get_random_msg(self, bsz: int = 1) -> torch.Tensor:
        return torch.randint(0, 2, (bsz, self.nbits))

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        msgs = msgs.float()
        imgs_w = self.encoder(self.preprocess(imgs), msgs)
        imgs_w = self.postprocess(imgs_w)
        return imgs_w - imgs


class BaselineTrustmarkExtractor(nn.Module):
    def __init__(
        self,
        decoder_path: str
    ) -> None:
        super(BaselineTrustmarkExtractor, self).__init__()
        self.decoder = torch.jit.load(decoder_path).eval()
        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def forward(
        self,
        imgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            The extracted messages.
        """
        imgs = self.preprocess(imgs)
        msgs = self.decoder(imgs)  # b k
        # add +1 to the last dimension to make it b k+1 (WAM compatible)
        msgs = torch.cat([torch.zeros(msgs.size(0), 1).to(msgs.device), msgs], dim=1)
        return msgs


class BaselineVideosealEmbedder(nn.Module):
    """
    Wrapper for the Videoseal jit model (e.g. checkpoints/y_256b_img.jit).
    The jit model exposes .embed(imgs, msgs, is_video=False) and expects images in [0,1].
    """
    def __init__(self, model_path: str):
        super(BaselineVideosealEmbedder, self).__init__()
        self.model = torch.jit.load(model_path).eval()
        self.nbits = 256

    def get_random_msg(self, bsz: int = 1) -> torch.Tensor:
        return torch.randint(0, 2, (bsz, self.nbits))

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            imgs: BxCxHxW (image) or TxCxHxW (video) / for batch of frames BxCxHxW
            msgs: BxK (or 1xK that will be broadcasted by the jit model)
            is_video: whether to call the jit model in video mode
        Returns:
            Watermarked images/frames in same shape as input, values in [0,1].
        """
        msgs = msgs.float()
        # ensure inputs are on same device as the model
        device = next(self.model.parameters()).device if any(True for _ in getattr(self.model, "parameters", lambda: [])()) else imgs.device
        imgs = imgs.to(device)
        msgs = msgs.to(device)
        imgs_w = self.model.embed(imgs, msgs, is_video=False)
        return imgs_w - imgs


class BaselineVideosealExtractor(nn.Module):
    """
    Wrapper for the Videoseal jit model detector (e.g. checkpoints/y_256b_img.jit).
    The jit model exposes .detect(imgs, is_video=False).
    Returns detection logits/probabilities. If the returned message length equals 256,
    a leading zero column is prepended to be compatible with WAM-style (k+1) outputs used elsewhere.
    """
    def __init__(self, model_path: str):
        super(BaselineVideosealExtractor, self).__init__()
        self.model = torch.jit.load(model_path).eval()
        self.nbits = 256

    def forward(
        self,
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            imgs: BxCxHxW (image) or TxCxHxW (video)
        Returns:
            preds: BxK (or Bx(K+1) if prefixed with a zero for compatibility)
        """
        device = next(self.model.parameters()).device if any(True for _ in getattr(self.model, "parameters", lambda: [])()) else imgs.device
        imgs = imgs.to(device)
        preds = self.model.detect(imgs, is_video=False)
        # If model returns length == nbits, prepend a zero column for WAM compatibility
        if preds.dim() == 2 and preds.size(1) == self.nbits:
            preds = torch.cat([torch.zeros(preds.size(0), 1, device=preds.device), preds], dim=1)
        return preds


class EmbedderExtractor(nn.Module):
    """
    A wrapper class that combines an embedder and extractor for baseline watermarking methods.
    This class provides a unified interface similar to the Wam class but simplified for baselines.
    """
    
    @property
    def device(self):
        """Return the device of the model."""
        return next(self.parameters()).device
    
    def __init__(
        self,
        embedder,
        detector,
        attenuation: JND = None,
        scaling_w: float = 1.0,
        scaling_i: float = 1.0,
        img_size: int = 256,
        clamp: bool = True,
        rounding: bool = True,
    ) -> None:
        """
        Initialize the EmbedderExtractor.
        
        Args:
            embedder: The watermark embedder
            detector: The watermark detector/extractor
            attenuation: The JND model to attenuate the watermark distortion
            scaling_w: The scaling factor for the watermark
            scaling_i: The scaling factor for the image
            img_size: The size at which the images are processed
            clamp: Whether to clamp the output images to [0, 1]
            rounding: Whether to apply rounding to simulate 8-bit quantization
        """
        super().__init__()
        self.embedder = embedder
        self.detector = detector
        self.attenuation = attenuation
        self.scaling_w = scaling_w
        self.scaling_i = scaling_i
        self.img_size = img_size
        self.clamp = clamp
        self.rounding = rounding
    
    def get_random_msg(self, bsz: int = 1) -> torch.Tensor:
        """Generate random messages."""
        return self.embedder.get_random_msg(bsz)
    
    def embed(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor = None,
        interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": True},
    ) -> dict:
        """
        Generates watermarked images from the input images and messages.
        
        Args:
            imgs: Batched images with shape BxCxHxW
            msgs: Optional messages with shape BxK
            interpolation: Interpolation parameters
            
        Returns:
            dict: Dictionary containing watermarked images and related data
        """
        # optionally create message
        if msgs is None:
            msgs = self.get_random_msg(imgs.shape[0])
            msgs = msgs.to(imgs.device)
        
        # interpolate to target size if needed
        original_size = imgs.shape[-2:]
        imgs_res = imgs.clone()
        if original_size != (self.img_size, self.img_size):
            imgs_res = F.interpolate(imgs, size=(self.img_size, self.img_size), **interpolation)
        
        # generate watermark
        preds_w = self.embedder(imgs_res, msgs)
        
        # scale watermark
        preds_w = preds_w * self.scaling_w
        
        # interpolate back to original size if needed
        if original_size != (self.img_size, self.img_size):
            preds_w = F.interpolate(preds_w, size=original_size, **interpolation)
        
        # blend watermark with original image
        imgs_w = imgs * self.scaling_i + preds_w
        
        # apply attenuation if provided
        if self.attenuation is not None:
            self.attenuation.to(imgs.device)
            imgs_w = self.attenuation(imgs, imgs_w)
        
        # clamp if requested
        if self.clamp:
            imgs_w = torch.clamp(imgs_w, 0, 1)
        
        # apply rounding to simulate 8-bit quantization
        if self.rounding:
            imgs_w = (torch.round(imgs_w * 255) / 255 - imgs_w).detach() + imgs_w
        
        outputs = {
            "msgs": msgs,
            "preds_w": preds_w,
            "imgs_w": imgs_w,
        }
        return outputs
    
    def detect(
        self,
        imgs: torch.Tensor,
        interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": True},
    ) -> dict:
        """
        Performs watermark detection on the input images.
        
        Args:
            imgs: Batched images with shape BxCxHxW
            interpolation: Interpolation parameters
            
        Returns:
            dict: Dictionary containing detection results
        """
        # interpolate to target size if needed
        original_size = imgs.shape[-2:]
        imgs_res = imgs.clone()
        if original_size != (self.img_size, self.img_size):
            imgs_res = F.interpolate(imgs, size=(self.img_size, self.img_size), **interpolation)
        
        # detect watermark
        preds = self.detector(imgs_res)
        
        outputs = {
            "preds": preds,
        }
        return outputs
    
    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor = None,
        interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": True},
    ) -> dict:
        """
        Forward pass that embeds and then detects watermarks.
        
        Args:
            imgs: Batched images with shape BxCxHxW
            msgs: Optional messages with shape BxK
            interpolation: Interpolation parameters
            
        Returns:
            dict: Dictionary containing all outputs
        """
        # embed watermark
        embed_outputs = self.embed(imgs, msgs, interpolation)
        
        # detect watermark in watermarked images
        detect_outputs = self.detect(embed_outputs["imgs_w"], interpolation)
        
        # combine outputs
        outputs = {**embed_outputs, **detect_outputs}
        return outputs


def build_baseline(
        method: str,
        attenuation: JND = None,
        scaling_w: float = 1.0,
        scaling_i: float = 1.0,
        img_size: int = 256,
        clamp: bool = True,
        rounding: bool = True,
    ) -> EmbedderExtractor:
    # ensure checkpoints directory is present; specific checkpoint existence is validated when loading
    assert os.path.exists('checkpoints'), """
Please download the baseline models into the checkpoints/ directory first.
See github.com/facebookresearch/videoseal/blob/main/docs/baselines.md for instructions.
"""
    if method == 'hidden':
        scaling_w = 0.2
        encoder_path = 'checkpoints/hidden_encoder_48b.pt'
        decoder_path = 'checkpoints/hidden_decoder_48b.pt'
        embedder = BaselineHiddenEmbedder(encoder_path)
        extractor = BaselineHiddenExtractor(decoder_path)
    elif method == 'mbrs':
        scaling_w = 1.0
        encoder_path = 'checkpoints/mbrs_256_m256_encoder.pt'
        decoder_path = 'checkpoints/mbrs_256_m256_decoder.pt'
        embedder = BaselineMBRSEmbedder(encoder_path)
        extractor = BaselineMBRSExtractor(decoder_path)
    elif method == 'cin':
        scaling_w = 1.0
        img_size = 128
        encoder_path = 'checkpoints/cin_nsm_encoder.pt'
        decoder_path = 'checkpoints/cin_nsm_decoder.pt'
        embedder = BaselineCINEmbedder(encoder_path)
        extractor = BaselineCINExtractor(decoder_path)
    elif method == 'wam':
        scaling_w = 2.0
        attenuation = JND(in_channels=1, out_channels=3, blue=True)
        encoder_path = 'checkpoints/wam_encoder.pt'
        decoder_path = 'checkpoints/wam_decoder.pt'
        embedder = BaselineWAMEmbedder(encoder_path)
        extractor = BaselineWAMExtractor(decoder_path)
    elif method == 'wam_noattenuation':
        scaling_w = 0.01
        encoder_path = 'checkpoints/wam_encoder.pt'
        decoder_path = 'checkpoints/wam_decoder.pt'
        embedder = BaselineWAMEmbedder(encoder_path)
        extractor = BaselineWAMExtractor(decoder_path)
    elif method == 'trustmark':
        scaling_w = 0.95  # set to 0.95 in the repo of TrustMark's authors
        encoder_path = 'checkpoints/trustmark_encoder_q.pt'
        decoder_path = 'checkpoints/trustmark_decoder_q.pt'
        embedder = BaselineTrustmarkEmbedder(encoder_path)
        extractor = BaselineTrustmarkExtractor(decoder_path)
    elif method == 'trustmark_scaling0p5':
        scaling_w = 0.5
        encoder_path = 'checkpoints/trustmark_encoder_q.pt'
        decoder_path = 'checkpoints/trustmark_decoder_q.pt'
        embedder = BaselineTrustmarkEmbedder(encoder_path)
        extractor = BaselineTrustmarkExtractor(decoder_path)
    elif method == 'videoseal':
        # Videoseal JIT model exposes .embed(imgs, msgs, is_video=False) and .detect(imgs, is_video=False)
        scaling_w = 1.0
        img_size = 256
        model_path = 'checkpoints/y_256b_img.pt'
        embedder = BaselineVideosealEmbedder(model_path)
        extractor = BaselineVideosealExtractor(model_path)
    else:
        raise ValueError(f'Unknown method: {method}')
    return EmbedderExtractor(
        embedder = embedder, 
        detector = extractor,
        attenuation = attenuation, 
        scaling_w = scaling_w, 
        scaling_i = scaling_i, 
        img_size = img_size, 
        clamp = clamp,
        rounding = rounding,
    )


if __name__ == '__main__':
    # Test the baseline models
    pass
