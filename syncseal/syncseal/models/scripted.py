# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
TorchScript-compatible SyncModel (watermark embedding + sync point detection).

Run with:
    python -m syncseal.models.scripted --checkpoint /path/to/checkpoint.pth
"""

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms.functional as TF
import argparse

from ..data.transforms import RGB2YUV
from ..modules.jnd import JND
from .embedder import Embedder
from .extractor import Extractor

class Blender(nn.Module):
    def __init__(self, scaling_i: float, scaling_w: float):
        super().__init__()
        self.scaling_i = float(scaling_i)
        self.scaling_w = float(scaling_w)

    def forward(self, imgs: torch.Tensor, preds_w: torch.Tensor) -> torch.Tensor:
        return self.scaling_i * imgs + self.scaling_w * preds_w


class SyncModelJIT(nn.Module):
    """
    Minimal TorchScript-friendly version of SyncModel:
        embed(imgs) -> dict{preds_w, imgs_w}
        detect(imgs) -> dict{preds} with preds shape B x (1+8)
    """

    def __init__(
        self,
        embedder: Embedder,
        detector: Extractor,
        attenuation: JND = None,
        scaling_w: float = 1.0,
        scaling_i: float = 1.0,
        img_size: int = 256,
        clamp: bool = True,
        round: bool = True,
        lowres_attenuation: bool = False,
    ):
        super().__init__()
        self.embedder = embedder
        self.detector = detector
        self.attenuation = attenuation
        self.img_size = int(img_size)
        self.rgb2yuv = RGB2YUV()
        self.blender = Blender(scaling_i, scaling_w)
        self.clamp = clamp
        self.round = round
        self.lowres_attenuation = lowres_attenuation

    def _apply_attenuation(self, imgs_ref: torch.Tensor, preds_w: torch.Tensor) -> torch.Tensor:
        if self.attenuation is None:
            return preds_w
        hmaps = self.attenuation.heatmaps(imgs_ref)
        return hmaps * preds_w

    def embed(
        self,
        imgs: torch.Tensor,
        mode: str = "bilinear",
        align_corners: bool = False,
        antialias: bool = True,
    ):
        # Resize to model resolution
        imgs_res = imgs
        do_resize = imgs.shape[-2:] != (self.img_size, self.img_size)
        if do_resize:
            imgs_res = F.interpolate(imgs, size=(self.img_size, self.img_size),
                                     mode=mode, align_corners=align_corners, antialias=antialias)
        # Predict watermark
        if self.embedder.yuv:
            preds_w = self.embedder(self.rgb2yuv(imgs_res)[:, 0:1])
        else:
            preds_w = self.embedder(imgs_res)
        # Optional low-res attenuation
        if self.attenuation is not None and self.lowres_attenuation:
            preds_w = self._apply_attenuation(imgs_res, preds_w)
        # Upscale back
        if do_resize:
            preds_w = F.interpolate(preds_w, size=imgs.shape[-2:],
                                     mode=mode, align_corners=align_corners, antialias=antialias)
        # Full-res attenuation path
        if self.attenuation is not None and not self.lowres_attenuation:
            preds_w = self._apply_attenuation(imgs, preds_w)
        # Blend + post-process
        imgs_w = self.blender(imgs, preds_w)
        if self.clamp:
            imgs_w = torch.clamp(imgs_w, 0, 1)
        if self.round:
            imgs_w = (torch.round(imgs_w * 255) / 255 - imgs_w).detach() + imgs_w
        return {"preds_w": preds_w, "imgs_w": imgs_w}

    def detect(
        self,
        imgs: torch.Tensor,
        mode: str = "bilinear",
        align_corners: bool = False,
        antialias: bool = True,
    ):
        # Resize
        imgs_res = imgs
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs_res = F.interpolate(imgs, size=(self.img_size, self.img_size),
                                     mode=mode, align_corners=align_corners, antialias=antialias)
        # Detector returns B x (1+8)
        preds = self.detector(imgs_res)
        return {"preds": preds, "preds_pts": preds[:, 1:]}

    def unwarp(
        self,
        imgs: torch.Tensor,
        pts_norm: torch.Tensor,
        original_size: tuple[int, int] = (256, 256),
    ) -> torch.Tensor:
        """
        Rectify imgs using normalized corner points ([-1,1]).
        Replicates corner_points path in helpers.warp_image_homography (perspective with torchvision).
        Args:
            imgs: BxCxHxW
            pts_norm: Bx8 (p1..p4 as (x,y) in [-1,1], order TL, TR, BR, BL),
            original_size: Optional original size (H,W) to use for unwarping, if None uses imgs shape
        Returns:
            BxCxHxW unwarped images
        """
        if pts_norm.dim() != 2 or pts_norm.shape[1] != 8:
            raise ValueError("pts_norm must be (B,8)")
        center = torch.tensor(
            [self.img_size / 2.0, self.img_size / 2.0], 
            device=imgs.device, dtype=imgs.dtype
        ) # tensor([128, 128])
        H, W = original_size
        start_pts_base = torch.tensor(
            [[0.0, 0.0], [W - 1.0, 0.0], [W - 1.0, H - 1.0], [0.0, H - 1.0]],
            device=imgs.device, dtype=imgs.dtype
        )  # (4,2)
        out = []
        for b in range(imgs.shape[0]):
            pts = pts_norm[b].view(4, 2)  # (4,2) normalized
            end_pts = (pts * center + center).round()  # denormalize to [0, 255]
            end_pts[..., 0] = end_pts[..., 0] * ((W - 1) / (self.img_size - 1)) # scale to [0, W-1] or [0, H-1]
            end_pts[..., 1] = end_pts[..., 1] * ((H - 1) / (self.img_size - 1)) # scale to [0, W-1] or [0, H-1]
            if torch.linalg.norm(end_pts - start_pts_base) > 1e6:
                out.append(imgs[b:b+1])
                continue
            # torchvision expects lists (or tensors) of 4 points each; operates per-image
            # Apply perspective warp from start rectangle to detected quadrilateral
            start_pts_list: list[list[int]] = start_pts_base.long().tolist()  # convert to list of lists
            end_pts_list: list[list[int]] = end_pts.long().tolist()  # convert to list of lists
            img_rect = imgs[b].clone()
            if img_rect.shape[-2:] != (H, W):
                img_rect = F.interpolate(img_rect.unsqueeze(0), size=(H, W),
                                         mode="bilinear", align_corners=False, antialias=True).squeeze(0)
            img_rect = TF.perspective(
                img_rect, start_pts_list, end_pts_list,
                interpolation=TF.InterpolationMode.BILINEAR, fill=None
            )
            out.append(img_rect.unsqueeze(0))
        return torch.cat(out, dim=0)

    # Convenience combined pass (optional)
    def forward(
        self,
        imgs: torch.Tensor,
        mode: str = "bilinear",
        align_corners: bool = False,
        antialias: bool = True,
    ):
        """
        Returns:
            emb: {'preds_w','imgs_w'}
            det: {'preds_pts'}
            unwarped: {'imgs_unwarped'}
        """
        emb = self.embed(imgs, mode, align_corners, antialias)
        det = self.detect(emb["imgs_w"], mode, align_corners, antialias)
        imgs_unwarped = self.unwarp(emb["imgs_w"], det["preds_pts"])
        return emb, det, {"imgs_unwarped": imgs_unwarped}


def test_sync_jit(ckpt: str):
    """
    Quick comparison between original loaded SyncModel and scripted SyncModelJIT.
    """
    from syncseal.utils.cfg import setup_model_from_checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = setup_model_from_checkpoint(ckpt)
    model.eval().to(device)

    # Dummy input
    imgs = torch.rand(2, 3, 512, 384, device=device)

    with torch.no_grad():
        emb_ref = model.embed(imgs)
        det_ref = model.detect(emb_ref["imgs_w"])

    jit_model = SyncModelJIT(
        model.embedder,
        model.detector,
        model.attenuation,
        scaling_w=model.blender.scaling_w,
        scaling_i=model.blender.scaling_i,
        img_size=model.img_size,
        clamp=model.clamp,
        round=model.rounding,
        lowres_attenuation=False,
    ).to(device).eval()

    scripted = torch.jit.script(jit_model)
    scripted.save("syncmodel.jit.pt")
    scripted = torch.jit.load("syncmodel.jit.pt").to(device).eval()

    with torch.no_grad():
        emb_j, det_j, uw_j = scripted(imgs)

    diff_embed = (emb_ref["imgs_w"] - emb_j["imgs_w"]).abs().mean().item()
    diff_detect = (det_ref["preds"][:, 1:] - det_j["preds_pts"]).abs().mean().item()
    print(f"Mean |Δ| embed imgs_w: {diff_embed:.6f}")
    print(f"Mean |Δ| detect preds_pts: {diff_detect:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Script and test SyncModelJIT")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to sync model checkpoint (.pth)")
    args = parser.parse_args()
    test_sync_jit(args.checkpoint)

if __name__ == "__main__":
    main()
