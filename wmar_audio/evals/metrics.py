# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import julius
import numpy as np

from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio, SignalNoiseRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

# Ensure tensors are on the correct device and have shape (batch, time) or (time,)
# For PESQ and STOI, the sample rate must be provided.

def calculate_sisnr(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculates Scale-Invariant Signal-to-Noise Ratio (SI-SNR)."""
    metric = ScaleInvariantSignalNoiseRatio().to(preds.device)
    return metric(preds, target)

def calculate_snr(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculates Signal-to-Noise Ratio (SNR)."""
    metric = SignalNoiseRatio().to(preds.device)
    return metric(preds, target)

def calculate_stoi(preds: torch.Tensor, target: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Calculates Short-Time Objective Intelligibility (STOI)."""
    metric = ShortTimeObjectiveIntelligibility(fs=sample_rate).to(preds.device)
    return metric(preds, target)

def calculate_pesq(preds: torch.Tensor, target: torch.Tensor, sample_rate: int, mode: str = "wb") -> torch.Tensor:
    """
    Calculates Perceptual Evaluation of Speech Quality (PESQ).
    Mode can be 'wb' (wide-band) or 'nb' (narrow-band).
    Returns NaN if PESQ calculation fails.
    """
    if mode not in ['wb', 'nb']:
        raise ValueError("Mode must be 'wb' or 'nb'")
    fs = 16000 # PESQ requires 16kHz sample rate
    metric = PerceptualEvaluationSpeechQuality(fs=fs, mode=mode).to(preds.device)
    try:
        pesq_val = metric(
        julius.resample_frac(preds, new_sr=fs, old_sr=sample_rate), 
        julius.resample_frac(target, new_sr=fs, old_sr=sample_rate)
    )
    except Exception as e:
        # Handle potential errors during PESQ calculation (e.g., NoUtterancesError)
        print(f"PESQ calculation failed: {e}")
        pesq_val = torch.tensor(np.nan, device=preds.device, dtype=preds.dtype)
    return pesq_val

