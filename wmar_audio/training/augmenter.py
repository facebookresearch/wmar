# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Run test with:
    python -m training.augmenter
"""

import logging
import torch
import torch.nn as nn
from pathlib import Path
import torchaudio

from .augmentations import (
    Identity, TimeStretch, Speed, Echo, NoiseInjection, PinkNoise,
    LowpassFilter, HighpassFilter, BandpassFilter, Smooth,
    BoostAudio, DuckAudio, UpDownResample, MP3Compression,
    TimeShift, TemporalCrop
)

logger = logging.getLogger(__name__)

# Dictionary mapping augmentation names to their respective classes
name2aug = {
    'identity': Identity,
    'time_stretch': TimeStretch,
    'speed': Speed,
    'echo': Echo,
    'noise_injection': NoiseInjection,
    'pink_noise': PinkNoise,
    'lowpass_filter': LowpassFilter,
    'highpass_filter': HighpassFilter,
    'bandpass_filter': BandpassFilter,
    'smooth': Smooth,
    'boost_audio': BoostAudio,
    'duck_audio': DuckAudio,
    'up_down_resample': UpDownResample,
    'mp3_compression': MP3Compression,
    'time_shift': TimeShift,
    'temporal_crop': TemporalCrop
}


class Augmenter(nn.Module):
    """
    Augments audio data with various transformations.
    """

    def __init__(
        self,
        augs: dict,
        augs_params: dict = None,
        num_augs: int = 1,
        sample_rate: int = 24000,
        **kwargs: dict
    ) -> None:
        """
        Args:
            augs: (dict) The augmentations to apply with their relative weights. 
                E.g. {'identity': 4, 'speed': 1, 'time_stretch': 1}
            augs_params: (dict) The parameters for each augmentation. 
                E.g. {'speed': {'min_speed': 0.7, 'max_speed': 1.5}}
            num_augs: (int) The number of augmentations to apply sequentially.
            sample_rate: (int) The sample rate of the audio data.
            **kwargs: (dict) Additional arguments.
        """
        super(Augmenter, self).__init__()

        self.sample_rate = sample_rate
        augs_params = augs_params or {}
        
        # Create augmentations and their probabilities
        self.augs, self.aug_probs = self.parse_augmentations(
            augs=augs,
            augs_params=augs_params,
            sample_rate=sample_rate
        )
        
        self.num_augs = num_augs
        # Put as module list to allow for to(device)
        self.augs = nn.ModuleList(self.augs)

    def parse_augmentations(
        self,
        augs: dict[str, float],
        augs_params: dict[str, dict[str, float]],
        sample_rate: int
    ):
        """
        Parse the augmentations into a list of augmentations with their probabilities.
        
        Args:
            augs: (dict) The augmentations to apply with their relative weights.
                e.g. {'identity': 4, 'speed': 1, 'time_stretch': 1}
            augs_params: (dict) The parameters for each augmentation.
                e.g. {'speed': {'min_speed': 0.7, 'max_speed': 1.5}}
            sample_rate: (int) The sample rate of the audio data.
        
        Returns:
            tuple: List of augmentation objects and their normalized probabilities
        """
        augmentations = []
        probs = []
        
        # Parse each augmentation
        for aug_name, aug_prob in augs.items():
            if aug_prob > 0:
                aug_params = augs_params.get(aug_name, {})
                
                # Add sample_rate parameter if the augmentation accepts it
                if aug_name in ['speed', 'echo', 'lowpass_filter', 'highpass_filter', 
                              'bandpass_filter', 'up_down_resample', 'mp3_compression', 
                              'time_shift']:
                    aug_params['sample_rate'] = sample_rate
                
                try:
                    selected_aug = name2aug[aug_name](**aug_params)
                except KeyError:
                    raise ValueError(
                        f"Augmentation {aug_name} not found. Available augmentations: {list(name2aug.keys())}")
                
                augmentations.append(selected_aug)
                probs.append(float(aug_prob))
        
        # Normalize probabilities
        total_prob = sum(probs)
        if total_prob == 0:
            # Default to identity if no valid augmentations
            augmentations = [Identity()]
            probs = [1.0]
        else:
            probs = [prob / total_prob for prob in probs]
            
        return augmentations, torch.tensor(probs)

    def augment(self, audio, mask=None):
        """
        Apply a randomly selected augmentation to the audio.
        
        Args:
            audio: (torch.Tensor) Audio data with shape [batch, channels, time]
            mask: (torch.Tensor, optional) Mask for the audio
            
        Returns:
            tuple: Augmented audio, updated mask, and the name of the applied augmentation
        """
        index = torch.multinomial(self.aug_probs, 1).item()
        selected_aug = self.augs[index]
        augmented_audio, augmented_mask = selected_aug(audio, mask)
        return augmented_audio, augmented_mask, selected_aug.__class__.__name__

    def forward(
        self,
        audio: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple:
        """
        Apply random augmentations to the audio data.
        
        Args:
            audio: (torch.Tensor) Audio data with shape [batch, channels, time]
            mask: (torch.Tensor, optional) Mask for the audio
            
        Returns:
            tuple: Augmented audio, updated mask, and names of applied augmentations
        """
        audio_aug = audio
        current_mask = mask
        
        # Apply multiple augmentations sequentially
        selected_augs = []
        for _ in range(self.num_augs):
            audio_aug, current_mask, selected_aug_ = self.augment(audio_aug, current_mask)
            selected_augs.append(selected_aug_)
            
        selected_aug = "+".join(selected_augs)
        return audio_aug, current_mask, selected_aug

    def __repr__(self) -> str:
        """Return a string representation of the augmenter."""
        augs = [aug.__class__.__name__ for aug in self.augs]
        return f"Augmenter(augs={augs}, probs={self.aug_probs}, num_augs={self.num_augs})"


if __name__ == "__main__":
    from pathlib import Path
    import os
    import torchaudio
    import matplotlib.pyplot as plt
    
    # Load audio file, wget https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3
    audio_path = Path("assets/bria.mp3")
    if not audio_path.exists():
        print(f"Audio file {audio_path} not found. Using a generated tone instead.")
        # Generate a simple tone as fallback
        sample_rate = 16000
        duration = 3  # seconds
        t = torch.linspace(0, duration, int(sample_rate * duration))
        waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)  # 440 Hz
    else:
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        # Cut to 5 seconds if longer
        if waveform.shape[1] > sample_rate * 5:
            waveform = waveform[:, :sample_rate * 5]
    
    # Ensure it's shaped as [batch_size, channels, time]
    if len(waveform.shape) == 2:
        waveform = waveform.unsqueeze(0)  # Add batch dimension
    
    # Define the output directory
    output_dir = Path("outputs/audio_augmenter_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the augmentations and their parameters
    augs = {
        'identity': 1,
        'time_stretch': 1,
        'speed': 1,
        'echo': 1,
        'noise_injection': 1,
        'pink_noise': 1,
        'lowpass_filter': 1,
        'highpass_filter': 1,
        'mp3_compression': 1,
        'time_shift': 1,
        'temporal_crop': 1
    }
    
    # "augs_params":          "{'lowpass_filter':{'min_cutoff_freq':2000,'max_cutoff_freq':6000},'highpass_filter':{'min_cutoff_freq':200,'max_cutoff_freq':600},'noise_injection':{'min_noise_std':0.005,'max_noise_std':0.015},'pink_noise':{'min_noise_std':0.005,'max_noise_std':0.015},'mp3_compression':{'min_bitrate':16,'max_bitrate':128},'smooth':{'min_window_frac':0.001,'max_window_frac':0.01}}",
    augs_params = {
        'time_stretch':    {'min_rate': 0.7,   'max_rate': 1.3},
        'speed':           {'min_speed': 0.7,  'max_speed': 1.3},
        'echo':            {'min_volume': 0.2,'max_volume': 0.5},
        'noise_injection': {'min_noise_std': 0.005, 'max_noise_std': 0.015},
        'pink_noise':      {'min_noise_std': 0.005, 'max_noise_std': 0.015},
        'lowpass_filter':  {'min_cutoff_freq': 2000, 'max_cutoff_freq': 6000},
        'highpass_filter': {'min_cutoff_freq': 200,  'max_cutoff_freq': 600},
        'mp3_compression': {'min_bitrate': 64, 'max_bitrate': 128},
        'time_shift':      {'min_shift_ms': 100, 'max_shift_ms': 300},
        'temporal_crop':   {'min_crop_ratio': 0.6, 'max_crop_ratio': 0.9}
    }
    
    # Create an instance of the Augmenter class
    augmenter = Augmenter(
        augs=augs,
        augs_params=augs_params,
        num_augs=2,
        sample_rate=sample_rate
    )
    print("Augmenter:", augmenter)
    
    # Save the original audio as reference
    original_path = os.path.join(output_dir, "original.wav")
    torchaudio.save(original_path, waveform[0], sample_rate)
    print(f"Saved original audio to {original_path}")
    
    # Apply the augmentations to the audio and save multiple examples
    for i in range(10):
        # Apply augmentations
        audio_aug, _, selected_aug = augmenter(waveform)
        
        # Save augmented audio
        aug_path = os.path.join(output_dir, f"aug_{i}_{selected_aug}.wav")
        torchaudio.save(aug_path, audio_aug[0], sample_rate)
        print(f"Saved augmented audio to {aug_path} (augmentations: {selected_aug})")

    print(f"All augmentations applied and saved to {output_dir}")
