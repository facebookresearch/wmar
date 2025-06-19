# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Test synchronization watermarking with AudioSeal. Not very good yet (see App. D of the paper).
Run with:
    python -m wmar_audio.moshi.watermark.sync
"""

import numpy as np
import torch
from julius import resample_frac
from scipy.signal import correlate
from audioseal import AudioSeal

import torch
import torch.nn as nn

class SyncPattern():
    def __init__(
        self,
        frames_per_period = 3,
        frame_size = 1920,
        sample_rate = 24000,
    ):
        assert sample_rate == 24000, "SyncPattern only supports 24kHz sample rate."

        self.generator = AudioSeal.load_generator("audioseal_wm_16bits").eval()
        self.detector = AudioSeal.load_detector("audioseal_detector_16bits").eval()

        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.samples_per_period = frame_size * frames_per_period # 3 frames = 0.24s

        # create long template
        seconds = 20
        nsamples = self.sample_rate * seconds
        mask = self.generate_template(nsamples, 2 * self.samples_per_period)
        mask = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        self.mask = nn.Parameter(mask, requires_grad=False)
        
    @torch.no_grad()
    def get_sync_wm(self, waveform, alpha=1.0):
        """ Generate watermark, and add it to the waveform with the synchronization mask. """
        delta = self.generator.get_watermark(waveform, None)
        num_samples = waveform.shape[-1]
        mask = self.mask[..., :num_samples].to(waveform.device)
        return waveform + alpha * delta * mask

    @torch.no_grad()
    def detect_sync_wm(self, audio, num_samples=-1):
        """ Detect the watermark in the audio using the detector. """
        with torch.no_grad():
            detection_results, _ = self.detector(audio[..., :num_samples])
            detection_results = detection_results[:, 1].cpu().numpy() # b s
        return detection_results

    @staticmethod
    def generate_template(n_samples, period, shift=0):
        """
        Generate a wave template for cross-correlation.
        """
        # method = "square"
        method = "square"
        if method == "square":
            # Generate an ideal square wave template.
            t = np.arange(n_samples)
            template = ((t - shift) % period) < (period // 2)
            return template.astype(float)
        elif method == "sine":
            # Generate a sine wave template.
            t = np.arange(n_samples)
            template = 0.5 * (1 + np.sin(2 * np.pi * (t - shift) / period))
            return template.astype(float)
        
    @staticmethod
    def cross_correlation_search(signal, t_min, t_max, step):
        """
        Perform a cross-correlation search to find the best period and shift.
        """
        n_samples = len(signal)
        best_corr = -np.inf
        best_period = None
        best_shift = None

        candidate_periods = np.arange(t_min, t_max + 1, step)
        for period in candidate_periods:
            template = SyncPattern.generate_template(n_samples, period, shift=0)
            corr = correlate(signal, template, mode='full')
            peak_corr = np.max(np.abs(corr))
            if peak_corr > best_corr:
                best_corr = peak_corr
                best_period = period

        fine_range = np.arange(max(t_min, best_period - step), min(t_max, best_period + step) + 1)
        best_corr_fine = -np.inf
        for period in fine_range:
            template = SyncPattern.generate_template(n_samples, period, shift=0)
            corr = correlate(signal, template, mode='full')
            curr_peak = np.max(np.abs(corr))
            if curr_peak > best_corr_fine:
                best_corr = corr
                best_corr_fine = curr_peak
                best_period = period
        best_period = int(best_period)
        best_shift = np.argmax(best_corr) - (n_samples - 1)
        return best_period, best_shift, corr

    def get_speedup_and_shift(self, detection_signal, downsample_factor=8, step=10):
        """
        Estimate the speedup and phase shift of the watermark.
        Args:
            detection_signal (np.ndarray): The detection signal.
            downsample_factor (int): The factor by which to downsample the signal, to go faster.
            step (int): The step size for the cross-correlation search.
        Returns:
            tuple: Estimated speedup and phase shift.
        """
        downsampled = np.interp(
            np.arange(0, len(detection_signal), downsample_factor), # downsampled indices
            np.arange(len(detection_signal)), # original indices
            detection_signal # original signal
        )
        # only assume speed between 0.7x and 1.3x
        t_min = 0.5 * self.samples_per_period * 2 / downsample_factor
        t_max = 1.5 * self.samples_per_period * 2 / downsample_factor
        est_T, est_shift, corr = self.cross_correlation_search(downsampled, t_min, t_max, step)
        speedup = self.samples_per_period * 2 / (est_T * downsample_factor)
        shift = int(est_shift * speedup * downsample_factor) % self.frame_size
        return speedup, shift, corr
        
    def invert(self, audio, speedup, shift):
        """ Inverts the audio by applying the speedup and phase shift."""
        new_freq = int(self.sample_rate / speedup)
        audio = T.Resample(orig_freq=self.sample_rate, new_freq=new_freq)(audio)
        audio = audio[..., shift:] # remove the first `shift` samples
        return audio

if __name__ == "__main__":

    import os
    import matplotlib.pyplot as plt
    import torchaudio
    import torchaudio.transforms as T

    from ..training.augmentations import Speed

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio file, wget https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3
    audio_path = "assets/bria.mp3"
    waveform, sample_rate = torchaudio.load(audio_path)  # 1 s
    new_freq = 24_000
    waveform = T.Resample(orig_freq=sample_rate, new_freq=new_freq)(waveform).unsqueeze(0) # 1 1 s
    sample_rate = new_freq
    waveform = waveform[..., :10 * sample_rate]  # 10s

    # Example usage
    sync_pattern = SyncPattern()
    watermarked_waveform = sync_pattern.get_sync_wm(waveform, alpha=1.0)
    watermarked_audio_path = os.path.join(output_dir, "bria_wm_sync.wav")
    torchaudio.save(watermarked_audio_path, watermarked_waveform.squeeze(0), sample_rate)

    # Augment the watermarked audio
    aug_audio = watermarked_waveform[..., int(sample_rate * 0.84): int(sample_rate * 10)]
    speed_aug = Speed(sample_rate=sample_rate)
    aug_audio, _ = speed_aug(aug_audio, None, 1.05)  # 1.05x speed

    # detect the watermark
    detection_results = sync_pattern.detect_sync_wm(aug_audio, num_samples=aug_audio.shape[-1])
    detection_results = detection_results[0]  # s

    # Plot the detection results
    plt.figure(figsize=(12, 4))
    plt.plot(detection_results, label='Detection Results')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    plt.title('Watermark Detection Results')
    plt.xlabel('Sample Index')
    plt.ylabel('Detection Score')
    plt.legend()
    plt.show()

    # find the speedup and shift
    speedup, shift, _ = sync_pattern.get_speedup_and_shift(detection_results)
    print("Estimated speedup:", speedup)
    print("Estimated shift:", shift)
    # invert the audio
    inverted_audio = sync_pattern.invert(aug_audio, speedup, shift)
    inverted_audio_path = os.path.join(output_dir, "bria_wm_sync_inverted.wav")
    torchaudio.save(inverted_audio_path, inverted_audio.squeeze(0), sample_rate)
