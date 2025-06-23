# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/utils/audio_effects.py

"""
Run with:
    python -m training.augmentations
"""

import io
import logging
import random
import re
import tempfile
import subprocess

import julius
from julius import fft_conv1d, resample_frac

import torch
import torch.nn as nn
from torch.nn.functional import pad
import torchaudio
from torchaudio.transforms import TimeStretch as TorchAudioTimeStretch
import torchaudio.transforms as T

try:
    import dac
except ImportError:
    print("Warning: Dac library not found. Install with `pip install descript-audio-codec`.")

try:
    from transformers import EncodecModel, AutoProcessor
except ImportError:
    print("Warning: Encodec library not found. Install with `pip install transformers`.")

logger = logging.getLogger(__name__)

def convert_to_format_and_back(audio_batch, sample_rate, format, bitrate=128, lowpass_freq=None):
    """
    Function to convert a batch of torch tensor audios to format and then back to tensors.
    
    Parameters:
    audio_batch (torch.Tensor): The batch of audio data in torch tensor format.
    sample_rate (int): The sample rate of the audio data.
    bitrate (int): The bitrate for the compressed audio.
    lowpass_freq (int, optional): The frequency for a low-pass filter. If None, no filter is applied.
    
    Returns:
    torch.Tensor: The batch of compressed and then decompressed audio data.
    """
    # Initialize an empty list to store the processed tensors
    processed_batch = []
    # Process each audio tensor in the batch
    for audio_tensor in audio_batch.cpu():
        # try:
        # Create a temporary file for the input audio
        with tempfile.NamedTemporaryFile(suffix=".wav") as f_in:
            input_path = f_in.name
            # Save the tensor as a WAV file
            torchaudio.save(input_path, audio_tensor, sample_rate)
            # Create a temporary file for the AAC audio
            with tempfile.NamedTemporaryFile(suffix=f".mp3") as f_out:
                output_path = f_out.name
                # Call FFmpeg to save the audio in AAC format
                command = [
                    "ffmpeg",
                    "-y",  # Overwrite output file if it exists
                    "-i",
                    input_path,  # Input file
                    "-ar",
                    str(sample_rate),  # Sample rate
                    "-b:a",
                    f"{bitrate}k",  # Bitrate
                    "-c:a",
                    f"{format}",  # Codec
                ]
                # Apply low-pass filter if frequency is provided
                if lowpass_freq is not None:
                    command += ["-cutoff", f"{lowpass_freq}"]
                command.append(output_path)  # Output file
                # Run FFmpeg - suppress output
                subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Load the AAC audio back into a tensor
                aac_tensor, _ = torchaudio.load(output_path)
            # Append the processed tensor to the list
            processed_batch.append(aac_tensor)
        # except:
        #     # If there is an error, append the original tensor to the list
        #     print(f'Error converting audio to {format}')
        #     processed_batch.append(audio_tensor)
    # Convert the list of tensors into a single tensor
    return torch.stack(processed_batch).to(audio_batch.device)

class Speed(nn.Module):
    def __init__(self, min_speed: float = 0.5, max_speed: float = 1.5, sample_rate: int = 16000):
        super().__init__()
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.sample_rate = sample_rate
        
    def __repr__(self):
        return self.__class__.__name__

    def get_random_speed(self):
        return torch.FloatTensor(1).uniform_(self.min_speed, self.max_speed)

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None, speed: float = None):
        speed = speed if speed is not None else self.get_random_speed()
        new_sr = int(self.sample_rate * 1 / speed)
        resampled_tensor = resample_frac(tensor, self.sample_rate, new_sr)
        # resampled_tensor = T.Resample(
        #     orig_freq=self.sample_rate,
        #     new_freq=new_sr,
        #     resampling_method="sinc_interp_hann"
        # )(tensor)
        
        if mask is None:
            return resampled_tensor, None
        else:
            return resampled_tensor, torch.nn.functional.interpolate(
                mask, size=resampled_tensor.size(-1), mode="nearest-exact"
            )

class TimeStretch(nn.Module):
    """
    Alternative way to speed up or slow down audio using the phase vocoder algorithm.
    This changes the speed without affecting the pitch directly.
    """
    def __init__(
        self,
        min_rate: float = 0.5,
        max_rate: float = 1.5,
        n_fft: int = 2048,
        hop_length: int = 512
    ):
        super().__init__()
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def __repr__(self):
        return self.__class__.__name__
        
    def get_random_rate(self):
        return torch.FloatTensor(1).uniform_(self.min_rate, self.max_rate).item()
        
    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None, rate: float = None):
        rate = rate if rate is not None else self.get_random_rate()
        
        # Save original shape
        batch_size, channels, length = tensor.shape
        
        # Process each channel in the batch separately
        output = torch.zeros_like(tensor)
        
        for b in range(batch_size):
            for c in range(channels):
                # Create the transformer for this specific rate
                transformer = TorchAudioTimeStretch(
                    fixed_rate=rate,
                    n_freq=self.n_fft // 2 + 1,
                    hop_length=self.hop_length,
                ).to(tensor.device)
                
                # Extract the single channel and reshape for torchaudio TimeStretch
                single_channel = tensor[b, c].unsqueeze(0)
                
                # Time stretch requires a spectrogram as input.
                spec = torch.stft(
                    single_channel,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    return_complex=True
                ) 
                
                # Apply time stretching
                stretched_spec = transformer(spec)
                
                # Convert back to time domain
                stretched_audio = torch.istft(
                    stretched_spec,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    length=int(length / rate),  # Approximate target length
                    window=torch.ones(self.n_fft, device=stretched_spec.device)
                )
                
                # Pad or trim to match original length
                if stretched_audio.size(-1) > length:
                    output[b, c] = stretched_audio[0, :length]
                else:
                    output[b, c, :stretched_audio.size(-1)] = stretched_audio[0]
        
        # Update mask if provided
        if mask is None:
            return output, None
        else:
            return output, torch.nn.functional.interpolate(
                mask, size=output.size(-1), mode="nearest-exact"
            )

class Echo(nn.Module):
    def __init__(
        self, 
        min_volume: float = 0.1,
        max_volume: float = 0.5,
        min_duration: float = 0.1,
        max_duration: float = 0.5,
        sample_rate: int = 16000
    ):
        super().__init__()
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.min_duration = min_duration
        self.max_duration = max_duration 
        self.sample_rate = sample_rate

    def __repr__(self):
        return self.__class__.__name__

    def get_random_params(self):
        duration = torch.FloatTensor(1).uniform_(self.min_duration, self.max_duration)
        volume = torch.FloatTensor(1).uniform_(self.min_volume, self.max_volume)
        return duration.item(), volume.item()

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None, params: tuple = None):
        duration, volume = params if params is not None else self.get_random_params()
        
        n_samples = int(self.sample_rate * duration)
        impulse_response = torch.zeros(n_samples).type(tensor.type()).to(tensor.device)
        impulse_response[0] = 1.0
        impulse_response[int(self.sample_rate * duration) - 1] = volume
        impulse_response = impulse_response.unsqueeze(0).unsqueeze(0)

        reverbed_signal = fft_conv1d(tensor, impulse_response)
        reverbed_signal = (
            reverbed_signal / 
            torch.max(torch.abs(reverbed_signal)) * 
            torch.max(torch.abs(tensor))
        )

        tmp = torch.zeros_like(tensor)
        tmp[..., :reverbed_signal.shape[-1]] = reverbed_signal
        reverbed_signal = tmp

        return reverbed_signal, mask

def generate_pink_noise(length: int) -> torch.Tensor:
    """
    Generate pink noise using Voss-McCartney algorithm with PyTorch.
    """
    num_rows = 16
    array = torch.randn(num_rows, length // num_rows + 1)
    reshaped_array = torch.cumsum(array, dim=1)
    reshaped_array = reshaped_array.reshape(-1)
    reshaped_array = reshaped_array[:length]
    # Normalize
    pink_noise = reshaped_array / torch.max(torch.abs(reshaped_array))
    return pink_noise

class NoiseInjection(nn.Module):
    def __init__(self, min_noise_std: float = 0.0005, max_noise_std: float = 0.0015):
        super().__init__()
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std

    def __repr__(self):
        return self.__class__.__name__
    
    def get_random_std(self):
        return self.min_noise_std + torch.rand(1).item() * (self.max_noise_std - self.min_noise_std)

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None, noise_std: float = None):
        noise_std = noise_std if noise_std is not None else self.get_random_std()
        noise = torch.randn_like(tensor) * noise_std
        noisy_tensor = tensor + noise
        return noisy_tensor, mask

class PinkNoise(nn.Module):
    def __init__(self, min_noise_std: float = 0.005, max_noise_std: float = 0.015):
        super().__init__()
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std

    def __repr__(self):
        return self.__class__.__name__
    
    def get_random_std(self):
        return torch.FloatTensor(1).uniform_(self.min_noise_std, self.max_noise_std)

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None, noise_std: float = None):
        noise_std = noise_std if noise_std is not None else self.get_random_std()
        noise = generate_pink_noise(tensor.shape[-1]) * noise_std
        noise = noise.to(tensor.device)
        noisy_tensor = tensor + noise.unsqueeze(0).unsqueeze(0).expand_as(tensor)
        return noisy_tensor, mask

class LowpassFilter(nn.Module):
    def __init__(self, min_cutoff_freq: float = 2500, max_cutoff_freq: float = 7500, sample_rate: int = 16000):
        super().__init__()
        self.min_cutoff_freq = min_cutoff_freq
        self.max_cutoff_freq = max_cutoff_freq
        self.sample_rate = sample_rate

    def __repr__(self):
        return self.__class__.__name__
    
    def get_random_cutoff(self):
        return torch.FloatTensor(1).uniform_(self.min_cutoff_freq, self.max_cutoff_freq)

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None, cutoff_freq: float = None):
        cutoff_freq = cutoff_freq if cutoff_freq is not None else self.get_random_cutoff()
        filtered = julius.lowpass_filter(tensor, cutoff=cutoff_freq / self.sample_rate)
        return filtered, mask

class HighpassFilter(nn.Module):
    def __init__(self, min_cutoff_freq: float = 250, max_cutoff_freq: float = 750, sample_rate: int = 16000):
        super().__init__()
        self.min_cutoff_freq = min_cutoff_freq
        self.max_cutoff_freq = max_cutoff_freq
        self.sample_rate = sample_rate

    def __repr__(self):
        return self.__class__.__name__
    
    def get_random_cutoff(self):
        return torch.FloatTensor(1).uniform_(self.min_cutoff_freq, self.max_cutoff_freq)

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None, cutoff_freq: float = None):
        cutoff_freq = cutoff_freq if cutoff_freq is not None else self.get_random_cutoff()
        filtered = julius.highpass_filter(tensor, cutoff=cutoff_freq / self.sample_rate)
        return filtered, mask

class BandpassFilter(nn.Module):
    def __init__(
        self,
        min_cutoff_low: float = 150, max_cutoff_low: float = 450,
        min_cutoff_high: float = 4000, max_cutoff_high: float = 10000,
        sample_rate: int = 16000
    ):
        super().__init__()
        self.min_cutoff_low = min_cutoff_low
        self.max_cutoff_low = max_cutoff_low
        self.min_cutoff_high = min_cutoff_high
        self.max_cutoff_high = max_cutoff_high
        self.sample_rate = sample_rate

    def __repr__(self):
        return self.__class__.__name__
    
    def get_random_cutoffs(self):
        low = torch.FloatTensor(1).uniform_(self.min_cutoff_low, self.max_cutoff_low)
        high = torch.FloatTensor(1).uniform_(self.min_cutoff_high, self.max_cutoff_high)
        return low.item(), high.item()

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None, cutoffs: tuple[float, float] = None):
        cutoff_low, cutoff_high = cutoffs if cutoffs is not None else self.get_random_cutoffs()
        filtered = julius.bandpass_filter(
            tensor,
            cutoff_low=cutoff_low / self.sample_rate,
            cutoff_high=cutoff_high / self.sample_rate
        )
        return filtered, mask

class Smooth(nn.Module):
    def __init__(self, min_window_frac: float = 0.001, max_window_frac: float = 0.01, sample_rate: int = 16000):
        super().__init__()
        self.min_window_frac = min_window_frac
        self.max_window_frac = max_window_frac
        self.sample_rate = sample_rate

    def __repr__(self):
        return self.__class__.__name__
    
    def get_random_window_size(self):
        # draw a fraction and convert to integer window size
        frac = torch.FloatTensor(1).uniform_(self.min_window_frac, self.max_window_frac).item()
        return int(frac * self.sample_rate)

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None, window_frac: float = None):
        # determine actual window size from fraction or random
        if window_frac is not None:
            window_size = int(window_frac * self.sample_rate)
        else:
            window_size = self.get_random_window_size()
        # Create a uniform smoothing kernel
        kernel = torch.ones(1, 1, window_size, device=tensor.device) / window_size

        smoothed = fft_conv1d(tensor, kernel)
        # Ensure tensor size is not changed
        tmp = torch.zeros_like(tensor)
        tmp[..., :smoothed.shape[-1]] = smoothed
        smoothed = tmp

        return smoothed, mask

class BoostAudio(nn.Module):
    def __init__(self, min_amount: float = 10, max_amount: float = 30):
        super().__init__()
        self.min_amount = min_amount
        self.max_amount = max_amount

    def __repr__(self):
        return self.__class__.__name__
    
    def get_random_amount(self):
        return torch.FloatTensor(1).uniform_(self.min_amount, self.max_amount).item()

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None, amount: float = None):
        amount = amount if amount is not None else self.get_random_amount()
        boosted = tensor * (1 + amount / 100)
        return boosted, mask

class DuckAudio(nn.Module):
    def __init__(self, min_amount: float = 10, max_amount: float = 30):
        super().__init__()
        self.min_amount = min_amount
        self.max_amount = max_amount

    def __repr__(self):
        return self.__class__.__name__
    
    def get_random_amount(self):
        return torch.FloatTensor(1).uniform_(self.min_amount, self.max_amount).item()

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None, amount: float = None):
        amount = amount if amount is not None else self.get_random_amount()
        ducked = tensor * (1 - amount / 100)
        return ducked, mask

class UpDownResample(nn.Module):
    def __init__(self, sample_rate: int = 16000, intermediate_freq: int = 32000):
        super().__init__()
        self.sample_rate = sample_rate
        self.intermediate_freq = intermediate_freq

    def __repr__(self):
        return self.__class__.__name__
    
    def get_random_intermediate_freq(self):
        return torch.randint(self.sample_rate, self.intermediate_freq * 2, (1,)).item()

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None, intermediate_freq: int = None):
        intermediate_freq = intermediate_freq if intermediate_freq is not None else self.get_random_intermediate_freq()
        orig_shape = tensor.shape
        # upsample
        tensor = resample_frac(tensor, self.sample_rate, intermediate_freq)
        # downsample
        tensor = resample_frac(tensor, intermediate_freq, self.sample_rate)

        assert tensor.shape == orig_shape
        return tensor, mask

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def __repr__(self):
        return self.__class__.__name__
    
    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None, *args, **kwargs):
        return tensor, mask

class MP3Compression(nn.Module):
    def __init__(self, sample_rate: int = 16000, min_bitrate: int = 64, max_bitrate: int = 320, passthrough: bool = True):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_bitrate = min_bitrate
        self.max_bitrate = max_bitrate
        self.passthrough = passthrough

    def __repr__(self):
        return self.__class__.__name__

    def get_random_bitrate(self) -> int:
        """Get random bitrate between min and max."""
        return torch.randint(self.min_bitrate, self.max_bitrate + 1, (1,)).item()

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None, bitrate: int = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply MP3 compression with straight-through estimator if passthrough=True."""
        bitrate = bitrate or self.get_random_bitrate()
        
        if self.passthrough:
            # Straight-through estimator
            compressed = convert_to_format_and_back(tensor.detach(), self.sample_rate, "libmp3lame", bitrate)
            out = tensor + (compressed - tensor).detach()
        else:
            out = convert_to_format_and_back(tensor, self.sample_rate, "libmp3lame", bitrate)
            
        return out, mask

class TimeShift(nn.Module):
    def __init__(self, min_shift_ms: float = 50, max_shift_ms: float = 200, sample_rate: int = 16000):
        super().__init__()
        self.min_shift = int(min_shift_ms * sample_rate / 1000)  # Convert ms to samples
        self.max_shift = int(max_shift_ms * sample_rate / 1000)
        self.sample_rate = sample_rate

    def __repr__(self):
        return self.__class__.__name__
    
    def get_random_shift(self):
        shift = torch.randint(self.min_shift, self.max_shift + 1, (1,)).item()
        return shift if torch.rand(1).item() > 0.5 else -shift

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None, shift_ms: int = None):
        shift = int(shift_ms * self.sample_rate / 1000) if shift_ms is not None else self.get_random_shift()
        # Create output tensor of same size
        shifted = torch.zeros_like(tensor)
        if shift > 0:
            # Shift right
            shifted[..., shift:] = tensor[..., :-shift]
        else:
            # Shift left
            shifted[..., :shift] = tensor[..., -shift:]
            
        return shifted, mask

class TemporalCrop(nn.Module):
    def __init__(self, min_crop_ratio: float = 0.5, max_crop_ratio: float = 0.9):
        """Randomly crop segments and zero-pad to maintain length.
        
        Args:
            min_crop_ratio: Minimum ratio of audio to keep (0-1)
            max_crop_ratio: Maximum ratio of audio to keep (0-1)
        """
        super().__init__()
        self.min_crop_ratio = min_crop_ratio
        self.max_crop_ratio = max_crop_ratio

    def __repr__(self):
        return self.__class__.__name__
    
    def get_random_crop_ratio(self):
        return torch.FloatTensor(1).uniform_(self.min_crop_ratio, self.max_crop_ratio).item()

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None, crop_ratio: float = None):
        crop_ratio = crop_ratio if crop_ratio is not None else self.get_random_crop_ratio()
        total_length = tensor.shape[-1]
        crop_length = int(total_length * crop_ratio)
        start_pos = torch.randint(0, total_length - crop_length + 1, (1,)).item()
        output = tensor[..., start_pos:start_pos + crop_length]
        return output, mask

class DacCompression(nn.Module):

    def __init__(self, sample_rate: int = 24000):
        super().__init__()
        assert sample_rate == 24000, "DacCompression only supports 24kHz sample rate."

        # load dac model
        model_path = dac.utils.download(model_type="24khz")
        self.model = dac.DAC.load(model_path)
        self.model.eval()
    
    def __repr__(self):
        return self.__class__.__name__
    
    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None, *args, **kwargs):
        assert mask is None, "DacCompression does not support masks for now."
        x = self.model.preprocess(tensor, 24_000)
        z, codes, latents, _, _ = self.model.encode(x)
        y = self.model.decode(z)
        return y, mask

class DacCompression16khz(nn.Module):

    def __init__(self, sample_rate: int = 24000):
        super().__init__()

        # load dac model
        model_path = dac.utils.download(model_type="16khz")
        self.model = dac.DAC.load(model_path)
        self.model.eval()

        # Resample to 16kHz
        self.resample = T.Resample(orig_freq=24000, new_freq=16000)
        self.back_resample = T.Resample(orig_freq=16000, new_freq=24000)
    
    def __repr__(self):
        return self.__class__.__name__
    
    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None, *args, **kwargs):
        assert mask is None, "DacCompression does not support masks for now."
        tensor = self.resample(tensor)
        x = self.model.preprocess(tensor, 16_000)
        z, codes, latents, _, _ = self.model.encode(x)
        y = self.model.decode(z)
        y = self.back_resample(y)
        return y, mask


class EncodecCompression(nn.Module):

    def __init__(self, sample_rate: int = 24000):
        super().__init__()
        assert sample_rate == 24000, "DacCompression only supports 24kHz sample rate."
        
        # load the model + processor (for pre-processing the audio) through huggingface
        model_path = "facebook/encodec_24khz"
        self.model = EncodecModel.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.eval()
    
    def __repr__(self):
        return self.__class__.__name__
    
    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None, *args, **kwargs):
        assert mask is None, "Encodec does not support masks for now."

        # or the equivalent with a forward pass
        # print(tensor.shape)
        # np_tensor = tensor.cpu().numpy().squeeze(0)
        # print(np_tensor.shape)
        # inputs = self.processor(raw_audio=np_tensor, sampling_rate=self.processor.sampling_rate, return_tensors="pt")
        # print(inputs["input_values"].shape, inputs["padding_mask"].shape)
        audio_values = self.model(tensor).audio_values
        return audio_values, mask

        


def get_validation_augs(sample_rate: int = 24000, frame_size: int = 1920) -> list[tuple]:
    """
    Get a list of validation augmentations.
    """
    frame_ms = 1000 * frame_size / sample_rate # 80ms
    return [
        (Identity(), [0.0]),
        # (TimeStretch(), [0.5, 1.0, 1.5]),
        (Speed(sample_rate=sample_rate), [0.75, 0.9, 1.0, 1.1, 1.25]),
        (Echo(sample_rate=sample_rate), [(0.1, 0.2), (0.3, 0.5), (0.5, 0.7)]),
        (NoiseInjection(), [0.001, 0.01, 0.05]),
        (PinkNoise(), [0.01, 0.05, 0.1]),
        (LowpassFilter(sample_rate=sample_rate), [1000, 3000, 8000]),
        (HighpassFilter(sample_rate=sample_rate), [100, 500, 1000]),
        (BandpassFilter(sample_rate=sample_rate), [(300, 3000), (500, 5000), (1000, 8000)]),
        (Smooth(), [0.001, 0.005, 0.01]),
        (BoostAudio(), [50, 90]),
        (DuckAudio(), [50, 90]),
        (UpDownResample(sample_rate=sample_rate), [sample_rate, int(sample_rate * 1.5), sample_rate * 2]),
        (MP3Compression(sample_rate=sample_rate), [16, 64, 128]),
        (TimeShift(sample_rate=sample_rate), [frame_ms/8, frame_ms/4, frame_ms/2]),
        (TemporalCrop(), [0.5, 0.7, 0.9]),  # Keep ratios as test strengths
        (DacCompression(), [0.0]),
        (DacCompression16khz(), [0.0]),
        (EncodecCompression(), [0.0]),
    ]

def get_dummy_augs() -> list[tuple]:
    return [(Identity(), [0.0])]

if __name__ == "__main__":

    from pathlib import Path
    import os
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the output directory
    output_dir = Path("outputs/audio_augmentations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load audio file, wget https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3
    audio_path = Path("assets/bria.mp3")
    waveform, sample_rate = torchaudio.load(audio_path)
    new_freq = 24_000
    waveform = T.Resample(orig_freq=sample_rate, new_freq=new_freq)(waveform).unsqueeze(0) # 1 1 s
    sample_rate = new_freq
    
    # cut to 10s
    waveform = waveform[..., :int(sample_rate * 10.3)]
    waveform = waveform.to(device)
    
    # Ensure it's shaped as [batch_size, channels, time]
    if len(waveform.shape) == 2:
        waveform = waveform.unsqueeze(0)  # Add batch dimension
    
    # Define the transformations and their parameter values
    transformations = [
        (TimeStretch, [0.5, 1.0, 1.5]),
        (Speed, [0.5, 1.0, 1.5]),
        (TimeShift, [0.3, 0.5, 10, 40]),  # shifts in milliseconds
        (Echo, [(0.1, 0.2), (0.3, 0.5), (0.5, 0.7)]),
        (NoiseInjection, [0.001, 0.01, 0.05]),
        (PinkNoise, [0.01, 0.05, 0.1]),
        (LowpassFilter, [1000, 3000, 8000]),
        (HighpassFilter, [100, 500, 1000]),
        (BandpassFilter, [(300, 3000), (500, 5000), (1000, 8000)]),
        (Smooth, [0.001, 0.005, 0.01]),
        (BoostAudio, [50, 90]),
        (DuckAudio, [50, 90]),
        (UpDownResample, [24000, 32000, 48000]),
        (MP3Compression, [16, 64, 128]),
        (TemporalCrop, [0.5, 0.7, 0.9]),
        (DacCompression, [0.0]),
        (EncodecCompression, [0.0]),
    ]
    
    # Save the original audio as reference
    original_path = os.path.join(output_dir, "original.wav")
    torchaudio.save(original_path, waveform[0].to("cpu"), sample_rate)
    print(f"Saved {original_path}")

    # Apply each transformation with different strength parameters and save the results
    for transform_class, param_values in transformations:
        # Initialize the transformation
        try:
            transform = transform_class(sample_rate=sample_rate)
        except TypeError:
            transform = transform_class()
        transform = transform.to(waveform.device)
        transform_name = transform_class.__name__

        print(f"Processing {transform_name}...")

        # Apply the transformation with different parameters
        for param_value in param_values:
            
            # Apply the transformation directly with the parameter value
            transformed_waveform, _ = transform(waveform.clone(), None, param_value)

            # Create a descriptive filename
            param_str = f"{param_value}".replace(".", "_").replace(" ", "_").replace(",", "_").replace("(", "").replace(")", "")
            filename = f"{transform.__class__.__name__}_{param_str}.wav"
            output_path = os.path.join(output_dir, filename)

            # Save the transformed audio
            current_sample_rate = sample_rate

            # Ensure the waveform has the correct shape for saving [channels, time]
            if transformed_waveform.dim() == 3:
                save_waveform = transformed_waveform[0]
            else:
                save_waveform = transformed_waveform

            torchaudio.save(output_path, save_waveform.to("cpu"), current_sample_rate)

            print(f"  Saved {filename}")

    print(f"All transformations applied and saved to {output_dir}")