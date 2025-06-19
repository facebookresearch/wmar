# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import json
import random
import sphn

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from .dist import get_rank, get_world_size

CACHE_DIR = '.cache/datafiles/'
if not os.path.exists(CACHE_DIR):
    print(f"Cache directory {CACHE_DIR} does not exist. Creating it.")
    os.makedirs(CACHE_DIR, exist_ok=True)

def get_cached_audio_files(audio_dir, extensions=('wav', 'mp3', 'flac', 'ogg')):
    """Find all audio files recursively with caching for faster loading."""
    
    # Create a cache filename based on the audio directory path
    cache_file = os.path.basename(audio_dir.rstrip('/')) + '_' + audio_dir.replace('/', '_') + '.json'
    cache_file = os.path.join(CACHE_DIR, cache_file)
    
    if os.path.exists(cache_file):
        print(f"Loading audio files from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            audio_files = json.load(f)
    else:
        print(f"Finding audio files in {audio_dir}...")
        audio_files = []
        for ext in extensions:
            audio_files.extend(glob.glob(os.path.join(audio_dir, f"**/*.{ext}"), recursive=True))
        audio_files = sorted(audio_files)
        
        print(f"Caching {len(audio_files)} audio paths to {cache_file}")
        with open(cache_file, 'w') as f:
            json.dump(audio_files, f)
    
    return audio_files


class AudioDataset(Dataset):
    """Dataset for loading audio files from a directory recursively."""
    
    def __init__(self, audio_dir, target_sr=24000, target_duration=5.0, extensions=('wav', 'mp3', 'flac', 'ogg')):
        """
        Initialize the dataset.
        
        Args:
            audio_dir: Directory containing audio files
            target_sr: Target sample rate
            target_duration: Target duration in seconds
            extensions: Audio file extensions to look for
        """
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_length = int(target_sr * target_duration)
        
        # Find all audio files recursively using caching
        self.audio_files = get_cached_audio_files(audio_dir, extensions)
        
        print(f"Found {len(self.audio_files)} audio files.")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        
        # Load audio and resample
        audio, sr = sphn.read(audio_file)
        if sr != self.target_sr:
            audio = sphn.resample(audio, src_sample_rate=sr, dst_sample_rate=self.target_sr)
        
        audio = torch.tensor(audio) # C T
        
        # Handle mono/stereo
        if len(audio.shape) > 1 and audio.shape[0] > 1:
            stereo_mode = "sum"
            if stereo_mode == "first": # Handle stereo by taking first channel
                audio = audio[0:1] # 1 T
            elif stereo_mode == "sum": # Handle stereo by summing channels
                audio = audio.sum(dim=0, keepdim=True) # 1 T
        
        # Handle duration
        if audio.shape[1] >= self.target_length:
            audio = audio[:, :self.target_length]
            # # Randomly crop to target length
            # start = random.randint(0, audio.shape[1] - self.target_length)
            # audio = audio[:, start:start + self.target_length]
        else:
            # Pad to target length with nn.functional.pad
            pad_length = self.target_length - audio.shape[1]
            pad = (0, pad_length)
            audio = F.pad(audio, pad, "constant", 0)

        return audio


def get_audio_dataloader(dataset: Dataset, batch_size=16, num_workers=16, shuffle=True, distributed=False):
    """Create a dataloader for a given dataset."""
    # dataset = AudioDataset(audio_dir, target_sr, target_duration) # Removed dataset creation
    
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler
        )
    else:
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )
    return dataloader
