# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Run with:
    python scripts/audioprompts.py \
        --prompt_file /path/to/prompts.txt \
        --output_dir /path/to/output/ --device cuda --chunk_idx 0 --total_chunks 1
"""

import argparse
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoProcessor, SeamlessM4Tv2Model


def read_prompts(prompt_file, chunk_idx, total_chunks):
    with open(prompt_file, "r") as f:
        all_prompts = [line.strip() for line in f.readlines() if line.strip()]

    # Calculate chunk size and boundaries
    chunk_size = len(all_prompts) // total_chunks
    start_idx = chunk_idx * chunk_size
    end_idx = start_idx + chunk_size if chunk_idx < total_chunks - 1 else len(all_prompts)
    prompts = all_prompts[start_idx:end_idx]
    print(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({len(prompts)} prompts)")

    return prompts, start_idx, chunk_size


def generate_audio_prompts(args):
    print(f"Initializing models on device: {args.device}")
    
    # Initialize models
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to(args.device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Read prompts with chunk selection
    prompts, start_idx, _ = read_prompts(args.prompt_file, args.chunk_idx, args.total_chunks)
    print(f"Generating audio for {len(prompts)} prompts starting at index {start_idx}")

    # Process each prompt
    for idx, prompt in enumerate(tqdm(prompts, desc="Generating audio")):
        try:
            # Generate audio
            text_inputs = processor(text=prompt, src_lang="eng", return_tensors="pt").to(args.device)

            with torch.no_grad():
                audio_array = model.generate(**text_inputs, tgt_lang="eng")[0].cpu().numpy().squeeze()

            # Calculate global index
            global_idx = start_idx + idx

            # Save audio with global index
            output_path = os.path.join(args.output_dir, f"prompt_{global_idx:05d}.wav")
            sf.write(output_path, audio_array, samplerate=16000)

            # Save prompt text with global index
            with open(os.path.join(args.output_dir, f"prompt_{global_idx:05d}.txt"), "w") as f:
                f.write(prompt)
                
        except Exception as e:
            print(f"Error processing prompt {global_idx}: {e}")
            continue

    print(f"Audio generation completed. Files saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    # Basic configuration
    parser.add_argument(
        "--output_dir", type=str, default="outputs/prompts", help="Directory to save generated audio prompts"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run generation on"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompts.txt",
        help="File containing text prompts, one per line",
    )

    # Chunk processing arguments
    parser.add_argument("--chunk_idx", type=int, default=0, help="Index of the chunk to process (0-based)")
    parser.add_argument("--total_chunks", type=int, default=1, help="Total number of chunks to split processing into")

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
    if args.chunk_idx < 0:
        raise ValueError("chunk_idx must be non-negative")
    if args.total_chunks <= 0:
        raise ValueError("total_chunks must be positive")
    if args.chunk_idx >= args.total_chunks:
        raise ValueError("chunk_idx must be less than total_chunks")

    # Run generation
    generate_audio_prompts(args)


if __name__ == "__main__":
    main()
