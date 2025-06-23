# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
# Moshi evaluation
python -m evals.token_match \
    --audio_dir audio_prompts/ \
    --mode moshi \
    --output_dir outputs/ \
    --batch_size 1 --nsamples 5\
    --temperature 1.0 \
    --steps 200 \

# Mimi evaluation 
python -m evals.token_match \
    --mode mimi \
    --output_dir outputs/ \
    --nsamples 10 \
    --mimi_weight path/to/mimi.pt \
    --audio_dir voxpopuli/

"""

import os
import numpy as np
import pandas as pd
import argparse
from huggingface_hub import hf_hub_download
import sentencepiece
import sphn
import glob
from tqdm import tqdm

import torch

from moshi.models import loaders, LMGen
from moshi.utils import bool_inst

from training import get_validation_augs, get_dummy_augs, get_cached_audio_files

def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_tm(tokens1, tokens2, per_channel=False):
    """
    Compute token matching rate between two tensors of shape (b, nc, seqlen1) and (b, nc, seqlen2).
    If lengths differ, roll the larger one along the last dimension and return the maximum matching rate.
    Args:
        tokens1 (torch.Tensor): First tensor of shape (b, nc, seqlen1).
        tokens2 (torch.Tensor): Second tensor of shape (b, nc, seqlen2).
        per_channel (bool): If True, compute matching rate for each channel separately.
    """
    bsz, nc, L1 = tokens1.shape
    _, _, L2 = tokens2.shape
    def single_channel_rate(t1, t2):
        # t1 and t2 are (b, L)
        if t1.shape[-1] == t2.shape[-1]:
            return (t1 == t2).float().mean().item()
        # Ensure t1 is the longer sequence
        if t1.shape[-1] < t2.shape[-1]:
            t1, t2 = t2, t1
        best = 0.0
        L_long, L_short = t1.shape[-1], t2.shape[-1]
        # for shift in range(L_long):
        for shift in range(1):
            rolled_seg = t1.roll(shift, dims=-1)[..., :L_short]
            match = (rolled_seg == t2).float().mean().item()
            if match > best:
                best = match
        return best

    if not per_channel:
        # Flatten over channels
        flat1 = tokens1.reshape(bsz, -1)
        flat2 = tokens2.reshape(bsz, -1)
        if flat1.shape[-1] == flat2.shape[-1]:
            return (flat1 == flat2).float().mean().item()
        # Ensure flat1 is longer
        if flat1.shape[-1] < flat2.shape[-1]:
            flat1, flat2 = flat2, flat1
        best = 0.0
        L_long, L_short = flat1.shape[-1], flat2.shape[-1]
        for shift in range(L_long):
            rolled_seg = flat1.roll(shift, dims=-1)[..., :L_short]
            match = (rolled_seg == flat2).float().mean().item()
            if match > best:
                best = match
        return best
    else:
        rates = []
        for ch in range(nc):
            t1 = tokens1[:, ch, :]
            t2 = tokens2[:, ch, :]
            rates.append(single_channel_rate(t1, t2))
        return rates

def save_results(results_list, output_dir):
    import pandas as pd
    df = pd.DataFrame(results_list)
    csv_path = os.path.join(output_dir, "token_match_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved token_match evaluation results to {csv_path}")
    df = df.drop(columns=["audio_file"])
    pd.set_option('display.max_columns', None)
    print(df.groupby(["aug", "strength"]).mean())

def run_moshi_eval(args):
    """Evaluate Moshi generation -> decode -> encode roundtrip"""
    # Load models
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    mimi_ori = loaders.get_mimi(args.mimi_weight_ori, args.device)
    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device)
    lm_gen = LMGen(lm)

    # Set generation parameters
    lm_gen.temp = args.temperature
    frame_size = int(mimi.sample_rate / mimi.frame_rate)

    # Find all audio files in directory
    audio_files = get_cached_audio_files(args.audio_dir, extensions=['wav', 'mp3', 'flac', 'ogg'])
    if args.nsamples > 0:
        audio_files = audio_files[:args.nsamples]
    print(f"Processing {len(audio_files)} audio files from {args.audio_dir}")
    
    results_list = []  # Accumulate token_match results here

    augs = get_validation_augs() if args.eval_aug else get_dummy_augs()
    for aug, _ in augs:
        aug.to(args.device)

    # Process files in batches
    for batch_idx in tqdm(range(0, len(audio_files), args.batch_size), desc="Batch processing"):
        batch_files = audio_files[batch_idx:batch_idx + args.batch_size]
        
        # Process each audio file in current batch
        prompt_codes_list = []
        for audio_path in batch_files:
            # Load and resample audio
            sample_pcm, sample_sr = sphn.read(audio_path, duration_sec=args.duration_sec)
            sample_pcm = sphn.resample(
                sample_pcm, src_sample_rate=sample_sr, dst_sample_rate=mimi.sample_rate
            )
            sample_pcm = torch.tensor(sample_pcm, device=args.device).unsqueeze(0)
            
            with torch.no_grad():
                # Encode audio
                prompt_code = mimi_ori.encode(sample_pcm)
                prompt_codes_list.append(prompt_code)
        
        len_toks = [code.shape[-1] for code in prompt_codes_list]
        max_len_toks = max(len_toks)
        min_pad = 0
        end_prompt_idx = max_len_toks + min_pad
        start_answer_idx = end_prompt_idx
        pad = torch.tensor(lm.zero_token_id, device=prompt_codes_list[0].device).unsqueeze(0).unsqueeze(0)
        prompt_codes_list = [
            torch.cat([
                pad.repeat((1, code.shape[-2], end_prompt_idx - code.shape[-1])), 
                code
            ], dim=-1) 
            for code in prompt_codes_list
        ]
        prompt_codes = torch.cat(prompt_codes_list, dim=0)

        # Generate and compare tokens for this batch
        all_tokens = []
        all_audios = []
        
        with torch.no_grad():
            with mimi_ori.streaming(len(batch_files)), mimi.streaming(len(batch_files)), lm_gen.streaming(len(batch_files)):
                # Generate initial tokens and audio
                for step in tqdm(range(args.steps), desc="Generating", leave=False):
                    if step < prompt_codes.shape[-1]:
                        codes = prompt_codes[:, :, step:step + 1]
                    else:
                        chunk = torch.zeros((len(batch_files), 1, frame_size), dtype=torch.float, device=args.device)
                        codes = mimi_ori.encode(chunk)
                    
                    tokens = lm_gen.step(codes[:, :, :1], force_epad= (step==start_answer_idx))
                    
                    if tokens is None:
                        continue
                        
                    if step < start_answer_idx:
                        continue
                        
                    all_tokens.append(tokens)
                    audio_tokens = tokens[:, 1:, :]
                    pcms = mimi.decode(audio_tokens)
                    all_audios.append(pcms)
            
        # Process and save tokens for this batch
        all_audio_th = torch.cat(all_audios, dim=-1)
        all_orig_tokens = torch.cat(all_tokens, dim=-1)
        
        # # Re-encode generated audio
        # tokens_roundtrip = mimi.encode(all_audio_th)
        
        # Evaluate token_match under augmentations if requested
        for aug, strengths in augs:
            for strength in strengths:
                # Apply augmentation with given strength
                aug_audio, _ = aug(all_audio_th, None, strength)
                tokens_roundtrip_aug = mimi.encode(aug_audio)
                tm_rates = compute_tm(all_orig_tokens[:, 1:], tokens_roundtrip_aug, per_channel=True)
                mean_tm = sum(tm_rates) / len(tm_rates)
    
                # Save tokens for each item in batch
                for idx, audio_path in enumerate(batch_files):
                    global_idx = batch_idx + idx
                    # Save evaluation result dict
                    result_dict = {
                        "global_index": global_idx,
                        "audio_file": audio_path,
                        "aug": f"{aug}",
                        "strength": f"{strength}",
                        "tm_rate": mean_tm,
                    }
                    # Save individual channel rates
                    for i, rate in enumerate(tm_rates):
                        result_dict[f"tm_rate_{i}"] = rate
                    results_list.append(result_dict)
                    if args.save_tokens > 0 and global_idx < args.save_tokens:
                        torch.save({
                            'original': all_orig_tokens[idx, 1:].detach().cpu(),  # Remove text tokens
                            'aug_roundtrip': tokens_roundtrip_aug[idx].detach().cpu(),
                        }, os.path.join(args.output_dir, f'{aug}_{strength}_{global_idx:03d}.pt'))
                    
                    # Save generated audio if requested
                    if global_idx < args.save_audio:
                        audio_output_dir = os.path.join(args.output_dir, f"audio")
                        os.makedirs(audio_output_dir, exist_ok=True)
                        audio = aug_audio[idx, 0].detach().cpu().numpy().astype(np.float32)
                        sphn.write_wav(
                            os.path.join(audio_output_dir, f'{aug}_{strength}_{global_idx:03d}.wav'),
                            audio,
                            mimi.sample_rate,
                        )
    
    # At the end of processing all batches, save evaluation results
    if results_list:
        save_results(results_list, args.output_dir)


def run_mimi_eval(args):
    """Evaluate Mimi encode -> decode -> encode roundtrip"""
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    mimi_ori = loaders.get_mimi(args.mimi_weight_ori, args.device)
    
    # Find all audio files in directory
    audio_files = get_cached_audio_files(args.audio_dir, extensions=['wav', 'mp3', 'flac', 'ogg'])
    if args.nsamples > 0:
        audio_files = audio_files[:args.nsamples]
    print(f"Processing {len(audio_files)} audio files from {args.audio_dir}")
    
    results_list = []  # Accumulate augmentation evaluation results

    # Process files in batches
    for batch_idx in tqdm(range(0, len(audio_files), args.batch_size), desc="Processing batches"):
        batch_files = audio_files[batch_idx:batch_idx + args.batch_size]
        batch_pcms = []
        
        # Load and preprocess batch
        for audio_path in tqdm(batch_files, desc="Loading audio", leave=False):
            sample_pcm, sample_sr = sphn.read(audio_path, duration_sec=args.duration_sec)
            sample_pcm = sphn.resample(
                sample_pcm, src_sample_rate=sample_sr, dst_sample_rate=mimi.sample_rate
            )
            sample_pcm = torch.tensor(sample_pcm, device=args.device).unsqueeze(0)
            batch_pcms.append(sample_pcm)
        
        # Pad to same length
        max_len = max(pcm.shape[-1] for pcm in batch_pcms)
        batch_pcms = [torch.nn.functional.pad(pcm, (0, max_len - pcm.shape[-1])) for pcm in batch_pcms]
        batch_pcms = torch.cat(batch_pcms, dim=0)
        
        with torch.no_grad():
            # First encoding
            orig_tokens = mimi_ori.encode(batch_pcms)
            
            # Decode to audio
            decoded_audio = mimi.decode(orig_tokens)
            
            # Re-encode
            new_tokens = mimi.encode(decoded_audio)
            
        # Evaluate token_match under augmentations if requested
        augs = get_validation_augs() if args.eval_aug else get_dummy_augs()
        for aug, _ in augs:
            aug.to(args.device)
        
        for aug, strengths in augs:
            for strength in strengths:
                aug_audio, _ = aug(decoded_audio, None, strength)
                new_tokens_aug = mimi.encode(aug_audio)
                tm_rates = compute_tm(orig_tokens, new_tokens_aug, per_channel=True)
                mean_tm = sum(tm_rates) / len(tm_rates)
                for idx, audio_path in enumerate(batch_files):
                    global_idx = batch_idx + idx
                    result_dict = {
                        "global_index": global_idx,
                        "audio_file": audio_path,
                        "aug": f"{aug}",
                        "strength": f"{strength}",
                        "tm_rate": mean_tm,
                    }
                    for i, rate in enumerate(tm_rates):
                        result_dict[f"tm_rate_{i}"] = rate
                    results_list.append(result_dict)
                    if args.save_tokens > 0 and global_idx < args.save_tokens:
                        torch.save({
                            'original': orig_tokens[idx].detach().cpu(),
                            'roundtrip': new_tokens_aug[idx].detach().cpu(),
                        }, os.path.join(args.output_dir, f'{aug}_{strength}_{global_idx:03d}.pt'))
                    if global_idx < args.save_audio:
                        audio_output_dir = os.path.join(args.output_dir, "audio")
                        os.makedirs(audio_output_dir, exist_ok=True)
                        audio = aug_audio[idx, 0].detach().cpu().numpy().astype(np.float32)
                        sphn.write_wav(
                            os.path.join(audio_output_dir, f'{aug}_{strength}_{global_idx:03d}.wav'),
                            audio,
                            mimi.sample_rate,
                        )
    
    # At the end, save evaluation results
    if results_list:
        save_results(results_list, args.output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['moshi', 'mimi'], required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.device_count() else "cpu")
    parser.add_argument("--seed", type=int, default=42424242)
    parser.add_argument("--hf_repo", type=str, default="kyutai/moshiko-pytorch-bf16")
    
    # Common args
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Batch size for processing (both modes)")
    parser.add_argument("--duration_sec", type=float, default=None,
                       help="Maximum duration in seconds for each audio file. None means full length")
    parser.add_argument("--save_audio", type=int, default=1,
                       help="Number of audio files to save (0 = none)")
    
    # Moshi specific args
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    
    # Mimi specific args
    parser.add_argument("--audio_dir", type=str, help="Directory containing audio files")
    parser.add_argument("--nsamples", type=int, default=-1, 
                       help="Number of audio files to process. -1 means all files")
    
    # Model weights
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--moshi_weight", type=str)
    parser.add_argument("--mimi_weight_ori", type=str)
    parser.add_argument("--mimi_weight", type=str)
    
    # Add argument to enable evaluation under augmentations
    parser.add_argument("--eval_aug", type=bool_inst, default=True,
                       help="Evaluate token_match rate under validation augmentations")
    # Add parameter to control saving token files (0 = do not save)
    parser.add_argument("--save_tokens", type=int, default=0,
                       help="Number of token files to save (0 = none)")
    
    args = parser.parse_args()
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download weights if not provided
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    if args.mimi_weight_ori is None:
        args.mimi_weight_ori = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    if args.mode == 'moshi':
        if args.moshi_weight is None:
            args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
        if args.tokenizer is None:
            args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    
    # Set random seed
    seed_all(args.seed)
    
    # Run evaluation
    if args.mode == 'moshi':
        run_moshi_eval(args)
    else:
        if args.audio_dir is None:
            parser.error("--audio-dir is required for mimi mode")
        run_mimi_eval(args)

if __name__ == "__main__":
    main()
