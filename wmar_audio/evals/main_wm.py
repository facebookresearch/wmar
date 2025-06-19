# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Run watermark generation and evaluation:

python -m evals.main_wm \
    --output_dir outputs/ \
    --nsamples 16 \
    --batch_size 1 \
    --temperature 1.0 \
    --steps 200 \
    --wm_method maryland \
    --wm_streams 1 2 3 4 \
    --wm_delta 2.0 \
    --wm_ngram 0 \
    --wm_seed 0 \
    --audio_dir audio_prompts/ \
    --mimi_weight checkpoints/finetunes/mimi_ft.pth
"""

import os
import torch
import numpy as np
import pandas as pd
import argparse
from huggingface_hub import hf_hub_download
import sentencepiece
import sphn
import glob
from tqdm import tqdm
import time
import random
from scipy import stats, special

from moshi.models import loaders, LMGen
from moshi.utils import bool_inst

from training import get_validation_augs, get_dummy_augs
from watermark.engine import get_wm_window_hash, GENERATOR
from watermark.sync import SyncPattern

def get_binomial_pval(x, n, p):
    """
    Calculates the p-value for a one-sided binomial test (greater).
    Args:
        x: The number of successes (e.g., number of matching bits).
        n: The number of trials (e.g., total number of bits).
        p: The hypothesized probability of success under the null hypothesis (e.g., 0.5 for random chance).
    Returns:
        The p-value.
    """
    # p_value = stats.binomtest(x,n,p=p,alternative='greater').pvalue
    p_value = special.betainc(x, 1 + n - x, p)
    return p_value

def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_watermark_scores(wm_stream, ngrams, audio_vocab_size, gamma, wm_seed, device='cpu'):
    """Compute watermark scores for a given stream of tokens.
    
    Args:
        wm_stream: Tensor of token indices to score
        ngrams: Tensor of ngrams used for hashing
        audio_vocab_size: Size of the audio vocabulary
        gamma: Watermark gamma parameter
        device: Device to run computation on
        
    Returns:
        Tuple of:
            - green_mask: Tensor of boolean masks indicating which tokens are "green"
            - to_score_mask: Tensor of boolean masks indicating which tokens are new/unseen
    """
    # Track unseen tokens
    seen_tokens = set()
    green_mask = torch.zeros_like(wm_stream, dtype=torch.bool)
    to_score_mask = torch.zeros_like(wm_stream, dtype=torch.bool)
    for ii, token in enumerate(wm_stream):
        window_hash = get_wm_window_hash(ngrams, wm_seed)  # full of 0 here
        GENERATOR.manual_seed(window_hash[0].item())
        vocab_perm = torch.randperm(audio_vocab_size, generator=GENERATOR)
        greenlist = vocab_perm[:int(gamma * audio_vocab_size)]  # list of tokens
        
        token = token.cpu().item()
        green_mask[ii] = token in greenlist
        if token not in seen_tokens:
            to_score_mask[ii] = 1
            seen_tokens.add(token)
            
    return green_mask, to_score_mask

def run_watermark_eval(args):
    """Generate audio with watermarks and evaluate watermark preservation"""
    # Load models
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    mimi_ori = loaders.get_mimi(args.mimi_weight_ori, args.device)
    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device)
    lm_gen = LMGen(lm)

    # Set generation parameters
    lm_gen.temp = args.temperature
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    
    # Configure watermark
    args.wm_streams = [int(x) for x in args.wm_streams]
    lm_gen.wm = args.wm_method
    lm_gen.wm_ngram = args.wm_ngram
    lm_gen.wm_streams = args.wm_streams
    lm_gen.wm_seed = args.wm_seed
    lm_gen.wm_aux_params["delta"] = args.wm_delta
    lm_gen.wm_aux_params["gamma"] = args.wm_gamma
    audio_vocab_size = lm_gen.lm_model.card
    print(f"Watermarking config: method={lm_gen.wm}, streams={lm_gen.wm_streams}, "
          f"ngram={lm_gen.wm_ngram}, delta={lm_gen.wm_aux_params['delta']}")

    # Configure synchronization if needed
    if args.wm_sync:
        sync_pattern = SyncPattern()
        sync_pattern.to(args.device)

    # Handle prompt preparation if using prompts
    nsamples = args.nsamples
    audio_files = []
    
    if args.use_prompts and args.audio_dir:
        # Find all audio files in directory
        for ext in ['*.wav', '*.mp3', '*.ogg']:
            audio_files.extend(glob.glob(os.path.join(args.audio_dir, ext)))
        
        # Sort and limit number of files if specified
        audio_files = sorted(audio_files)
        if args.nsamples > 0:
            audio_files = audio_files[:args.nsamples]
        
        if len(audio_files) == 0:
            print(f"No audio files found in {args.audio_dir}. Proceeding without prompts.")
        else:
            nsamples = min(nsamples, len(audio_files))
            print(f"Using {nsamples} audio files as prompts")
    
    # Initialize global containers for tokens and watermark results
    tokens_saved = [dict() for _ in range(nsamples)]
    global_watermark_results = []
    
    # Loop over samples in batches
    for batch_start in tqdm(range(0, nsamples, args.batch_size)):
        batch_size = min(args.batch_size, nsamples - batch_start)
        
        # Process prompt codes for current batch if using prompts
        current_prompt_codes = None
        if args.use_prompts and audio_files:
            batch_files = audio_files[batch_start:batch_start + batch_size]
            
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
            
            # Pad prompts to the same length for current batch
            assert len(prompt_codes_list) == batch_size, "Batch size mismatch"

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
            current_prompt_codes = torch.cat(prompt_codes_list, dim=0)
        
        else:
            raise ValueError("No audio files found in the specified directory.")
        
        batch_all_tokens = []
        batch_all_audios = []
        batch_all_texts = [[] for _ in range(batch_size)]
        
        # Continue with generation using current_prompt_codes
        with torch.no_grad():
            with mimi.streaming(batch_size), lm_gen.streaming(batch_size):
                for step in range(args.steps):
                    if current_prompt_codes is not None and step < current_prompt_codes.shape[-1]:
                        codes = current_prompt_codes[:, :, step:step + 1]
                    else:
                        chunk = torch.zeros((batch_size, 1, frame_size), dtype=torch.float, device=args.device)
                        codes = mimi_ori.encode(chunk)
                    
                    # print(codes.shape)
                    tokens = lm_gen.step(codes[:, :, :1], force_epad= (step==start_answer_idx))
                    # tokens = lm_gen.step(codes[:, :, :1], force_epad=False)
                    if tokens is None:
                        continue
                    if current_prompt_codes is not None and step < start_answer_idx:  # min_pad as before
                        continue
                        
                    batch_all_tokens.append(tokens.detach().cpu())
                    
                    # Process text tokens
                    text_tokens = tokens[:, 0, :]  # b 1
                    for idx in range(batch_size):
                        text_token = text_tokens[idx].item()
                        if text_token not in (0, 3):  # Skip special tokens
                            _text = text_tokenizer.id_to_piece(text_token)
                            _text = _text.replace("▁", " ")
                            batch_all_texts[idx].append(_text)
                    
                    # Decode audio tokens
                    audio_tokens = tokens[:, 1:, :]
                    pcms = mimi.decode(audio_tokens)
                    batch_all_audios.append(pcms)
        
        # Concatenate batch tokens and audio segments
        batch_all_audio = torch.cat(batch_all_audios, dim=-1)  # b 1 t
        batch_all_tokens_th = torch.cat(batch_all_tokens, dim=-1)  # b 9 s
        
        # Add synchronization watermark if needed
        if args.wm_sync:
            batch_all_audio = sync_pattern.get_sync_wm(batch_all_audio, alpha=0.5)

        # Prepare validation augmentations, similar to before
        augs = get_validation_augs() if args.eval_aug else get_dummy_augs()
        for aug, _ in augs:
            aug.to(args.device)
        batch_audio_saved = batch_all_audio.clone()
        
        for validation_aug, strengths in augs:
            for strength in strengths:
                # Apply augmentation
                batch_aug_audio, _ = validation_aug(batch_audio_saved, None, strength)

                # Use synchronization watermark if needed
                if args.wm_sync:
                    detection_results = sync_pattern.detect_sync_wm(batch_aug_audio) # b s

                for idx in range(batch_size):
                    synced_audio = batch_aug_audio[idx:idx+1]
                    if args.wm_sync:
                        detection_score =  detection_results[idx].mean()
                        threshold = 0.25
                        if np.abs(detection_score - 0.5) < threshold:
                            speedup, shift = sync_pattern.get_speedup_and_shift(detection_results[idx])
                            synced_audio = sync_pattern.invert(synced_audio, speedup, shift)
                            print(f"Sync watermark detected - Score: {detection_score} - Speedup: {speedup}, Shift: {shift}")

                    # Encode augmented audio
                    tokens_roundtrip = mimi.encode(synced_audio)  # 1 1 s -> 1 8 s

                    # Get watermarked streams
                    wm_tokens = batch_all_tokens_th[0, args.wm_streams, :]  # w s
                    new_wm_streams = [stream - 1 for stream in args.wm_streams if stream > 0]
                    wm_tokens_roundtrip = tokens_roundtrip[0, new_wm_streams, :] if new_wm_streams else None
                    
                    # Calculate watermark stats
                    ngrams = torch.zeros((1, 0), device='cpu')
                    orig_greens, orig_scored = [], []
                    for stream_idx in range(wm_tokens.shape[0]):
                        wm_stream = wm_tokens[stream_idx, :]
                        green_mask, to_score_mask = compute_watermark_scores(wm_stream, ngrams, lm_gen.lm_model.card, args.wm_gamma, args.wm_seed)
                        orig_greens.append((green_mask * to_score_mask).float().sum().item())
                        orig_scored.append(to_score_mask.float().sum().item())
                    
                    greens, scored = [], []
                    if wm_tokens_roundtrip is not None:
                        for stream_idx in range(wm_tokens_roundtrip.shape[0]):
                            wm_stream = wm_tokens_roundtrip[stream_idx, :]
                            green_mask, to_score_mask = compute_watermark_scores(wm_stream, ngrams, lm_gen.lm_model.card, args.wm_gamma, args.wm_seed)
                            greens.append((green_mask * to_score_mask).float().sum().item())
                            scored.append(to_score_mask.float().sum().item())
                    
                    tot_orig_greens = float(sum(orig_greens))
                    tot_orig_scored = float(sum(orig_scored))
                    orig_pval = get_binomial_pval(tot_orig_greens, tot_orig_scored, args.wm_gamma)
                    if wm_tokens_roundtrip is not None:
                        tot_greens = sum(greens)
                        tot_scored = sum(scored)
                        pval = get_binomial_pval(tot_greens, tot_scored, args.wm_gamma)
                    else:
                        pval = None
                    
                    global_idx = batch_start + idx
                    result = {
                        "idx": global_idx,
                        "aug_name": str(validation_aug),
                        "strength": strength,
                        "original_greens": orig_greens,
                        "original_ntoks": wm_tokens.shape[-1],
                        "original_pval": orig_pval,
                        "greens": greens,
                        "scored": scored,
                        "ntoks": wm_tokens_roundtrip.shape[-1] if wm_tokens_roundtrip is not None else 0,
                        "pval": pval,
                    }
                    global_watermark_results.append(result)
                    
                    # Save generated audio if within limit
                    if args.save_audio > 0 and global_idx < args.save_audio:
                        audio_output_dir = os.path.join(args.output_dir, f"audio")
                        os.makedirs(audio_output_dir, exist_ok=True)
                        aug_audio = batch_aug_audio[idx, 0].detach().cpu().numpy().astype(np.float32)
                        sphn.write_wav(
                            os.path.join(audio_output_dir, f'{validation_aug}_{strength}_{global_idx:03d}.wav'),
                            aug_audio,
                            mimi.sample_rate,
                        )
                        if args.wm_sync:
                            synced_audio = synced_audio[0, 0].detach().cpu().numpy().astype(np.float32)
                            sphn.write_wav(
                                os.path.join(audio_output_dir, f'{validation_aug}_{strength}_{global_idx:03d}_synced.wav'),
                                synced_audio,
                                mimi.sample_rate,
                            )

        with open(os.path.join(args.output_dir, f"generated_texts.txt"), "a", encoding="utf-8") as f:
            for idx in range(batch_size):
                f.write(f"{idx + batch_start:04d},{''.join(batch_all_texts[idx])}\n")

    # Save only summary results, remove tokens.pt save
    summary = {
        'config': vars(args),
        'results': global_watermark_results
    }
    torch.save(summary, os.path.join(args.output_dir, 'summary.pt'))

    # Calculate statistics and print pandas DataFrame
    # Keep generated_text for the full df, but exclude from mean calculation
    df_data = [
        {
            "idx": wmr["idx"],
            "aug_name": wmr["aug_name"],
            "strength": str(wmr["strength"]),
            "greens": sum(wmr["greens"]),
            "scored": sum(wmr["scored"]),
            "ntoks": wmr["ntoks"],
            "pval": wmr["pval"],
            "logpval": -np.log10(wmr["pval"]) if wmr["pval"] is not None and wmr["pval"] > 0 else None, # ensure pval > 0 for log
        }
        for wmr in global_watermark_results
    ]

    df = pd.DataFrame(df_data)
    # Select columns for mean aggregation, excluding 'generated_text' and 'idx'
    numeric_cols_for_mean = ["greens", "scored", "ntoks", "pval", "logpval"]
    # Filter df for numeric_cols_for_mean, handling cases where some might be missing
    cols_to_aggregate = [col for col in numeric_cols_for_mean if col in df.columns]
    
    mean_df = df.groupby(["aug_name", "strength"])[cols_to_aggregate].agg("mean")
        
    pd.set_option('display.max_rows', None)
    print(mean_df)
    
    # Optionally, save the full DataFrame with texts to a CSV
    df.to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)
    

def main():
    parser = argparse.ArgumentParser()
    # Basic configuration
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.device_count() else "cpu")
    parser.add_argument("--seed", type=int, default=42424242)
    parser.add_argument("--hf_repo", type=str, default="kyutai/moshiko-pytorch-bf16")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    
    # Prompt configuration
    parser.add_argument("--use_prompts", type=bool_inst, default=True)
    parser.add_argument("--audio_dir", type=str, help="Directory containing audio files for prompts")
    parser.add_argument("--duration_sec", type=float, default=None,
                    help="Maximum duration in seconds for each audio file. None means full length")
    parser.add_argument("--nsamples", type=int, default=-1, help="Number of audio files to process. -1 means all files")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for watermark evaluation")
    
    # Watermarking configuration
    parser.add_argument("--wm_method", type=str, default="maryland", 
                    help="Watermarking method to use")
    parser.add_argument("--wm_streams", nargs='+', default=[1],
                    help="List of stream indices to apply watermarking")
    parser.add_argument("--wm_delta", type=float, default=8.0,
                    help="Watermark delta parameter")
    parser.add_argument("--wm_gamma", type=float, default=0.25,
                    help="Watermark gamma parameter")
    parser.add_argument("--wm_ngram", type=int, default=0,
                    help="Watermark n-gram parameter")
    parser.add_argument("--wm_seed", type=int, default=0)
    parser.add_argument("--wm_sync", type=bool_inst, default=False,
                    help="Whether to use sync watermarking")
    
    # Model weights
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--moshi_weight", type=str)
    parser.add_argument("--mimi_weight", type=str)
    parser.add_argument("--mimi_weight_ori", type=str)
    
    # Analysis and saving options - update save_audio and add save_tokens
    parser.add_argument("--save_audio", type=int, default=10,
                    help="Number of audio files to save (0 = none)")
    parser.add_argument("--save_tokens", type=int, default=0,
                    help="Number of token files to save (0 = none)")
    
    # Add evaluation under augmentations argument:
    parser.add_argument("--eval_aug", type=bool_inst, default=True,
                        help="Evaluate watermark under validation augmentations")
                        
    args = parser.parse_args()

    if args.mimi_weight is None or args.mimi_weight.lower() == "none":
        args.mimi_weight = None
    
    # Download weights if not provided
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    if args.mimi_weight_ori is None:
        args.mimi_weight_ori = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    seed_all(args.seed)
    
    # Run watermark evaluation
    run_watermark_eval(args)

if __name__ == "__main__":
    main()
