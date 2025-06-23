# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Audioseal evaluation script.

Example usage:
python -m moshi.evals.eval_audioseal \
    --audio_dir "/audio" \
    --output_dir "outputs_audioseal/" \
    --batch_size 4 \
    --nsamples 100 \
    --save_audio 2 \
    --eval_aug True

python -m moshi.evals.eval_audioseal \
    --audio_dir "audio" \
    --output_dir "outputs_audioseal_for_mosnet/" \
    --batch_size 4 \
    --nsamples 500 \
    --save_audio 500 \
    --eval_aug False
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import argparse
import torch
import torchaudio
from tqdm import tqdm
from audioseal import AudioSeal

from moshi.utils import bool_inst
from training import get_cached_audio_files, get_dummy_augs
from training.augmentations import get_validation_augs

def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_tpr_at_fpr_and_save(results_list, output_dir, fpr_target=0.01):
    if not results_list:
        print("No results to process.")
        return

    df = pd.DataFrame(results_list)
    csv_path = os.path.join(output_dir, "audioseal_eval_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved Audioseal evaluation results to {csv_path}")

    results = []
    df = pd.read_csv(csv_path)
    grouped = df.groupby(["aug_name", "strength"])
    
    y_score_negative = df['score_orig']

    for (aug_name, strength), group in grouped:
        # For each row in the group, treat score_wm as a positive sample and score_orig as a negative sample
        y_scores = pd.concat([group['score_wm'], y_score_negative])
        y_true = [1]*len(group) + [0]*len(y_score_negative)

        # Compute FPR and TPR using roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        tpr_at_fpr001 = np.interp(0.01, fpr, tpr)

        # Store results
        results.append({
            'aug_name': aug_name,
            'strength': strength,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'tpr_at_fpr001': tpr_at_fpr001
        })

    # Example: print results
    for r in results:
        print(f"Aug: {r['aug_name']}, Strength: {r['strength']}")
        print(f"TPR at FPR=0.01: {r['tpr_at_fpr001']:.4f}")

def run_audioseal_eval(args):
    device = torch.device(args.device)

    # Load AudioSeal models
    generator = AudioSeal.load_generator(args.generator_model_name).eval().to(device)
    detector = AudioSeal.load_detector(args.detector_model_name).eval().to(device)

    audio_files = get_cached_audio_files(args.audio_dir, extensions=['wav', 'mp3', 'flac', 'ogg'])
    if args.nsamples > 0:
        audio_files = audio_files[:args.nsamples]
    print(f"Processing {len(audio_files)} audio files from {args.audio_dir}")

    results_list = []
    augs_to_run = get_validation_augs(args.target_sr) if args.eval_aug else get_dummy_augs()

    for batch_idx in tqdm(range(0, len(audio_files), args.batch_size), desc="Batch processing"):
        batch_files = audio_files[batch_idx : batch_idx + args.batch_size]
        
        batch_waveforms_orig_list = []
        max_len = 0

        for audio_path in batch_files:
            try:
                waveform, sr = torchaudio.load(audio_path)
                waveform = waveform.to(device)
                if sr != args.target_sr:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=args.target_sr).to(device)
                    waveform = resampler(waveform)
                
                if waveform.shape[0] > 1: # Ensure mono
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                if args.duration_sec is not None:
                    num_samples = int(args.duration_sec * args.target_sr)
                    if waveform.shape[-1] > num_samples:
                        waveform = waveform[..., :num_samples]
                    elif waveform.shape[-1] < num_samples and args.pad_short_audio:
                         # Pad if shorter than duration_sec, only if enabled
                        padding_needed = num_samples - waveform.shape[-1]
                        waveform = torch.nn.functional.pad(waveform, (0, padding_needed))

                batch_waveforms_orig_list.append(waveform)
                if waveform.shape[-1] > max_len:
                    max_len = waveform.shape[-1]
            except Exception as e:
                print(f"Error loading or processing {audio_path}: {e}")
                continue
        
        if not batch_waveforms_orig_list:
            continue

        # Pad to max_len for batching
        current_batch_size = len(batch_waveforms_orig_list)
        batch_waveforms_orig = torch.zeros(current_batch_size, 1, max_len, device=device)
        for ii, wf in enumerate(batch_waveforms_orig_list):
            batch_waveforms_orig[ii, 0, :wf.shape[-1]] = wf

        # Watermark 
        with torch.no_grad():
            delta = generator.get_watermark(batch_waveforms_orig, args.target_sr)
            batch_waveforms_wm = batch_waveforms_orig + args.wm_alpha * delta

        for validation_aug, strengths in augs_to_run:
            validation_aug = validation_aug.to(args.device)
            for strength in strengths:
                aug_name = validation_aug.__class__.__name__
                
                current_waveforms_wm = batch_waveforms_wm
                current_waveforms_orig = batch_waveforms_orig

                current_waveforms_wm, _ = validation_aug(current_waveforms_wm, None, strength)
                current_waveforms_orig, _ = validation_aug(current_waveforms_orig, None, strength)
                
                with torch.no_grad():
                    scores_wm_batch, _ = detector(current_waveforms_wm, args.target_sr)
                    scores_orig_batch, _ = detector(current_waveforms_orig, args.target_sr)
                
                # prob is (B, 2, T_feat), we need prob[:, 1, :] (watermark presence prob)
                # Then take max over T_feat dimension
                mean_score_wm = torch.mean(scores_wm_batch[:, 1, :], dim=-1).cpu().numpy()
                mean_score_orig = torch.mean(scores_orig_batch[:, 1, :], dim=-1).cpu().numpy()

                for idx in range(current_batch_size):
                    global_idx = batch_idx + idx
                    
                    results_list.append({
                        "global_index": global_idx,
                        "audio_file": batch_files[idx],
                        "aug_name": aug_name,
                        "strength": strength,
                        "score_wm": mean_score_wm[idx],
                        "score_orig": mean_score_orig[idx]
                    })

                    if args.save_audio > 0 and global_idx < args.save_audio:
                        audio_output_dir = os.path.join(args.output_dir, "audio_samples")
                        os.makedirs(audio_output_dir, exist_ok=True)
                        
                        # Save watermarked & augmented
                        save_path_wm = os.path.join(audio_output_dir, f"{global_idx:04d}_{aug_name}_{strength}_wm.wav")
                        torchaudio.save(save_path_wm, current_waveforms_wm[idx].cpu(), args.target_sr)

    calculate_tpr_at_fpr_and_save(results_list, args.output_dir, args.fpr_target)


def main():
    parser = argparse.ArgumentParser(description="Audioseal Evaluation Script")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results and outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--nsamples", type=int, default=-1, help="Number of audio files to process (-1 for all)")
    parser.add_argument("--duration_sec", type=float, default=None, help="Maximum duration in seconds for each audio file (None for full length)")
    parser.add_argument("--target_sr", type=int, default=24000, help="Target sample rate for audio files")
    parser.add_argument("--pad_short_audio", type=bool_inst, default=False, help="Pad audio if shorter than duration_sec to meet the duration_sec length.")
    
    parser.add_argument("--save_audio", type=int, default=0, help="Number of (watermarked, augmented) audio examples to save (0 for none)")
    parser.add_argument("--eval_aug", type=bool_inst, default=True, help="Whether to evaluate under augmentations")
    
    parser.add_argument("--generator_model_name", type=str, default="audioseal_wm_16bits", help="Audioseal generator model name or path")
    parser.add_argument("--detector_model_name", type=str, default="audioseal_detector_16bits", help="Audioseal detector model name or path")
    parser.add_argument("--wm_alpha", type=float, default=1.0, help="Factor to scale the watermark signal (delta)") # Defaulting to 1.0, example used 2.0
    
    parser.add_argument("--fpr_target", type=float, default=0.01, help="Target False Positive Rate for TPR calculation (e.g., 0.01 for 1%)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seed_all(args.seed)
    
    # Print args for records
    print("Running Audioseal evaluation with arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    run_audioseal_eval(args)

if __name__ == "__main__":
    main()
