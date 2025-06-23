# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Example run:
torchrun --nproc_per_node=2 -m training.finetune_mimi --debug_slurm true --local_rank 0 \
    --audio_dir /path/to/voxpopuli/ \
    --output_dir output --target_duration 10.0 \
    --learning_rate 1e-5 --epochs 10 --warmup_epochs 1 \
    --save_freq 1 --eval_freq 1 --num_workers 2 \
    --augs '{"identity":1,"lowpass_filter":1,"highpass_filter":1,"noise_injection":1,"pink_noise":1,"mp3_compression":0}' \
    --augs_params '{"lowpass_filter":{"min_cutoff_freq":2000,"max_cutoff_freq":6000},"highpass_filter":{"min_cutoff_freq":200,"max_cutoff_freq":600},"noise_injection":{"min_noise_std":0.005,"max_noise_std":0.015},"pink_noise":{"min_noise_std":0.005,"max_noise_std":0.015},"mp3_compression":{"min_bitrate":64,"max_bitrate":128}}'

To run on a single GPU:
torchrun --nproc_per_node=1 -m moshi.finetune_mimi --debug_slurm true \
"""

import time
import random
import json
import logging
from pathlib import Path
import argparse
import sphn
import numpy as np
from copy import deepcopy
from contextlib import contextmanager
import re

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split

from huggingface_hub import hf_hub_download

from moshi.models import loaders
from moshi.utils import bool_inst

from evals.metrics import calculate_sisnr, calculate_snr, calculate_stoi, calculate_pesq

from .optim import get_cosine_schedule_with_warmup, restart_from_checkpoint
from .dist import is_main_process, get_rank, save_on_master, init_distributed_mode
from .logger import MetricLogger
from .dataloader import AudioDataset, get_audio_dataloader
from .mimi_ft_wrapper import MimiFTWrapper
from .augmenter import Augmenter
from .losses import (
    SISNR,
    LogSTFTMagnitudeLoss,
    MRSTFTLoss,
    SpectralConvergenceLoss,
    STFTLoss,
    MelSpectrogramL1Loss,
    MultiScaleMelSpectrogramLoss,
    TFLoudnessRatio
)

@contextmanager
def TrackGrads(tensors_to_track: dict):
    """Context manager to track gradients of specified tensors."""
    for tensor in tensors_to_track.values():
        if tensor.requires_grad:
            tensor.retain_grad()
    try:
        yield
    finally:
        for name, tensor in tensors_to_track.items():
            if tensor.requires_grad and tensor.grad is not None:
                logging.info(f"{name} grad norm: {tensor.grad.norm().item():.2e}")
            elif tensor.requires_grad:
                logging.warning(f"{name} requires grad but grad is None.")

def seed_all(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_audio_loss(loss_type):
    """Get the audio loss function based on the specified type."""
    if loss_type == "mse":
        return torch.nn.MSELoss()
    elif loss_type == "l1":
        return torch.nn.L1Loss()
    elif loss_type == "sisnr":
        return SISNR(sample_rate=24000)
    elif loss_type == "multi_mel":
        return MultiScaleMelSpectrogramLoss(sample_rate=24000)
    elif loss_type == "stft":
        return STFTLoss()
    elif loss_type == "mrstft":
        return MRSTFTLoss()
    elif loss_type == "tf_loudness":
        return TFLoudnessRatio(sample_rate=24000)
    else:
        raise ValueError(f"Unknown audio loss type: {loss_type}")

def get_code_loss(loss_type):
    """Get the code loss function based on the specified type."""
    if loss_type == "mse":
        return torch.nn.MSELoss()
    elif loss_type == "l1":
        return torch.nn.L1Loss()
    else:
        raise ValueError(f"Unknown code loss type: {loss_type}")

def parse_code_target_indices(code_target_type: str) -> list[int] | None:
    """Parses the code_target_type string into a list of indices."""
    if code_target_type in ["pre_q", "post_q"]:
        return None
    
    indices = set()
    # Split by comma first to handle multiple ranges/indices
    parts = code_target_type.split(',')
    for part in parts:
        part = part.strip()
        # Check for range format like "0-3"
        range_match = re.match(r"(\d+)-(\d+)$", part)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            if start > end:
                raise ValueError(f"Invalid range in code_target_type: {start}-{end}")
            indices.update(range(start, end + 1))
        # Check for single or multiple digits like "0" or "012"
        elif part.isdigit():
            for digit in part:
                indices.add(int(digit))
        else:
            raise ValueError(f"Invalid format in code_target_type: {part}. Use 'pre_q', 'post_q', digits (e.g., '0', '13'), or ranges (e.g., '0-2', '1-3,5').")
            
    if not indices:
         raise ValueError(f"Could not parse any indices from code_target_type: {code_target_type}")

    return sorted(list(indices))


def train_one_epoch(
    model: MimiFTWrapper,
    dataloader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device, 
    epoch: int, 
    steps_per_epoch: int, 
    audio_loss_weight: float, 
    code_loss_weight: float,
    audio_loss_fn: torch.nn.Module,
    code_loss_fn: torch.nn.Module,
    audio_target_type: str = "replica",
    code_target_type: str = "post_q",
    output_dir: str = "output",
    target_sr: int = 24000,
) -> dict:
    model.train()
    header = f"Epoch: [{epoch}]"
    metric_logger = MetricLogger(delimiter="  ")

    # Create directory for augmented samples if running on main process
    if is_main_process():
        aug_samples_dir = Path(output_dir) / f"aug_samples_epoch_{epoch}"
        aug_samples_dir.mkdir(exist_ok=True, parents=True)

    for batch_idx, audio in enumerate(metric_logger.log_every(dataloader, 10, header)):
        if batch_idx >= steps_per_epoch:
            break
        audio = audio.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(audio, epoch=epoch)
        embs_pre_q = outputs["embs_pre_q"] # b d s/f
        codes = outputs["codes"].transpose(0,1) # k b t/f -> b k t/f
        embs_post_q = outputs["embs_post_q"] # b d s/f
        audio_recon = outputs["audio_recon"] # b 1 s
        audio_recon_pred = outputs["audio_recon_pred"] # b 1 s
        recons_embs_pre_q_pred = outputs["recons_embs_pre_q_pred"] # b d s/f
        recons_codes = outputs["recons_codes"].transpose(0,1) # b k t/f
        recons_embs_post_q_pred = outputs["recons_embs_post_q_pred"] # b d s/f
        all_post_q = outputs["all_post_q"] # nq b d s/f 
        recons_all_pre_q = outputs["recons_all_pre_q"] # nq b d s/f
        audio_recon_pred_aug = outputs["audio_recon_pred_aug"] # b 1 s

        # Save augmented audio samples (for first 5 batches)
        if is_main_process() and batch_idx < 5:
            aug_path = aug_samples_dir / f"batch{batch_idx}_{outputs['selected_aug']}.wav"
            sphn.write_wav(str(aug_path), 5*audio_recon_pred_aug[0, 0].detach().cpu().numpy(), target_sr)

        # Calculate losses using provided loss function
        if audio_target_type == "replica":
            audio_target = audio_recon.detach()
        elif audio_target_type == "original":
            audio_target = audio
        else:
            raise ValueError(f"Unknown audio target type: {audio_target_type}")
        audio_loss = audio_loss_fn(audio_recon_pred, audio_target)

        code_target_indices = parse_code_target_indices(code_target_type)

        if code_target_indices is None: # Handle pre_q and post_q cases
            if code_target_type == "post_q":
                code_target = embs_post_q.detach()
                code_pred = recons_embs_post_q_pred
            elif code_target_type == "pre_q":
                code_target = embs_pre_q.detach()
                code_pred = recons_embs_pre_q_pred
            else: # Should not happen due to parsing logic, but for safety
                 raise ValueError(f"Unknown code target type: {code_target_type}")
            code_loss = code_loss_fn(code_pred, code_target)
        else:
            # Select layers based on indices. Target is post-quantization, prediction is pre-quantization.
            code_target = all_post_q[code_target_indices].detach() # num_indices b d s/f
            code_pred = recons_all_pre_q[code_target_indices] # num_indices b d s/f
            losses = []
            for i in range(code_target.shape[0]):
                losses.append(code_loss_fn(code_pred[i], code_target[i]))
            code_loss = torch.mean(torch.stack(losses))


        loss = audio_loss_weight * audio_loss + code_loss_weight * code_loss
    
        # Backward pass.
        track_grads_dict = {
            "recons_embs_pre_q_pred": recons_embs_pre_q_pred,
            "recons_embs_post_q_pred": recons_embs_post_q_pred,
            "audio_recon_pred": audio_recon_pred,
        }
        track_grads_dict = {} # Uncomment to disable grad tracking
        with TrackGrads(track_grads_dict):
            loss.backward()

        optimizer.step()
        scheduler.step()

        # Calculate idempotence rate.
        idempotence_rate = (codes == recons_codes).float().mean(dim=[0, 2])
        method = "all"
        if method == "avg":
            idempotence_rate = {f"idemp_avg": idempotence_rate.mean().item()}
        elif method == "all":
            idempotence_rate = {f"idemp_{ii}": idempotence_rate[ii].item() for ii in range(idempotence_rate.shape[0])}

        # log.
        log_stats = {
            "lr": scheduler.get_last_lr()[0],
            "loss": loss.item(),
            "audio_loss": audio_loss.item(),
            "code_loss": code_loss.item(),
            **idempotence_rate,
        }
        
        metric_logger.update(**log_stats)
    
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def eval_one_epoch(
    model: MimiFTWrapper,
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device, 
    output_dir: Path, 
    epoch: int, 
    target_sr: int,
    audio_loss_fn: torch.nn.Module,
    code_loss_fn: torch.nn.Module,
    audio_target_type: str = "replica",
    code_target_type: str = "pre_q",
) -> dict:
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Eval Epoch: [{epoch}]"

    with torch.no_grad():
        for batch_idx, audio in enumerate(metric_logger.log_every(dataloader, 10, header)):
            audio = audio.to(device)
            outputs = model(audio, epoch=epoch)
            
            audio_recon = outputs["audio_recon"]
            audio_recon_pred = outputs["audio_recon_pred"]
            embs_pre_q = outputs["embs_pre_q"]
            embs_post_q = outputs["embs_post_q"]
            recons_embs_pre_q_pred = outputs["recons_embs_pre_q_pred"]
            recons_embs_post_q_pred = outputs["recons_embs_post_q_pred"]
            codes = outputs["codes"].transpose(0,1) # k b t/f -> b k t/f
            recons_codes = outputs["recons_codes"].transpose(0,1) # b k t/f
            all_post_q = outputs["all_post_q"] # nq b d s/f 
            recons_all_pre_q = outputs["recons_all_pre_q"] # nq b d s/f

            # Calculate losses function
            if audio_target_type == "replica":
                audio_target = audio_recon
            elif audio_target_type == "original":
                audio_target = audio
            else:
                raise ValueError(f"Unknown audio target type: {audio_target_type}")
            audio_loss = audio_loss_fn(audio_recon_pred, audio_target)

            code_target_indices = parse_code_target_indices(code_target_type)

            if code_target_indices is None: # Handle pre_q and post_q cases
                if code_target_type == "post_q":
                    code_target = embs_post_q
                    code_pred = recons_embs_post_q_pred
                elif code_target_type == "pre_q":
                    code_target = embs_pre_q
                    code_pred = recons_embs_pre_q_pred
                else: # Should not happen
                    raise ValueError(f"Unknown code target type: {code_target_type}")
                code_loss = code_loss_fn(code_pred, code_target)
            else: # Handle specific indices
                try:
                    # Select layers based on indices
                    code_target = all_post_q[code_target_indices] # num_indices b d s/f
                    code_pred = recons_all_pre_q[code_target_indices] # num_indices b d s/f
                except IndexError:
                    max_idx = all_post_q.shape[0] - 1
                    raise ValueError(f"Invalid index in code_target_type: {code_target_type}. Max index available: {max_idx}")
                
                # Calculate loss for each selected layer and average
                losses = []
                for i in range(code_target.shape[0]):
                    losses.append(code_loss_fn(code_pred[i], code_target[i]))
                code_loss = torch.mean(torch.stack(losses))


            # Calculate audio metrics
            sisnr_val = calculate_sisnr(audio_recon_pred, audio_target)
            snr_val = calculate_snr(audio_recon_pred, audio_target)
            stoi_val = calculate_stoi(audio_recon_pred, audio_target, sample_rate=target_sr)
            pesq_val = calculate_pesq(audio_recon_pred, audio_target, sample_rate=target_sr)

            # Calculate idempotence rate
            idempotence_rate = (codes == recons_codes).float().mean(dim=[0, 2])
            method = "all"
            if method == "avg":
                idempotence_rate = {f"idemp_avg": idempotence_rate.mean().item()}
            elif method == "all":
                idempotence_rate = {f"idemp_{ii}": idempotence_rate[ii].item() for ii in range(idempotence_rate.shape[0])}

            # Save sample audio at first iteration
            if batch_idx == 0 and is_main_process():
                eval_audio_path = output_dir / f"{epoch:03d}_ori.wav"
                recon_audio_target_path = output_dir / f"{epoch:03d}_target.wav"
                recon_audio_pred_path = output_dir / f"{epoch:03d}_pred.wav"
                
                sphn.write_wav(str(eval_audio_path), 5*audio[0, 0].cpu().numpy(), target_sr)
                sphn.write_wav(str(recon_audio_target_path), 5*audio_recon[0, 0].cpu().numpy(), target_sr)
                sphn.write_wav(str(recon_audio_pred_path), 5*audio_recon_pred[0, 0].cpu().numpy(), target_sr)
                logging.info(f"Saved evaluation audio samples to {eval_audio_path}")

            log_stats = {
                "audio_loss": audio_loss.item(),
                "code_loss": code_loss.item(),
                "sisnr": sisnr_val.item(),
                "snr": snr_val.item(),
                "stoi": stoi_val.item(),
                "pesq": pesq_val.item(), # Use .item(), handle potential NaN from PESQ
                **idempotence_rate,
            }
            metric_logger.update(**log_stats)

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_mimi(
    audio_dir,
    output_dir = "output",
    hf_repo = "kyutai/moshiko-pytorch-bf16",
    batch_size = 16,
    target_sr = 24000,
    target_duration = 5.0,
    num_workers = 16,
    learning_rate = 1e-5,
    warmup_steps = 0,
    epochs = 10,
    steps_per_epoch = 100,
    num_valid = 100,
    device = "cuda" if torch.cuda.is_available() else "cpu",
    save_freq = 10,
    eval_freq = 1,
    audio_loss_type = "mse",
    code_loss_type = "mse",
    audio_loss_weight = 1.0,
    code_loss_weight = 1.0,
    audio_target_type = "replica",
    code_target_type = "pre_q",
    finetune_encoder: bool = False,
    seed = 42424242,
    distributed = False,
    resume_from = None,
    augs: dict = None,
    augs_params: dict = None,
    num_augmentations: int = 1,
    augmentation_start: int = -1,
) -> None:
    """Train the Mimi encoder-decoder model."""
    
    # Set random seed
    seed_all(seed + get_rank())  # Different seed for each process
    
    # Create output directory
    output_dir = Path(output_dir)
    if is_main_process():
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set the device based on local rank in distributed mode
    local_rank = get_rank() if distributed else 0
    if distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    
    # Initialize loss functions
    audio_loss_fn = get_audio_loss(audio_loss_type).to(device)
    code_loss_fn = get_code_loss(code_loss_type).to(device)
    
    # Load Mimi model
    if is_main_process():
        logging.info("Loading Mimi model...")
    mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME) # should be: /home/pfz/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors
    mimi = loaders.get_mimi(mimi_weight, device)
    for param in mimi.quantizer.parameters():
        param.requires_grad = False
    if is_main_process():
        logging.info("Mimi loaded")
    
    # Create a copy of the model for use during training
    mimi_replica = deepcopy(mimi)
    mimi_replica.eval()
    for param in mimi_replica.parameters():
        param.requires_grad = False
    
    # Set up optimizer
    params_to_optimize = [
        {"params": mimi.decoder.parameters(), "lr": learning_rate},
        {"params": mimi.decoder_transformer.parameters(), "lr": learning_rate},
    ]
    if finetune_encoder:
        params_to_optimize.append({"params": mimi.encoder.parameters(), "lr": learning_rate})
        params_to_optimize.append({"params": mimi.encoder_transformer.parameters(), "lr": learning_rate})

    optimizer = AdamW(params_to_optimize)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        training_steps=epochs * steps_per_epoch,
        min_mult=1e-2,
        cycles=0.5,  # decay to min_mult
    )

    # Create augmenter if augmentations are specified
    if augs:
        augs_dict = augs
        augs_params_dict = augs_params if augs_params is not None else {}
        augmenter = Augmenter(augs=augs_dict, augs_params=augs_params_dict, num_augs=num_augmentations, sample_rate=target_sr)
    else:
        augmenter = None

    mimi_ft_wrapper = MimiFTWrapper(mimi, mimi_replica, augmenter=augmenter, augmentation_start=augmentation_start)
    if distributed:
        # mimi_ft_wrapper = DDP(mimi_ft_wrapper, device_ids=[local_rank], find_unused_parameters=True)
        mimi_ft_wrapper = DDP(mimi_ft_wrapper, device_ids=[local_rank])
    
    # Create the full dataset
    full_dataset = AudioDataset(
        audio_dir,
        target_sr=target_sr,
        target_duration=target_duration,
    )

    # Split dataset into train and validation
    total_size = len(full_dataset)
    if num_valid >= total_size:
        raise ValueError(f"num_valid ({num_valid}) must be smaller than the total dataset size ({total_size})")
    train_size = total_size - num_valid
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, num_valid], generator=torch.Generator().manual_seed(seed))

    if is_main_process():
        logging.info(f"Dataset split: Train={len(train_dataset)}, Valid={len(valid_dataset)}")

    # Set up dataloaders
    train_dataloader = get_audio_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True, # Shuffle only for training
        distributed=distributed,
    )
    valid_dataloader = get_audio_dataloader(
        valid_dataset,
        batch_size=batch_size, # Can use a different batch size for validation if needed
        num_workers=num_workers,
        shuffle=False, # No need to shuffle validation data
        distributed=distributed, # Use distributed sampler for validation too if needed
    )
    
    # Optionally resume training
    if resume_from is not None:
        components_to_load = {
            'model': mimi,
        }
        restart_from_checkpoint(resume_from, **components_to_load)

    # Restart some variables where we left them if restarting. Typically because job was killed.
    run_variables = {"epoch": 0}
    restart_from_checkpoint(
        output_dir / "checkpoint.pt",
        run_variables=run_variables,
        model=mimi,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    # Training loop
    for epoch in range(run_variables["epoch"], epochs):
        if is_main_process():
            logging.info(f"Epoch {epoch}/{epochs}")
        
        # Reset sampler for distributed training
        if distributed:
            train_dataloader.sampler.set_epoch(epoch)
            if valid_dataloader.sampler: # Check if validation sampler exists (in distributed mode)
                valid_dataloader.sampler.set_epoch(epoch)
        
        # Train one epoch with audio loss
        train_logs = train_one_epoch(
            model=mimi_ft_wrapper,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            steps_per_epoch=steps_per_epoch,
            audio_loss_weight=audio_loss_weight,
            code_loss_weight=code_loss_weight,
            audio_loss_fn=audio_loss_fn,
            code_loss_fn=code_loss_fn,
            audio_target_type=audio_target_type,
            code_target_type=code_target_type,
            output_dir=output_dir,
            target_sr=target_sr,
        )
        train_logs['epoch'] = epoch

        if (epoch + 1) % eval_freq == 0:
            eval_logs = eval_one_epoch(
                model=mimi_ft_wrapper,
                dataloader=valid_dataloader,
                device=device,
                output_dir=output_dir,
                epoch=epoch,
                target_sr=target_sr,
                audio_loss_fn=audio_loss_fn,
                code_loss_fn=code_loss_fn,
                audio_target_type=audio_target_type,
                code_target_type=code_target_type,
            )
            eval_logs = {f"eval_{k}": v for k, v in eval_logs.items()}
            train_logs = {**train_logs, **eval_logs}
    
        if is_main_process():
            with open(output_dir / 'log.txt', 'a') as f:
                f.write(json.dumps(train_logs) + "\n")
        
        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint.pt"
        save_on_master({
            'epoch': epoch + 1,
            'model': mimi.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, checkpoint_path)
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = output_dir / f"checkpoint{epoch:03d}.pt"
            save_on_master({
                'epoch': epoch + 1,
                'model': mimi.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the Mimi encoder-decoder model")
    parser.add_argument("--hf_repo", type=str, default="kyutai/moshiko-pytorch-bf16", help="HuggingFace repo for model")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")

    # Dataset arguments
    parser.add_argument("--audio_dir", type=str, default="voxpopuli/", help="Directory containing audio files")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--target_sr", type=int, default=24000, help="Target sample rate")
    parser.add_argument("--target_duration", type=float, default=10.0, help="Target audio duration in seconds. Should be multiple of 80ms (s/frame of mimi).")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of dataloader workers")
    parser.add_argument("--num_valid", type=int, default=100, help="Number of validation samples")

    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="Maximum iterations per epoch")
    
    # Losses
    parser.add_argument("--code_loss_type", type=str, default="mse", help="Type of code loss to use")
    parser.add_argument("--audio_loss_type", type=str, default="mrstft", help="Type of audio loss to use")
    parser.add_argument("--audio_loss_weight", type=float, default=1e-3, help="Weight for audio reconstruction loss")
    parser.add_argument("--code_loss_weight", type=float, default=1.0, help="Weight for code reconstruction loss")
    parser.add_argument("--audio_target_type", type=str, default="replica", help="Target for audio loss ('replica' or 'original')")
    parser.add_argument("--code_target_type", type=str, default="pre_q", help="Target for code loss ('post_q', 'pre_q', or indices)")

    # Fine-tuning specific parts
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint to resume training from")
    parser.add_argument("--finetune_encoder", type=bool_inst, default=True, help="Fine-tune the encoder")

    # Misc
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")
    parser.add_argument("--save_freq", type=int, default=10, help="Frequency of saving checkpoints (epochs)")
    parser.add_argument("--eval_freq", type=int, default=1, help="Frequency of evaluation (epochs)")
    parser.add_argument("--seed", type=int, default=42424242, help="Random seed")
    
    # Augmentation arguments
    parser.add_argument("--augmentation_start", type=int, default=-1, help="Epoch to start applying augmentations. -1 means never apply.")
    parser.add_argument("--augs", type=str, default="{}", help="JSON dict of augmentation weights")
    parser.add_argument("--augs_params", type=str, default="{}", help="JSON dict of augmentation parameters")
    parser.add_argument("--num_augmentations", type=int, default=1, help="Number of augmentations to apply sequentially")

    # Distributed training arguments
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--master_port", type=int, default=-1, help="Master port for DDP")
    parser.add_argument("--debug_slurm", type=bool_inst, default=False, help="Debug SLURM setup")
    
    args = parser.parse_args()

    # create some new args and check some args
    args.warmup_steps = args.warmup_epochs * args.steps_per_epoch
    assert (args.target_duration * 1000) % 80 == 0, "Target duration should be a multiple of 80ms (s/frame of mimi)."
    print(args)
    args.augs = args.augs.replace("'", '"') # useful for clutils
    args.augs_params = args.augs_params.replace("'", '"')
    args.augs = json.loads(args.augs)
    args.augs_params = json.loads(args.augs_params)

    # Initialize distributed training if needed and fill some args
    init_distributed_mode(args)    

    # Main optimization loop
    time0 = time.time()
    train_mimi(
        audio_dir = args.audio_dir,
        output_dir = args.output_dir,
        hf_repo = args.hf_repo,
        batch_size = args.batch_size,
        target_sr = args.target_sr,
        target_duration = args.target_duration,
        num_workers = args.num_workers,
        learning_rate = args.learning_rate,
        warmup_steps = args.warmup_steps,
        epochs = args.epochs,
        steps_per_epoch = args.steps_per_epoch,
        device = args.device,
        save_freq = args.save_freq,
        eval_freq = args.eval_freq,
        num_valid = args.num_valid,
        audio_loss_type = args.audio_loss_type,
        code_loss_type = args.code_loss_type,
        audio_loss_weight = args.audio_loss_weight,
        code_loss_weight = args.code_loss_weight,
        audio_target_type = args.audio_target_type,
        code_target_type = args.code_target_type,
        finetune_encoder = args.finetune_encoder,
        seed = args.seed,
        distributed = args.distributed,
        resume_from = args.resume_from,
        augs = args.augs,
        augs_params = args.augs_params,
        num_augmentations = args.num_augmentations,
        augmentation_start = args.augmentation_start,
    )
    elapsed_time = time.time() - time0

    if is_main_process():
        print(f"Training completed. Elapsed time: {elapsed_time / 3600:.2f} hours.")
    

if __name__ == "__main__":
    main()
