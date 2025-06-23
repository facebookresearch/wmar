# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.​

import argparse
import os
import random
import sys

try:
    pass
except ImportError:
    pass
import json

import numpy as np
import torch
import torch.distributed as dist
from deps.taming.util import get_ckpt_path
from huggingface_hub import hf_hub_download
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from wmar.augmentations.geometric import Identity, Rotate, UpperLeftCropWithPadBack
from wmar.augmentations.valuemetric import JPEG, Brightness, GaussianBlur, GaussianNoise
from wmar.models.armm_wrapper import load_model
from wmar.utils.distributed import (
    average_metrics,
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_distributed,
    is_main_process,
)
from wmar.utils.tensorboard import CustomTensorboardWriter
from wmar.utils.utils import (
    CodesOnDiskDataset,
    calculate_gradient_norm,
    compute_and_save_delta,
    get_decoder_dist,
    get_encoder_dist,
    get_model_property,
)


def get_precomputed_imagenet_paths(basedir, n, seed, val_percent):
    paths = [os.path.join(basedir, p) for p in os.listdir(basedir) if p.endswith(".npy")]
    np.random.RandomState(seed).shuffle(paths)
    paths = paths[:n]
    n_train = int(n * (1 - val_percent))
    return paths[:n_train], paths[n_train:]


def log_tensorboard(res_dict):
    rec_unnorm = res_dict["rec_x"] / 2.0 + 0.5
    rec_orig_decoder_unnorm = res_dict["rec_x_orig_decoder"] / 2.0 + 0.5
    rec_maybe_augmented_unnorm = res_dict["rec_x_maybe_augmented"] / 2.0 + 0.5

    # Diff image is abs, in [0, 10]
    diff = 10.0 * torch.abs(res_dict["rec_x"] - res_dict["rec_x_orig_decoder"]) / 2.0
    diff = torch.clamp(diff, 0.0, 1.0)

    tensorboard.add_images("train/images/rec", rec_unnorm, log_step)
    tensorboard.add_images("train/images/rec_orig_decoder", rec_orig_decoder_unnorm, log_step)
    tensorboard.add_images("train/images/rec_maybe_augmented", rec_maybe_augmented_unnorm, log_step)
    tensorboard.add_images("train/images/diff", diff, log_step)


def validate(epoch, vqgan, dataloader_val, augmentations=[], tensorboard=None, device=None, codes=None):
    augmentations = [(Identity, [0])] + augmentations  # Add no augmentation to validation
    for cls, params in augmentations:
        for param in params:
            curr_augmentations = [(cls, [param])]
            curr_key = f"{str(cls)}_{str(param)}"
            vqgan.eval()
            running_stats = {"cnt": 0}
            for batch_idx, batch in enumerate(dataloader_val):
                batch = batch.to(device, non_blocking=True)
                loss, res_dict, log_dict, _ = vqgan(
                    batch, split="val", batch_idx=batch_idx, augmentations=curr_augmentations, augmentations_p=1.0
                )
                log_dict["l0"] = (res_dict["orig_z_indices"] != res_dict["rec_z_indices"]).float().mean()

                for k, v in log_dict.items():
                    running_stats[k] = running_stats.get(k, 0) + v * batch.shape[0]
                running_stats["cnt"] += batch.shape[0]

                if is_main_process() and cls is Identity and batch_idx == 0:
                    log_tensorboard(res_dict)

            # Average locally and if there is a need average across GPUs
            for k, v in running_stats.items():
                if k == "cnt":
                    continue
                running_stats[k] = v / running_stats["cnt"]
            if is_distributed():
                running_stats = average_metrics(running_stats, count=running_stats["cnt"])

            if is_main_process():
                if tensorboard is not None:
                    if cls is not Identity:
                        tensorboard.add_scalars(
                            f"val-{str(cls())}={str(param)}", running_stats, epoch + 1
                        )  # one val logging per epoch
                    else:
                        tensorboard.add_scalars("val", running_stats, epoch + 1)  # one val logging per epoch
                        important = {
                            "val_loss": running_stats["loss"],
                            "val_l0": running_stats["l0"],
                            "val_vqgan_rec_loss": running_stats["vqgan_rec_loss"],
                        }
                        tensorboard.add_scalars("important", important, epoch + 1)
                s = f"[R{get_rank()}] Validation {curr_key}"
                s += f"| Loss: {running_stats['loss']:.5f}"
                s += f"| IdemLoss: {running_stats['idem_loss']:.5f}"
                s += f"| VQGANLoss: {running_stats['vqgan_loss']:.5f}"
                s += f"| L0: {running_stats['l0']:.5f}"
                logger.info(s)

    # Only once in a while check the distance between encs and decs
    if is_main_process():
        enc_dist = get_encoder_dist(vqgan) if get_model_property(vqgan, "use_watermark_encoder") else -1
        dec_dist = get_decoder_dist(vqgan) if backup_orig_decoder else -1
        logger.info(f"[R{get_rank()}] [Val] ENC L2 Distance: {enc_dist:.5f}, DEC L2 Distance: {dec_dist:.5f}")


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["taming", "chameleon7b", "rar"], help="model to use")
    parser.add_argument("--modelpath", type=str, help="path to the model (see README.md)")
    parser.add_argument("--dataset", type=str, help="dataset to use")
    parser.add_argument("--datapath", type=str, help="path to the dataset (precomputed imagenet codes)")
    parser.add_argument("--dataset_size", type=int, help="size of the dataset to subselect")
    parser.add_argument("--mode", type=str, default="newenc-dec")
    parser.add_argument("--nb_epochs", type=int, help="number of epochs")
    parser.add_argument("--augs_schedule", type=str, help="augmentations schedule (e.g., 1,1,4,4)")
    parser.add_argument("--optimizer", type=str, help="optimizer")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--batch_size_per_gpu", type=int, default=10)
    parser.add_argument("--disable_gan", action="store_true")
    parser.add_argument("--idempotence_loss_weight", type=float, help="idempotence loss weight compared to reg")
    parser.add_argument("--idempotence_loss_weight_factor", type=float, help="factor to multiply idem. loss weight by")
    parser.add_argument("--loss", type=str, default="hard-to-soft-with-ae")
    parser.add_argument("--augs", type=str, choices=["none", "all+geom"], help="augmentations to use in training")
    parser.add_argument("--outdir", type=str, help="output directory")

    # DDP params
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--master_port", type=int, default=-1, help="Master port for DDP")
    parser.add_argument("--debug_slurm", type=bool, default=False, help="Debug SLURM setup")

    args, unknown_args = parser.parse_known_args()

    # Set up logging
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    logger.add(sys.stderr, level="ERROR")

    # DDP
    device = "cuda"
    init_distributed_mode(args)
    s = f"<r>[R{get_rank()}] DDP init done. Rank: {args.local_rank} WSz: {get_world_size()}"
    s += f" Port: {args.master_port} | Debug: {args.debug_slurm}</r>"
    logger.opt(colors=True).info(s)
    logger.debug(f"[R{get_rank()}] Args: {vars(args)}")

    # Set random seeds
    if is_distributed():
        seed = args.local_rank * 100
        logger.info(f"[R{get_rank()}] Setting seed to {seed}")
    else:
        seed = 1
        logger.info(f"[R{get_rank()}] Setting seed to {seed} (fixed)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set up tensorboard (will check internally if main process)
    if is_main_process():
        tensorboard = CustomTensorboardWriter(log_dir=f"{args.outdir}/tensorboard")
        logger.info(f"[R{get_rank()}] Logging to {args.outdir}")
        os.makedirs(args.outdir, exist_ok=True)
    else:
        tensorboard = None

    """
        Data
    """
    n_images = args.dataset_size
    val_percent = 0.05

    if args.dataset == "codes-imagenet":
        paths_train, paths_val = get_precomputed_imagenet_paths(
            args.datapath, n=n_images, seed=1, val_percent=val_percent
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    dataset_train = CodesOnDiskDataset(paths_train, transforms.Compose([]))
    dataset_val = CodesOnDiskDataset(paths_val, transforms.Compose([]))

    if is_distributed():
        train_sampler = DistributedSampler(
            dataset_train, num_replicas=args.world_size, rank=args.global_rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            dataset_val, num_replicas=args.world_size, rank=args.global_rank, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size_per_gpu,
        num_workers=10,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=10,
        sampler=val_sampler,
        pin_memory=True,
        drop_last=False,
    )
    logger.info(
        f"[R{get_rank()}] Using {len(dataset_train)} {args.dataset} for training and {len(dataset_val)} for evaluating"
    )

    """
        Model
    """
    vqgan_codebase = "taming" if args.model != "rar" else "rar"

    # Downloads: LPIPS vgg.pth and RAR VQGAN (if training RAR)
    # We fetch on rank 0 and make sure it's accessible by all other GPUs
    if is_main_process():
        get_ckpt_path("vgg_lpips", "checkpoints/lpips")  # hardcoded
        logger.info(f"[R{get_rank()}] Prefetched LPIPS VGG checkpoint -- should be accessible by all other GPUs now")
        if args.model == "rar":
            hf_hub_download(
                repo_id="fun-research/TiTok", filename="maskgit-vqgan-imagenet-f16-256.bin", local_dir=args.modelpath
            )
            logger.info(f"[R{get_rank()}] Got RAR VQGAN -- should be accessible by all other GPUs now")

    if is_distributed():
        dist.barrier()

    # Load VQGAN only
    if args.model == "taming":
        vqgan_config_path = os.path.join(args.modelpath, "configs", "vqgan.yaml")
        vqgan_ckpt_path = os.path.join(args.modelpath, "checkpoints", "vqgan.ckpt")
    elif args.model == "chameleon7b":
        vqgan_config_path = os.path.join(args.modelpath, "tokenizer", "vqgan.yaml")
        vqgan_ckpt_path = os.path.join(args.modelpath, "tokenizer", "vqgan_patched.ckpt")
        # NOTE: make sure to patch it first!
        vqgan = load_model(vqgan_config_path, vqgan_ckpt_path, device="cuda")
    elif args.model == "rar":
        vqgan_config_path = "deps/rar/configs/training/generator/rar.yaml"
        vqgan_ckpt_path = None  # downloaded
    else:
        raise ValueError(f"Model {args.model} not supported")

    vqgan_codebase = "rar" if args.model == "rar" else "taming"
    backup_orig_decoder = True
    do_clone_encoder = "newenc" in args.mode
    vqgan = load_model(
        vqgan_config_path,
        vqgan_ckpt_path,
        clone_encoder=do_clone_encoder,
        backup_orig_decoder=backup_orig_decoder,
        device=device,
        vqgan_codebase=vqgan_codebase,
    )
    logger.info(f"[R{get_rank()}] Loaded VQGAN+Loss model from {args.modelpath}")

    # Set up the model
    vqgan.loss_name = args.loss
    vqgan.loss_weight = args.idempotence_loss_weight
    if vqgan_codebase == "taming" and args.disable_gan:
        vqgan.loss.codebook_weight = 0.0
        vqgan.loss.discriminator_weight = 0.0
        logger.debug(f"[R{get_rank()}] Disabled GAN")

    # Collect params to optimize
    # By default none are trainable, loading functions ensured that
    params_to_optimize = []
    vqgan.use_watermark_encoder = False
    if args.mode == "newenc-dec":
        for param in list(vqgan.watermark_encoder.parameters()) + list(vqgan.decoder.parameters()):
            params_to_optimize.append(param)
        vqgan.watermark_encoder.train()
        vqgan.decoder.train()
        vqgan.use_watermark_encoder = True
    else:
        raise RuntimeError(f"Mode {args.mode} not supported")

    logger.info(f"[R{get_rank()}] #Params to optimize: {len(params_to_optimize)}")
    for param in params_to_optimize:
        param.requires_grad = True

    """
        DDP
    """
    if is_distributed():
        vqgan = DDP(vqgan, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        logger.error(
            f"[R{get_rank()}] Not using the DDP model, either is_dist returned false by mistake or this is a single GPU run"
        )

    """
        Prepare Augmentations
    """
    if args.augs == "none":
        augmentations = []
    elif args.augs == "all+geom":
        augmentations_warmup = []
        augmentations_weak = [
            (JPEG, [90, 80, 70]),
            (GaussianBlur, [1, 3]),
            (GaussianNoise, [0.005, 0.01, 0.015, 0.02]),
            (Brightness, [1.0, 1.1, 1.2]),
            (Rotate, [-1, 1]),
            (UpperLeftCropWithPadBack, [0.8, 0.9]),
        ]
        augmentations_medium = [
            (JPEG, [80, 60, 40]),
            (GaussianBlur, [3, 5]),
            (GaussianNoise, [0.02, 0.04, 0.06]),
            (Brightness, [1.2, 1.3, 1.4]),
            (Rotate, [-3, -2, -1, 1, 2, 3]),
            (UpperLeftCropWithPadBack, [0.5, 0.6, 0.7, 0.8, 0.9]),
        ]
        augmentations_strong = [
            (JPEG, [40, 30, 20]),
            (GaussianBlur, [5, 7, 9]),
            (GaussianNoise, [0.06, 0.08, 0.1]),
            (Brightness, [1.4, 1.7, 2.0]),
            (Rotate, [-3, -2, -1, 1, 2, 3]),
            (UpperLeftCropWithPadBack, [0.5, 0.6, 0.7, 0.8, 0.9]),
        ]
        augs_schedule = [int(x) for x in args.augs_schedule.split(",")]
        assert sum(augs_schedule) == args.nb_epochs, f"Sum of augs schedule {sum(augs_schedule)} != {args.nb_epochs}"
        augmentations_per_epoch = []
        for epochs, augs in zip(
            augs_schedule, [augmentations_warmup, augmentations_weak, augmentations_medium, augmentations_strong]
        ):
            print(f"Adding {epochs} epochs of augs")
            augmentations_per_epoch.extend([augs] * epochs)
        augmentations_per_epoch = {i: augmentations_per_epoch[i] for i in range(len(augmentations_per_epoch))}
    else:
        raise ValueError(f"Augmentations {args.augs} not supported")

    """
        Optimizer
    """
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, betas=(0.9, 0.999))
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    # Reduce every epoch
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    """
        Training
    """
    LOG_EVERY_BATCHES = 50
    log_step = 1

    # Training loop
    for epoch in range(args.nb_epochs):
        augmentations = [] if args.augs == "none" else augmentations_per_epoch[epoch]

        if is_distributed():
            train_sampler.set_epoch(epoch)

        # Validation first
        logger.info(f"[R{get_rank()}] Epoch {epoch+1}/{args.nb_epochs} Validation")
        vqgan.eval()
        with torch.no_grad():
            validate(epoch, vqgan, dataloader_val, augmentations=augmentations, tensorboard=tensorboard, device=device)

        # Training
        vqgan.train()
        running_stats = {"cnt": 0}
        n_batches = len(dataloader_train)
        logger.info(f"[R{get_rank()}] Epoch {epoch+1}/{args.nb_epochs} Training - {n_batches} batches")
        for batch_idx, batch in enumerate(dataloader_train):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()

            # Propagate
            loss, res_dict, log_dict, was_augmented = vqgan(
                batch, split="train", batch_idx=batch_idx, augmentations=augmentations, augmentations_p=0.5
            )
            log_dict["l0"] = (res_dict["orig_z_indices"] != res_dict["rec_z_indices"]).float().mean().item()
            loss.backward()

            # Log gradient norms
            log_dict["wenc_grad_L2"] = (
                calculate_gradient_norm(vqgan, "watermark_encoder")
                if get_model_property(vqgan, "use_watermark_encoder")
                else -1
            )
            log_dict["dec_grad_L2"] = calculate_gradient_norm(vqgan, "decoder")

            # Step
            optimizer.step()

            # Add to running stats
            # Weigh it by batch size so we can average later (DDP: in case different GPUs have different batch sizes)
            for k, v in log_dict.items():
                running_stats[k] = running_stats.get(k, 0) + v * batch.shape[0]
            running_stats["cnt"] += batch.shape[0]

            # Logging
            if is_main_process() and tensorboard is not None:
                # Always log the local batch stats on main for a more granular view
                tensorboard.add_scalars("train-local-batch", log_dict, epoch * n_batches + batch_idx)

            if (batch_idx + 1) % LOG_EVERY_BATCHES == 0 or batch_idx == n_batches - 1:
                # Average locally and if there is a need average across GPUs
                for k, v in running_stats.items():
                    if k != "cnt":
                        running_stats[k] = v / running_stats["cnt"]
                if is_distributed():
                    running_stats = average_metrics(running_stats, count=running_stats["cnt"])

                # Only main prints and logs
                if is_main_process():
                    current_lr = optimizer.param_groups[0]["lr"]
                    s = f"[R{get_rank()}] Batches done: {batch_idx+1}/{len(dataloader_train)}"
                    s += f" (logstep={log_step}) (lr={current_lr:.5f})"
                    s += f"| Loss: {running_stats['loss']:.5f}"
                    s += f"| IdemLoss: {running_stats['idem_loss']:.5f}"
                    s += f"| VQGANLoss: {running_stats['vqgan_loss']:.5f}"
                    s += f"| L0: {running_stats['l0']:.5f}"
                    logger.info(s)
                    if tensorboard is not None:
                        tensorboard.add_scalars("train", running_stats, log_step)
                        important = {
                            "train_loss": running_stats["loss"],
                            "train_l0": running_stats["l0"],
                            "train_vqgan_rec_loss": running_stats["vqgan_rec_loss"],
                        }
                        tensorboard.add_scalars("important", important, log_step)

                    img_outdir = f"{args.outdir}/images_train"
                    os.makedirs(img_outdir, exist_ok=True)
                    log_tensorboard(res_dict)

                    # Sometimes check the distance between encs and decs
                    enc_dist = get_encoder_dist(vqgan) if get_model_property(vqgan, "use_watermark_encoder") else -1
                    dec_dist = get_decoder_dist(vqgan) if backup_orig_decoder else -1
                    logger.info(f"[R{get_rank()}] ENC L2 Distance: {enc_dist:.5f}, DEC L2 Distance: {dec_dist:.5f}")
                    if tensorboard is not None:
                        tensorboard.add_scalar("train/enc_dist", enc_dist, log_step)
                        tensorboard.add_scalar("train/dec_dist", dec_dist, log_step)
                    log_step += 1

                # Everyone should reset running stats
                running_stats = {"cnt": 0}

        # End of epoch, how many log steps did we do?
        logger.info(f"[R{get_rank()}] End of epoch {epoch+1}, last log step is #{log_step}")
        if tensorboard is not None:
            tensorboard.add_scalars("important", {"last_log_step": log_step}, epoch)

        # Sync all processes before starting next epoch
        if is_distributed():
            dist.barrier()

        # Save at end of epoch
        if is_main_process():
            logger.info(f"[R{get_rank()}] Saving weights at end of epoch {epoch+1}")
            if args.mode == "newenc-dec":
                wm_enc = get_model_property(vqgan, "watermark_encoder")
                wm_enc_path = os.path.join(args.outdir, f"watermark_encoder_epoch_{epoch+1}.pth")
                torch.save(wm_enc.state_dict(), wm_enc_path)
                compute_and_save_delta(get_model_property(vqgan, "encoder"), wm_enc_path)

                dec = get_model_property(vqgan, "decoder")
                dec_path = os.path.join(args.outdir, f"decoder_epoch_{epoch+1}.pth")
                torch.save(dec.state_dict(), dec_path)
                compute_and_save_delta(get_model_property(vqgan, "orig_decoder"), dec_path)
            else:
                raise ValueError(f"Mode {args.mode} not supported")

        # Update scheduler
        lr_scheduler.step()

        # Update the local loss weight
        if is_distributed():
            vqgan.module.loss_weight = args.idempotence_loss_weight_factor * vqgan.module.loss_weight
        else:
            vqgan.loss_weight = args.idempotence_loss_weight_factor * vqgan.loss_weight

    # Validation
    logger.info(f"[R{get_rank()}] Done! Doing final validation.")
    vqgan.eval()
    with torch.no_grad():
        validate(
            args.nb_epochs, vqgan, dataloader_val, augmentations=augmentations, tensorboard=tensorboard, device=device
        )

    if is_distributed():
        dist.destroy_process_group()  # needed to gracefully end DDP
