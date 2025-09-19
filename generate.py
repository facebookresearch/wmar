# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.​

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
from wmar.augmentations.augmentation_manager import AugmentationManager
from wmar.utils.metrics import compute_metric
from wmar.watermarking.gentime_watermark import (
    GentimeWatermark,
    SeedStrategy,
    SplitStrategy,
)

try:
    pass
except ImportError:
    pass
import json

from loguru import logger
from wmar.models.chameleon_wrapper import ChameleonARMMWrapper
from wmar.models.rar_wrapper import RarARMMWrapper
from wmar.models.taming_wrapper import TamingARMMWrapper
from wmar.utils.utils import chw_to_pillow, update_weights
from wmar.watermarking.synchronization import SyncManager


def compute_metrics_and_save_from_batch_log(log, outdir, watermarker, eval_params, cond_indices, compressors=None):
    logger.debug("Computing metrics and saving from batch log")

    for method in log.keys() - ["batch"]:
        orig_codes = log[method]["roundtrips"][0][1]
        orig_imgs = log[method]["roundtrips"][0][2]
        orig_imgs = [chw_to_pillow(img) for img in orig_imgs]
        logger.debug(f"Computing metrics and saving from batch log method: {method}")
        # Transforms: "orig", "roundtrips", "jpeg"....
        for transform in log[method].keys():
            logger.debug(f"  Computing metrics and saving from batch log transform: {transform}")
            # Param: 20, Codes: [B, 256], Imgs: [B, 3, 256, 256]
            # Code was first, Img is the decoded version
            for _, (param, codes, imgs, imgs_nosync) in enumerate(log[method][transform]):
                for i in range(len(codes)):
                    # Extract
                    conditioning = log["batch"][i]
                    if isinstance(conditioning, torch.Tensor):
                        conditioning = conditioning.detach().cpu().item()
                    if isinstance(conditioning, tuple):
                        conditioning = conditioning[0]  # get only index if there's a prompt string too
                    code, orig_code = codes[i], orig_codes[i]
                    img = chw_to_pillow(imgs[i])
                    orig_img = orig_imgs[i]

                    metrics = {}
                    for metric_name in eval_params["metric_names"]:
                        metrics[metric_name] = compute_metric(
                            metric_name,
                            code,
                            orig_code,
                            img,
                            orig_img,
                            watermarker,
                            transform,
                            param,
                            compressors=compressors,
                        )

                    # Save and update count per conditioning
                    cond_index = cond_indices[i]
                    if not eval_params["orig_only"]:
                        curr_outdir = os.path.join(outdir, f"c={conditioning},idx={cond_index}")
                        os.makedirs(curr_outdir, exist_ok=True)
                        img.save(os.path.join(curr_outdir, f"{cond_index:04}_{method}_{transform}_{param}.png"))
                        if imgs_nosync is not None:
                            chw_to_pillow(imgs_nosync[i]).save(
                                os.path.join(curr_outdir, f"{cond_index:04}_{method}_{transform}_{param}_nosync.png")
                            )
                        np.save(os.path.join(curr_outdir, f"{cond_index:04}_{method}_{transform}_{param}.npy"), code)
                        # Save metrics
                        with open(
                            os.path.join(curr_outdir, f"{cond_index:04}_{method}_{transform}_{param}.json"), "w"
                        ) as f:
                            json.dump(metrics, f)
                    else:
                        # New format for FID and all together fixed folder
                        assert param == 0 and transform == "roundtrips"
                        curr_outdir = outdir
                        os.makedirs(curr_outdir, exist_ok=True)
                        os.makedirs(os.path.join(curr_outdir, "images"), exist_ok=True)
                        os.makedirs(os.path.join(curr_outdir, "codes"), exist_ok=True)
                        if len(log.keys()) > 2:
                            img.save(
                                os.path.join(curr_outdir, "images", f"{conditioning}:{cond_index:04}_{method}.png")
                            )
                            np.save(
                                os.path.join(curr_outdir, "codes", f"{conditioning}:{cond_index:04}_{method}.npy"), code
                            )
                        else:
                            img.save(os.path.join(curr_outdir, "images", f"{conditioning}:{cond_index:04}.png"))
                            np.save(os.path.join(curr_outdir, "codes", f"{conditioning}:{cond_index:04}.npy"), code)


@torch.no_grad()
def fill_batch_log(batch_log, key, model, codes, eval_params, sync_manager=None):
    # Decode codes to images
    imgs = model.codes_to_images(codes)  # [b, 3, 256, 256] in [-1, 1]
    if sync_manager is not None:
        imgs = sync_manager.add_sync(imgs)
    logger.debug(f"Filling batch log for {key}")
    batch_log[key] = {}

    # Generate roundtrips, 0 roundtrips = original
    logger.debug(f"Working on roundtrips for {key}")
    batch_log[key]["roundtrips"] = [(0, codes.cpu().numpy(), imgs.cpu().numpy(), None)]
    curr_imgs = imgs
    for T in range(1, eval_params["max_roundtrips"] + 1):
        # Get the next roundtrip
        if sync_manager is not None:
            curr_imgs_nosync = sync_manager.remove_sync(curr_imgs)
            curr_codes = model.images_to_codes(curr_imgs_nosync)
        else:
            curr_imgs_nosync = None
            curr_codes = model.images_to_codes(curr_imgs)
        curr_imgs = model.codes_to_images(curr_codes)
        batch_log[key]["roundtrips"].append(
            (
                T,
                curr_codes.cpu().numpy(),
                curr_imgs.cpu().numpy(),
                curr_imgs_nosync.cpu().numpy() if curr_imgs_nosync is not None else None,
            )
        )  # codes -> image

    for aug_name, aug_fn, aug_params in eval_params["augmentations"]:
        logger.debug(f"Working on {aug_name} for {key}")
        batch_log[key][aug_name] = []
        for aug_param in aug_params:
            # expect inputs in [0, 1] and we have [-1, 1]
            imgs_zero_to_one = imgs / 2.0 + 0.5  # [0, 1] for sure, because we clamped earlier
            aug_imgs_zero_to_one = aug_fn(imgs_zero_to_one, aug_param)
            aug_imgs_zero_to_one = aug_imgs_zero_to_one.clamp(0, 1)  # clamp after
            aug_imgs = aug_imgs_zero_to_one * 2.0 - 1.0  # [-1, 1] again
            if sync_manager is not None:
                aug_imgs_nosync = sync_manager.remove_sync(aug_imgs)
                aug_codes = model.images_to_codes(aug_imgs_nosync)
            else:
                aug_imgs_nosync = None
                aug_codes = model.images_to_codes(aug_imgs)
            batch_log[key][aug_name].append(
                (
                    aug_param,
                    aug_codes.cpu().numpy(),
                    aug_imgs.cpu().numpy(),
                    aug_imgs_nosync.cpu().numpy() if aug_imgs_nosync is not None else None,
                )
            )  # image-->codes


@torch.no_grad()
def generate(
    outdir,
    model,
    all_inputs,
    watermarker,
    eval_params,
    gen_params,
    chunk_id=0,
    num_chunks=1,
    compressors=None,
    sync_manager=None,
):
    batch_size = gen_params["batch_size"]
    batches = []  # last might be smaller
    for i in range(len(all_inputs) // batch_size):
        batches.append(all_inputs[i * batch_size : (i + 1) * batch_size])
    if len(all_inputs) % batch_size != 0:
        batches.append(all_inputs[(len(all_inputs) // batch_size) * batch_size :])

    # Maintain base count per conditioning to save nicely all outputs
    base_count_per_conditioning = {}

    logger.info(f"There are {len(batches)} batches total")
    for batch_idx, batch in enumerate(batches):
        # Get current cond_indices
        cond_indices = []
        for c in batch:
            if isinstance(c, torch.Tensor):
                c = c.detach().cpu().item()
            if isinstance(c, tuple):
                c = c[0]
            if c not in base_count_per_conditioning:
                base_count_per_conditioning[c] = 0
            base_count_per_conditioning[c] += 1
            cond_indices.append(base_count_per_conditioning[c])

        if batch_idx % num_chunks != chunk_id:
            logger.info(f"Skipping batch {batch_idx} due to chunking")
            # Update counts anyways
            continue
        logger.info(f"Not skipping batch {batch_idx} due to chunking")
        logger.info(f"Batch has size {len(batch)}")

        # Sample
        t_start = time.time()
        all_codes = {}
        if watermarker is None:
            codes = model.sample(batch, gen_params, apply_watermark=False)
            all_codes = {str(watermarker): codes}
        elif isinstance(watermarker, GentimeWatermark):
            codes = model.sample(batch, gen_params, apply_watermark=True)
            all_codes = {str(watermarker): codes}
        logger.info(f"Sampling took {time.time() - t_start:.2f} seconds")
        torch.cuda.empty_cache()

        # Fill batch log
        batch_log = dict()
        batch_log["batch"] = batch
        for key, codes in all_codes.items():
            fill_batch_log(batch_log, key, model, codes, eval_params, sync_manager=sync_manager)

        # Compute metrics and save from batch log for this batch, updates counts
        compute_metrics_and_save_from_batch_log(
            batch_log, outdir, watermarker, eval_params, cond_indices=cond_indices, compressors=compressors
        )


def get_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, help="where to save the samples")
    parser.add_argument("--model", type=str, choices=["taming", "chameleon7b", "rar"], help="model to use")
    parser.add_argument("--modelpath", type=str, help="path to the model (see README.md)")
    parser.add_argument("--encoder_ft_ckpt", type=str, help="path to the encoder patch")
    parser.add_argument("--decoder_ft_ckpt", type=str, help="path to the decoder patch")

    # Dataset
    parser.add_argument("--num_samples_per_conditioning", type=int, help="samples per imgnet class or coco prompt")
    parser.add_argument("--conditioning", type=str, help="comma-sep classes (imagenet) or coco txt file")

    # Generation params
    parser.add_argument("--batch_size", type=int, nargs="?", help="batch size", default=10)
    parser.add_argument("--top_k", type=int, nargs="?", help="top-k value to sample with", default=600)
    parser.add_argument("--temperature", type=float, nargs="?", help="temperature value to sample with", default=1.0)
    parser.add_argument("--top_p", type=float, nargs="?", help="top-p value to sample with", default=0.92)

    # Chunking
    parser.add_argument("--chunk_id", type=int, nargs="?", help="chunk id", default=0)
    parser.add_argument("--num_chunks", type=int, nargs="?", help="number of chunks", default=1)

    # Special
    parser.add_argument("--orig_only", type=str2bool, nargs="?", help="orig only", default=False)
    parser.add_argument("--include_neural_compress", type=str2bool, nargs="?", help="include NC", default=True)
    parser.add_argument("--include_diffpure", type=str2bool, nargs="?", help="include diffpure", default=True)

    # Watermark
    parser.add_argument("--wm_method", type=str, nargs="?", help="method", choices=["none", "gentime"])
    parser.add_argument("--wm_seed_strategy", type=str, nargs="?", help="", choices=["fixed", "linear", "spatial"])
    parser.add_argument(
        "--wm_split_strategy", type=str, nargs="?", help="", choices=["rand", "stratifiedrand", "clustering"]
    )
    parser.add_argument("--wm_context_size", type=int, nargs="?", help="context size", default=0)
    parser.add_argument("--wm_delta", type=float, nargs="?", help="wm strength")
    parser.add_argument("--wm_gamma", type=float, nargs="?", help="wm gamma", default=0)
    parser.add_argument("--sync", type=str2bool, default=False)
    parser.add_argument(
        "--syncpath",
        type=str,
    )
    parser.add_argument("--seed", type=int, nargs="?", help="seed", default=42)
    return parser


if __name__ == "__main__":
    # Set up logger 
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    logger.add(sys.stderr, level="ERROR")

    # Load args and set up logging
    sys.path.append(os.getcwd())
    args, _ = get_parser().parse_known_args()
    assert args.outdir, "Output directory is not set"
    logger.info(f"Logging to {args.outdir}")
    os.makedirs(args.outdir, exist_ok=True)

    # Thorough seeding
    seed = args.seed + (1000 * args.chunk_id)  # important for parallel runs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load model
    # Assert that world_size is 1 for the 7b model
    possible_vars = ["SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE"]
    ngpus = next((int(os.environ[var]) for var in possible_vars if var in os.environ), 1)  # Default to 1 if not set
    if "7b" in args.modelpath:
        assert ngpus == 1, f"Chameleon7b model should be run with 1 GPU, but got {ngpus} GPUs."

    if args.model == "taming":
        model = TamingARMMWrapper(args.modelpath)
    elif args.model == "chameleon7b":
        model = ChameleonARMMWrapper(args.modelpath, seed)
    elif args.model == "rar":
        model = RarARMMWrapper(args.modelpath)
    else:
        raise ValueError(f"Model {args.model} not supported")

    # Patch model: whole model or enc and/or dec
    if args.encoder_ft_ckpt is not None and args.encoder_ft_ckpt != "none":
        logger.info(f"Patching encoder from {args.encoder_ft_ckpt}")
        update_weights(model.get_image_tokenizer().encoder, args.encoder_ft_ckpt)
    if args.decoder_ft_ckpt is not None and args.decoder_ft_ckpt != "none":
        logger.info(f"Patching decoder from {args.decoder_ft_ckpt}")
        update_weights(model.get_image_tokenizer().decoder, args.decoder_ft_ckpt)

    # Dataset
    if ".txt" in args.conditioning:
        # File with prompts
        prompts = []
        with open(args.conditioning, "r") as f:
            for idx, line in enumerate(f):
                prompts.append((idx, line.strip()))
        conditionings = prompts
    else:
        # Imagenet classes
        conditionings = [int(c) for c in args.conditioning.split(",")]
    # Repeat each conditioning num_samples_per_conditioning times
    all_inputs = [[c for _ in range(args.num_samples_per_conditioning)] for c in conditionings]
    all_inputs = [item for sublist in all_inputs for item in sublist]
    print(f"All inputs length: {len(all_inputs)}")

    # Watermark
    if "chameleon" in args.model or "rar" in args.model:
        assert (
            args.wm_method in ["none", "gentime"]
            and args.wm_seed_strategy in ["linear", "fixed"]
            and args.wm_split_strategy == "stratifiedrand"
        ), f"Chameleon and RAR models only support none or gentime watermarking with fixed/linear seed and stratifiedrand split"

    vocab_size = model.get_total_vocab_size()  # Passing this for chameleon includes all and should just work
    if args.wm_method == "none":
        watermarker = None
    elif args.wm_method == "gentime":
        watermarker = GentimeWatermark(
            model.get_vq(),
            vocab_size,
            SeedStrategy(args.wm_seed_strategy),
            SplitStrategy(args.wm_split_strategy),
            args.wm_context_size,
            args.wm_delta,
            args.wm_gamma,
            model.device,
        )
    model.set_watermarker(watermarker)

    # Augmentation and metrics setup
    aug_manager = AugmentationManager(args.include_neural_compress, args.include_diffpure, load_augs=True)
    augmentations = aug_manager.augs
    max_roundtrips = 1
    metric_names = ["pvalue", "l0", "psnr", "bpp"]

    if args.orig_only:
        augmentations = []
        max_roundtrips = 0
        metric_names = []

    eval_params = {
        "metric_names": metric_names,
        "augmentations": augmentations,
        "max_roundtrips": max_roundtrips,
        "orig_only": args.orig_only,
    }

    # Generation params
    gen_params = {
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }

    try:
        compressors = aug_manager.compressors
    except Exception as e:
        logger.error(f"Error getting compressors: {e}")
        compressors = None

    # Sync manager
    if not args.sync:
        sync_manager = None
    else:
        sync_manager = SyncManager(args.syncpath, device="cuda")

    # Actually run generation and save outputs
    generate(
        args.outdir,
        model,
        all_inputs,
        watermarker,
        eval_params,
        gen_params,
        chunk_id=args.chunk_id,
        num_chunks=args.num_chunks,
        compressors=compressors,
        sync_manager=sync_manager,
    )

    logger.info("Done.")
