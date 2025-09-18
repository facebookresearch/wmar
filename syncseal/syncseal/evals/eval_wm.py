# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate watermark detection accuracy with sync-based geometric inversion.

The pipeline:
1. Embed watermark using baseline method
2. Add synchronization watermark
3. Apply geometric + value-metric augmentations
4. Use sync model to invert geometric transformations
5. Extract watermark using baseline extractor
6. Evaluate bit accuracy, p-value, and corner prediction error

Usage:
python -m syncseal.evals.eval_wm \
    --baseline videoseal \
    --sync_model path/to/syncmodel.jit.pt \
    --dataset sa-1b-full-resized --num_samples 10 \
    --short_edge_size 512 --square_images false \
    --output_dir outputs
"""

import argparse
import os
import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
import torchvision.transforms.functional as TF
from scipy import stats
import pandas as pd

from ..augmentation import get_validation_augs
from ..augmentation.geometricunified import GeometricAugmenter
from ..utils.cfg import setup_dataset, setup_model_from_checkpoint
from ..utils.helpers import warp_image_homography, display_wrap, create_diff_img, compute_corner_error
from ..utils import Timer, bool_inst
from ..models.sync_model import SyncModel, SIFTSyncModel, WAMSyncModel
from .baselines import build_baseline
from .metrics import bit_accuracy, pvalue


@torch.no_grad()
def evaluate_watermark_with_sync(
    baseline_model,
    sync_model,
    dataset: Dataset,
    output_dir: str,
    only_identity: bool = False
):
    """
    Evaluate watermark detection with sync-based geometric inversion.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    csv_path = os.path.join(output_dir, "watermark_sync_metrics.csv")
    print(f"Saving watermark sync metrics to {csv_path}")

    if only_identity:
        validation_augs = get_validation_augs(only_identity=True)
    else:
        validation_augs = get_validation_augs(only_valuemetric=True)

    # Create geometric augmenters
    geometric_augs = {
        'identity':     [0],  # No parameters needed for identity
        'hflip':        [0],  # No parameters needed for horizontal flip
        'rotate':       [5, 10, 20, 30, 45, 90],  # degrees
        'crop':         [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9],  # size ratio
        'perspective':  [0.1, 0.2, 0.3, 0.4, 0.5],  # size ratio
    }
    
    # Create geometric augmenters for each augmentation type and parameter
    geometric_augmenters = {}
    for aug_name, param_list in geometric_augs.items():
        geometric_augmenters[aug_name] = {}
        for param in param_list:
            augs_config = {name: 0 for name in geometric_augs.keys()}
            augs_config[aug_name] = 1.0

            # Create parameters config
            augs_params = {}
            if aug_name == 'rotate':
                augs_params['rotate'] = {'min_angle': param, 'max_angle': param, 'do90': False, 'fill': False}
            elif aug_name == 'crop':
                augs_params['crop'] = {'min_size': param, 'max_size': param}
                if isinstance(sync_model, WAMSyncModel): # WAMSyncModel uses topleft crop
                    augs_params['crop']['topleft'] = True
            elif aug_name == 'perspective':
                augs_params['perspective'] = {'max_distortion_scale': param}
            
            # Create augmenter
            augmenter = GeometricAugmenter(augs_config, augs_params, num_augs=1)
            geometric_augmenters[aug_name][param] = augmenter

    with open(csv_path, "w") as f:
        f.write("index,geom_aug,geom_strength,val_aug,val_strength,bit_accuracy,log_pvalue,corner_error,wm_embed_time,sync_embed_time,sync_detect_time,unwrap_time,wm_detect_time\n")
        
        for idx, batch_items in enumerate(tqdm.tqdm(dataset)):
            imgs, masks = batch_items[0], batch_items[1]
            imgs = imgs.unsqueeze(0)  # c h w -> 1 c h w
            masks = masks.unsqueeze(0) if isinstance(masks, torch.Tensor) else masks
            B, _, H, W = imgs.shape
            center_pt = torch.tensor([(W-1) / 2, (H-1) / 2], device=baseline_model.device)
            imgs = imgs.to(baseline_model.device)
            original_size = imgs.shape[2:]  # (H, W)
            
            # Generate random message for baseline
            msgs_true = baseline_model.get_random_msg(B).to(baseline_model.device)
            
            # Step 1: Embed watermark using baseline method
            wm_embed_timer = Timer(); wm_embed_timer.begin()
            baseline_embedded = baseline_model.embed(imgs, msgs_true)
            imgs_baseline_w = baseline_embedded["imgs_w"]
            if torch.cuda.is_available(): torch.cuda.synchronize()
            wm_embed_time = wm_embed_timer.end()
            
            # Step 2: Add synchronization watermark (if sync model is provided)
            sync_embed_timer = Timer(); sync_embed_timer.begin()
            if sync_model is not None:
                sync_embedded = sync_model.embed(imgs_baseline_w)
                imgs_sync_w = sync_embedded["imgs_w"]
            else:
                # Skip sync watermark if no sync model
                imgs_sync_w = imgs_baseline_w
            if torch.cuda.is_available(): torch.cuda.synchronize()
            sync_embed_time = sync_embed_timer.end()
            
            # Save visualization images for first few samples
            if idx < 5:
                base_name = f"img_{idx:03d}"
                diff_img = create_diff_img(imgs_sync_w[0], imgs[0])
                ori_path = os.path.join(output_dir, base_name + '_0_ori.png')
                baseline_wm_path = os.path.join(output_dir, base_name + '_1_baseline_wm.png')
                sync_wm_path = os.path.join(output_dir, base_name + '_2_sync_wm.png')
                diff_path = os.path.join(output_dir, base_name + '_3_diff.png')
                TF.to_pil_image(imgs[0].cpu().clamp(0, 1)).save(ori_path)
                TF.to_pil_image(imgs_baseline_w[0].cpu().clamp(0, 1)).save(baseline_wm_path)
                TF.to_pil_image(imgs_sync_w[0].cpu().clamp(0, 1)).save(sync_wm_path)
                TF.to_pil_image(diff_img.cpu()).save(diff_path)
            
            # Loop over geometric augmentations
            for geom_aug_name, param_list in geometric_augs.items():
                for geom_param in param_list:
                    # Step 3: Apply geometric augmentation
                    geom_augmenter = geometric_augmenters[geom_aug_name][geom_param]
                    imgs_geom_aug, masks_geom, geom_info = geom_augmenter(imgs_sync_w, masks)
                    
                    # Get original and augmented corner points
                    orig_points = geom_info['startpoints']
                    aug_points = geom_info['endpoints']

                    # Loop over validation augmentations
                    for aug, params in validation_augs:
                        for strength in params:
                            # Apply validation augmentation
                            imgs_final_aug, masks_aug = aug(imgs_geom_aug, masks_geom, strength)
                            
                            # Step 4: Use sync model to detect sync points and invert geometric transformations
                            sync_detect_timer = Timer(); sync_detect_timer.begin()
                            if sync_model is not None:
                                if isinstance(sync_model, SIFTSyncModel):
                                    sync_detected = sync_model.detect(imgs_final_aug, imgs)
                                else:
                                    sync_detected = sync_model.detect(imgs_final_aug)
                                
                                if torch.cuda.is_available(): torch.cuda.synchronize()
                                sync_detect_time = sync_detect_timer.end()
                                
                                # Get corner points and unwarp
                                corner_points = sync_detected["preds"][:, 1:]  # b nparams ...
                                
                                # test for crops. substract all x and y coordinates by the top left point
                                pad = "none"
                                if pad == "left":
                                    corner_points[:, 0::2] = corner_points[:, 0::2] - corner_points[:, 0] - 1  # x coordinates
                                    corner_points[:, 1::2] = corner_points[:, 1::2] - corner_points[:, 1] - 1  # y coordinates
                                elif pad == "center":
                                    center_x = torch.mean(corner_points[:, 0::2])
                                    center_y = torch.mean(corner_points[:, 1::2])
                                    corner_points[:, 0::2] = corner_points[:, 0::2] - center_x
                                    corner_points[:, 1::2] = corner_points[:, 1::2] - center_y
                                else:
                                    pass

                                # Unwarp image with timing
                                unwrap_timer = Timer(); unwrap_timer.begin()
                                # Check if we have a scripted model with unwarp method
                                if hasattr(sync_model, 'unwarp'):
                                    imgs_unwarped = sync_model.unwarp(imgs_final_aug, corner_points, original_size)
                                else:
                                    # Use the helper function for unwarping
                                    img_size = getattr(sync_model, 'img_size', 256)
                                    imgs_unwarped = warp_image_homography(
                                        images=imgs_final_aug, 
                                        corner_points=corner_points,
                                        original_size=original_size,
                                        img_size=img_size
                                    )
                                if torch.cuda.is_available(): torch.cuda.synchronize()
                                unwrap_time = unwrap_timer.end()
                                
                                # Compute corner error using scripted-style computation
                                # Get model image size (default to 256 if not available) 
                                img_size = getattr(sync_model, 'img_size', 256)
                                corner_error, detected_points_array = compute_corner_error(
                                    corner_points, orig_points, original_size, img_size, return_pts=True
                                )
                            else:
                                # No sync model - skip geometric inversion
                                if torch.cuda.is_available(): torch.cuda.synchronize()
                                sync_detect_time = sync_detect_timer.end()
                                imgs_unwarped = imgs_final_aug
                                unwrap_time = 0.0  # No unwrapping performed
                                corner_error = float('nan')  # No corner prediction to evaluate
                            
                            # Step 5: Extract watermark using baseline extractor
                            wm_detect_timer = Timer(); wm_detect_timer.begin()
                            baseline_detected = baseline_model.detect(imgs_unwarped)
                            msgs_pred = baseline_detected["preds"][:, 1:]  # Remove first channel (mask)
                            if torch.cuda.is_available(): torch.cuda.synchronize()
                            wm_detect_time = wm_detect_timer.end()
                            
                            # Step 6: Evaluate metrics
                            # Bit accuracy - metrics.bit_accuracy expects (B, K, H, W) preds and (B, K) targets
                            # But our preds are (B, K) and targets are (B, K)
                            # We need to reshape msgs_pred to match the expected format
                            msgs_pred_reshaped = msgs_pred.unsqueeze(-1).unsqueeze(-1)  # B, K, 1, 1
                            bit_acc_tensor = bit_accuracy(msgs_pred_reshaped, msgs_true)  # Returns tensor of shape (B,)
                            bit_acc_value = bit_acc_tensor.mean().item()  # Average across batch
                            
                            # P-value - metrics.pvalue expects same format
                            pvalue_tensor = pvalue(msgs_pred_reshaped, msgs_true)  # Returns tensor of shape (B,)
                            log_pvalue = torch.log10(pvalue_tensor.mean() + 1e-300).item()  # Take log of average p-value
                            
                            # Save results
                            geom_aug_str = f"{geom_aug_name}_{geom_param}"
                            strength = str(strength).replace(', ', '_')
                            val_aug_str = f"{str(aug).replace(', ', '_')}_{strength}"
                            f.write(f"{idx},{geom_aug_str},{geom_param},{val_aug_str},{strength},{bit_acc_value:.4f},{log_pvalue:.4f},{corner_error:.4f},{wm_embed_time:.6f},{sync_embed_time:.6f},{sync_detect_time:.6f},{unwrap_time:.6f},{wm_detect_time:.6f}\n")
                            f.flush()
                            
                            # Save visualization images for first few samples
                            if idx < 5 and sync_model is not None:
                                img_dir = os.path.join(output_dir, f"img_{idx:03d}")
                                os.makedirs(img_dir, exist_ok=True)
                                
                                # Convert detected points back to list format for display_wrap
                                detected_points_list = detected_points_array.numpy().tolist()
                                # Create visualization showing original watermarked with points, and predicted with points
                                viz_img = display_wrap(
                                    imgs_sync_w[0],  # original watermarked image
                                    imgs_unwarped[0],  # predicted (augmented) image
                                    orig_points,  # original corner points
                                    detected_points_list,  # detected corner points
                                    orig_color="red",
                                    aug_color="blue",
                                    left_title="Original Watermarked",
                                    right_title="Predicted (Augmented)",
                                    save_path=os.path.join(img_dir, f"{geom_aug_str}_{val_aug_str}_points.png")
                                )

    print(f"Saved watermark sync metrics to {csv_path}")
    return results


def load_baseline_model(baseline_path: str, device: str):
    """
    Load baseline watermark model.
    
    Args:
        baseline_path: Path to baseline model or baseline identifier (e.g., 'hidden', 'baseline/wam')
        device: Device to load model on
        
    Returns:
        Baseline model
    """
    if baseline_path.startswith('baseline/'):
        # Extract method name from path
        method = baseline_path.split('/')[-1]
        baseline_model = build_baseline(method)
    else:
        # Assume it's a method name directly
        baseline_model = build_baseline(baseline_path)
    
    baseline_model.to(device)
    baseline_model.eval()
    return baseline_model


def load_sync_model(sync_path: str, device: str):
    """
    Load sync model.
    
    Args:
        sync_path: Path to sync model checkpoint or baseline identifier, or "none" to skip sync
        device: Device to load model on
        
    Returns:
        Sync model or None if sync_path is "none"
    """
    if sync_path.lower() == "none":
        return None
    elif sync_path == "baseline/sift":
        sync_model = SIFTSyncModel(img_size=256, device=device)
        sync_model.eval()
        sync_model.to(device)
    elif sync_path == "baseline/wam":
        sync_model = WAMSyncModel(img_size=256, device=device)
        sync_model.eval()
        sync_model.to(device)
    elif sync_path.endswith('.pt') or sync_path.endswith('.pth'):
        # Check if it's a scripted model
        if 'jit' in sync_path:
            sync_model = torch.jit.load(sync_path, map_location=device)
            sync_model.eval()
        else:
            # Regular checkpoint
            sync_model, _ = setup_model_from_checkpoint(sync_path)
            sync_model.eval()
            sync_model.to(device)
    else:
        raise ValueError(f"Unknown sync model path: {sync_path}")
    
    return sync_model


def main():

    parser = argparse.ArgumentParser(description="Evaluate watermark detection with sync-based geometric inversion")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline watermark method (e.g., 'hidden', 'wam', 'baseline/trustmark')")
    parser.add_argument("--sync_model", type=str, required=True, help="Path to sync model checkpoint, baseline identifier, or 'none' to skip geometric inversion")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    group = parser.add_argument_group("Dataset")
    group.add_argument("--dataset", type=str, default="sa-1b-full-resized", help="Name of the dataset.")
    group.add_argument('--short_edge_size', type=int, default=-1, help='Resizes the short edge of the image to this size at loading time, and keep the aspect ratio. If -1, no resizing.')
    group.add_argument('--square_images', type=bool_inst, default=False, help='If true, after resizing, center crop the image to square. If false, keep the aspect ratio.')
    
    group = parser.add_argument_group("Experiment")
    group.add_argument("--output_dir", type=str, default="output/wm_sync_eval", help="Output directory for CSV files")
    group.add_argument('--only_identity', type=bool_inst, default=False, help='If true, only use identity augmentation for validation')

    group = parser.add_argument_group("Interpolation")
    group.add_argument("--interpolation_mode", type=str, default="bilinear", choices=["nearest", "bilinear", "bicubic", "area"], help="Interpolation mode for resizing")
    group.add_argument("--interpolation_align_corners", type=bool_inst, default=False, help="Align corners for interpolation")
    group.add_argument("--interpolation_antialias", type=bool_inst, default=True, help="Use antialiasing for interpolation")
    args = parser.parse_args()

    # Set random seed for reproducibility
    import random, numpy as np, torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Setup device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    # Load baseline watermark model
    print(f"Loading baseline model: {args.baseline}")
    baseline_model = load_baseline_model(args.baseline, device)

    # Load sync model
    if args.sync_model.lower() == "none":
        print("No sync model - skipping geometric inversion")
        sync_model = None
    else:
        print(f"Loading sync model: {args.sync_model}")
        sync_model = load_sync_model(args.sync_model, device)

    # Setup the dataset
    dataset = setup_dataset(args)
    dataset = Subset(dataset, range(args.num_samples))

    # Run watermark evaluation with sync
    evaluate_watermark_with_sync(
        baseline_model=baseline_model,
        sync_model=sync_model,
        dataset=dataset,
        output_dir=args.output_dir,
        only_identity=args.only_identity
    )

    # Load and print grouped bit accuracy from CSV
    csv_path = os.path.join(args.output_dir, "watermark_sync_metrics.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        grouped = df.groupby(['geom_aug', 'val_aug'])['bit_accuracy'].mean().reset_index()
        grouped.loc[len(grouped)] = ['all', 'all', df['bit_accuracy'].mean()]
        print("\nGrouped Bit Accuracy by Geometric and Value-Metric Augmentation:")
        print(grouped)
    else:
        print(f"CSV file not found at {csv_path}")

if __name__ == "__main__":
    main()
