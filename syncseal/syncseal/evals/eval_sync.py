# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate sync point detection accuracy for watermark models.

Usage:
python -m syncseal.evals.eval_sync \
    --checkpoint 'path/to/checkpoint.pth' \
    --dataset sa-1b-full-resized --num_samples 2 \
    --short_edge_size 512 --square_images true \
    --output_dir outputs
"""

import argparse
import os
import tqdm
import pandas as pd
import lpips

import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms.functional as TF
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

from ..augmentation import get_validation_augs
from ..augmentation.geometricunified import GeometricAugmenter
from ..utils.cfg import setup_dataset, setup_model_from_checkpoint
from ..utils.helpers import warp_image_homography, display_wrap, create_diff_img, compute_corner_error
from ..utils import Timer, bool_inst
from ..models.sync_model import SyncModel, SIFTSyncModel, WAMSyncModel



@torch.no_grad()
def evaluate_sync(
    model: SyncModel | SIFTSyncModel | WAMSyncModel,
    dataset: Dataset,
    output_dir: str,
    only_identity: bool = False
):
    """
    For each image, embed watermark, apply geometric aug, detect sync points, revert, and compute pixel error.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    csv_path = os.path.join(output_dir, "sync_metrics.csv")
    quality_csv_path = os.path.join(output_dir, "image_quality_metrics.csv")
    print(f"Saving sync metrics to {csv_path}")
    print(f"Saving image quality metrics to {quality_csv_path}")

    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='alex').to(model.device)

    if only_identity:
        validation_augs = get_validation_augs(only_identity=True)
    else:
        validation_augs = get_validation_augs(only_valuemetric=True)

    # Create geometric augmenters that for every geometric augmentation and parameters
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
                if isinstance(model, WAMSyncModel): # WAMSyncModel uses topleft crop
                    augs_params['crop']['topleft'] = True
            elif aug_name == 'perspective':
                augs_params['perspective'] = {'max_distortion_scale': param}
            
            # Create augmenter
            augmenter = GeometricAugmenter(augs_config, augs_params, num_augs=1)
            geometric_augmenters[aug_name][param] = augmenter

    with open(csv_path, "w") as f, open(quality_csv_path, "w") as f_quality:
        # Added detect_time and unwrap_time columns
        f.write("index,geom_aug,geom_strength,val_aug,val_strength,avg_pixel_diff,detect_time,unwrap_time\n")
        # Added embed_time column
        f_quality.write("index,psnr,ssim,lpips,embed_time\n")
        
        for idx, batch_items in enumerate(tqdm.tqdm(dataset)):

            imgs, masks = batch_items[0], batch_items[1]
            imgs = imgs.unsqueeze(0)  # c h w -> 1 c h w
            masks = masks.unsqueeze(0) if isinstance(masks, torch.Tensor) else masks
            B, _, H, W = imgs.shape
            center_pt = torch.tensor([(W-1) / 2, (H-1) / 2], device=model.device)
            original_size = (H, W)  # Store original size for corner error computation
            imgs = imgs.to(model.device)
            
            # Embed watermark with timing
            embed_timer = Timer(); embed_timer.begin()
            embedded = model.embed(imgs)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            embed_time = embed_timer.end()
            imgs_w = embedded["imgs_w"]
            
            # Compute image quality metrics between original and watermarked (once per image)
            orig_img = imgs.clamp(0, 1)
            watermarked_img = imgs_w.clamp(0, 1)
            psnr = peak_signal_noise_ratio(watermarked_img, orig_img, data_range=1.0).item()
            ssim = structural_similarity_index_measure(watermarked_img, orig_img, data_range=1.0).item()
            lpips_score = lpips_model(watermarked_img, orig_img).item()
            f_quality.write(f"{idx},{psnr:.4f},{ssim:.4f},{lpips_score:.4f},{embed_time:.6f}\n")
            f_quality.flush()
            
            # Save original, watermarked, and difference images for first 5 samples
            if idx < 5:
                base_name = f"img_{idx:03d}"
                diff_img = create_diff_img(watermarked_img[0], orig_img[0])
                ori_path = os.path.join(output_dir, base_name + '_0_ori.png')
                wm_path = os.path.join(output_dir, base_name + '_1_wm.png')
                diff_path = os.path.join(output_dir, base_name + '_2_diff.png')
                TF.to_pil_image(orig_img[0].cpu()).save(ori_path)
                TF.to_pil_image(watermarked_img[0].cpu()).save(wm_path)
                TF.to_pil_image(diff_img.cpu()).save(diff_path)
            
            # Loop over geometric augmentations
            for geom_aug_name, param_list in geometric_augs.items():
                for geom_param in param_list:
                    # Apply geometric augmentation
                    geom_augmenter = geometric_augmenters[geom_aug_name][geom_param]
                    imgs_geom_aug, _, geom_info = geom_augmenter(imgs_w, masks)
                    
                    # Get original and augmented corner points
                    orig_points = geom_info['startpoints']
                    aug_points = geom_info['endpoints']

                    # Loop over validation augmentations
                    for aug, params in validation_augs:
                        for strength in params:
                            # Apply validation augmentation
                            imgs_final_aug, masks_aug = aug(imgs_geom_aug, masks, strength)
                            
                            # Detect sync points with timing
                            detect_timer = Timer(); detect_timer.begin()
                            try:
                                if isinstance(model, SIFTSyncModel):
                                    detected = model.detect(imgs_final_aug, imgs)
                                else:
                                    detected = model.detect(imgs_final_aug)
                            except Exception as e:
                                detected = {"preds": torch.tensor([[ 0., -1., -1.,  1., -1.,  1.,  1., -1.,  1.]], device=model.device)}
                            if torch.cuda.is_available(): torch.cuda.synchronize()
                            detect_time = detect_timer.end()
                            corner_points = detected["preds"][:, 1:]  # b nparams ...
                            
                            # Compute corner error using scripted-style computation
                            # Get model image size (default to 256 if not available)
                            img_size = getattr(model, 'img_size', 256)
                            pixel_diff, detected_points_array = compute_corner_error(
                                corner_points, orig_points, original_size, img_size, return_pts=True, device=model.device
                            )
                            
                            # Warp image back using detected points with timing (still needed for visualization)
                            unwrap_timer = Timer(); unwrap_timer.begin()
                            warped_image = warp_image_homography(
                                images=imgs_final_aug, 
                                corner_points=corner_points,
                                original_size=original_size,
                                img_size=img_size,
                            )
                            if torch.cuda.is_available(): torch.cuda.synchronize()
                            unwrap_time = unwrap_timer.end()

                            # Save results
                            geom_aug_str = f"{geom_aug_name}_{geom_param}"
                            strength = str(strength).replace(', ', '_')
                            val_aug_str = f"{str(aug).replace(', ', '_')}_{strength}"
                            f.write(f"{idx},{geom_aug_str},{geom_param},{val_aug_str},{strength},{pixel_diff:.4f},{detect_time:.6f},{unwrap_time:.6f}\n")
                            f.flush()
                            
                            # Save visualization images for first few samples
                            if idx < 5:
                                img_dir = os.path.join(output_dir, f"img_{idx:03d}")
                                os.makedirs(img_dir, exist_ok=True)
                                
                                # Convert detected points back to list format for display_wrap
                                detected_points_list = detected_points_array.numpy().tolist()
                                
                                # Create visualization showing original -> augmented -> unwarped
                                viz_img = display_wrap(
                                    imgs_w[0],  # original watermarked
                                    warped_image[0],  # unwarped (should match original)
                                    orig_points,  # original corner points
                                    detected_points_list,  # detected corner points
                                    orig_color="red",
                                    aug_color="blue",
                                    left_title="Original Watermarked",
                                    right_title="Unwarped (Detected)",
                                    save_path=os.path.join(img_dir, f"{geom_aug_str}_{val_aug_str}_sync.png")
                                )

    print(f"Saved sync metrics to {csv_path}")
    print(f"Saved image quality metrics to {quality_csv_path}")
    return results


def main():

    parser = argparse.ArgumentParser(description="Evaluate sync point detection accuracy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--scaling_w", type=str, default=None, help="Override scaling_w factor for watermark blending (float or 'none')")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    group = parser.add_argument_group("Dataset")
    group.add_argument("--dataset", type=str, default="sa-1b-full-resized", help="Name of the dataset.")
    group.add_argument('--short_edge_size', type=int, default=-1, help='Resizes the short edge of the image to this size at loading time, and keep the aspect ratio. If -1, no resizing.')
    group.add_argument('--square_images', type=bool_inst, default=False, help='If true, after resizing, center crop the image to square. If false, keep the aspect ratio.')
    
    group = parser.add_argument_group("Experiment")
    group.add_argument("--output_dir", type=str, default="output/sync_eval", help="Output directory for CSV files")
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

    # Setup the model
    if args.checkpoint == "baseline/sift":
        model = SIFTSyncModel(img_size=args.short_edge_size if args.short_edge_size > 0 else 256, device=args.device)
        model.eval()
        model.to(args.device)
        config = None
    elif args.checkpoint == "baseline/wam":
        model = WAMSyncModel(img_size=256, device=args.device)
        # model = WAMSyncModel(img_size=args.short_edge_size if args.short_edge_size > 0 else 256, device=args.device)
        model.eval()
        model.to(args.device)
        config = None
    else:
        model, config = setup_model_from_checkpoint(args.checkpoint)
        model.eval()
        device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # Optionally override scaling_w
        if args.scaling_w is not None and args.scaling_w != "none":
            try:
                scaling_w_val = float(args.scaling_w)
                model.blender.scaling_w = scaling_w_val
                print(f"Overriding scaling_w to {scaling_w_val}")
            except Exception as e:
                print(f"Could not override scaling_w: {e}")

    # Setup the dataset
    dataset = setup_dataset(args)
    dataset = Subset(dataset, range(args.num_samples))

    # Run sync evaluation
    evaluate_sync(
        model=model,
        dataset=dataset,
        output_dir=args.output_dir,
        only_identity=args.only_identity
    )

    # Load and print grouped avg pixel diff from CSV
    csv_path = os.path.join(args.output_dir, "sync_metrics.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        grouped = df.groupby(['geom_aug', 'val_aug'])['avg_pixel_diff'].mean().reset_index()
        grouped.loc[len(grouped)] = ['all', 'all', df['avg_pixel_diff'].mean()]
        print("\nGrouped Avg Pixel Diff by Geometric and Value-Metric Augmentation:")
        print(grouped)
    else:
        print(f"CSV file not found at {csv_path}")

if __name__ == "__main__":
    main()
