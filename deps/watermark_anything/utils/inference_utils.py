# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from sklearn.cluster import DBSCAN
from torchvision import transforms
from deps.watermark_anything.augmentation.augmenter import Augmenter
from deps.watermark_anything.data.transforms import (
    default_transform,
    normalize_img,
    unnormalize_img,
)
from deps.watermark_anything.models import Wam, build_embedder, build_extractor
from deps.watermark_anything.modules.jnd import JND


def msg2str(msg):
    return "".join([('1' if el else '0') for el in msg])

def str2msg(str):
    return [True if el=='1' else False for el in str]

def load_model_from_checkpoint(json_path, ckpt_path):
    """
    Load a model from a checkpoint file and a JSON file containing the parameters.
    Args:
    - json_path (str): the path to the JSON file containing the parameters
    - ckpt_path (str): the path to the checkpoint file
    """
    # Load the JSON file
    with open(json_path, 'r') as file:
        params = json.load(file)
    # Create an argparse Namespace object from the parameters
    args = argparse.Namespace(**params)
    # print(args)
    
    # Load configurations
    embedder_cfg = omegaconf.OmegaConf.load(args.embedder_config)
    embedder_params = embedder_cfg[args.embedder_model]
    extractor_cfg = omegaconf.OmegaConf.load(args.extractor_config)
    extractor_params = extractor_cfg[args.extractor_model]
    augmenter_cfg = omegaconf.OmegaConf.load(args.augmentation_config)
    attenuation_cfg = omegaconf.OmegaConf.load(args.attenuation_config)
        
    # Build models
    embedder = build_embedder(args.embedder_model, embedder_params, args.nbits)
    extractor = build_extractor(extractor_cfg.model, extractor_params, args.img_size, args.nbits)
    augmenter = Augmenter(**augmenter_cfg)
    try:
        attenuation = JND(**attenuation_cfg[args.attenuation], preprocess=unnormalize_img, postprocess=normalize_img)
    except:
        attenuation = None
    
    # Build the complete model
    wam = Wam(embedder, extractor, augmenter, attenuation, args.scaling_w, args.scaling_i)
    
    # Load the model weights
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        wam.load_state_dict(checkpoint)
        print("Model loaded successfully from", ckpt_path)
        print(params)
    else:
        print("Checkpoint path does not exist:", ckpt_path)
    
    return wam



def create_random_mask(img_pt, num_masks=1, mask_percentage=0.1):
    _, _, height, width = img_pt.shape
    mask_area = int(height * width * mask_percentage)
    masks = torch.zeros((num_masks, 1, height, width), dtype=img_pt.dtype)
    for ii in range(num_masks):
        placed = False
        while not placed:
            # Calculate maximum possible dimensions
            max_dim = int(mask_area ** 0.5)
            mask_width = random.randint(1, max_dim)
            mask_height = mask_area // mask_width
            # Ensure the ratio is between 1/2 and 1/4
            if 1/2 <= mask_width / mask_height <= 1:
                # Ensure dimensions fit within the image
                if mask_height <= height and mask_width <= width:
                    # Randomly select top-left corner
                    x_start = random.randint(0, width - mask_width)
                    y_start = random.randint(0, height - mask_height)
                    # Check for overlap with existing masks
                    overlap = False
                    for jj in range(ii):
                        if torch.sum(masks[jj, :, y_start:y_start + mask_height, x_start:x_start + mask_width]) > 0:
                            overlap = True
                            break
                    if not overlap:
                        masks[ii, :, y_start:y_start + mask_height, x_start:x_start + mask_width] = 1
                        placed = True
    return masks.to(img_pt.device)

def multiwm_dbscan(preds: torch.Tensor, masks: torch.Tensor = None, gt_masks= None, threshold: float = 0.0, epsilon = 1, min_samples = 3000) -> torch.Tensor:
    """
    Perform DBSCAN clustering on the predicted masks to identify clusters of pixels that are part of the same watermark.

    Args:
    - preds (torch.Tensor): the predicted bits per pixel from the model
    - masks (torch.Tensor): the predicted mask from the model
    - gt_masks (torch.Tensor): the ground truth mask
    - threshold (float): the threshold for the predicted bits
    - epsilon (float): the maximum distance between two samples for one to be considered as in the neighborhood of the other
    - min_samples (int): the number of samples in a neighborhood for a point to be considered as a core point
    """
    preds = preds > threshold  # B, K, H, W


    union_mask = (masks.squeeze(1)>0.5).float() # B, H, W

    bit_accuracies = []
    H, W = union_mask[0].shape
    K = preds.shape[1] # number of bits

    nb_clusters_detected = 0

    predictions = None 
    full_labels_store = None

    # print(preds)

    for i in range(preds.shape[0]):
        # select the corresponding mask union and predicition
        mask = union_mask[i] # H, W
        pred = preds[i] # K, H, W
        
        pred = pred.view(K, -1).t().cpu() # H*W, K
        valid_indices = (mask.view(-1)>0).cpu() 
        # print(f"proportion of pixels detected as wm: {valid_indices.sum().item()/(H*W)}")
        valid_pred = pred[valid_indices].float() # shape [num_valid, K]
        if valid_pred.shape[0] == 0:
            print("no valid pixels detected")
            continue
        db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(valid_pred)
        labels = torch.Tensor(db.labels_).float()
        # Store labels in the original shape, respecting the mask
        full_labels = -torch.ones(pred.shape[0]).float()  # shape [H*W], Start with all as noise
        full_labels[valid_indices] = labels  # Only fill where mask was 1
        full_labels = full_labels.reshape(H, W)
        # if i == 1:
        full_labels_store = full_labels
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]  # Exclude noise
        nb_clusters_detected += len(unique_labels)
        predictions = {}
        for label in unique_labels:
            cluster_points = valid_pred[labels == label]
            centroid = cluster_points.mean(0)
            centroid = centroid > 0.5
            predictions[label] = centroid.int()
    return predictions, full_labels_store


def torch_to_np(img_tensor):
    img_tensor = unnormalize_img(img_tensor).clamp(0, 1)
    img_tensor = img_tensor.squeeze().permute(1, 2, 0).cpu()
    return img_tensor.numpy()

# Define a color map for each unique value for multiple wm viz
color_map = {
    -1: [0, 0, 0],       # Black for -1
    0: [255, 0, 255], # ? for 0
    1: [255, 0, 0],     # Red for 1
    2: [0, 255, 0],     # Green for 2
    3: [0, 0, 255],     # Blue for 3
    4: [255, 255, 0],   # Yellow for 4
}

def plot_outputs(img, img_w, mask, mask_pred, labels = None, centroids = None):
    """
    Plot the original image, the watermarked image, the difference image, the ground truth mask, and the predicted mask.
    Args:
    - img (torch.Tensor): the original image
    - img_w (torch.Tensor): the watermarked image
    - mask (torch.Tensor): the ground truth mask
    - mask_pred (torch.Tensor): the predicted mask

    Optional when doing multiple watermark detection:
    - labels (torch.Tensor): the predicted labels (corresponds to the different clusters / watermarks)
    - centroids (dict): the centroids of the clusters (predicted messages)
    """
    resize_ori = transforms.Resize(img.shape[-2:])
    # prepare images
    delta = (img_w * mask - img * mask).squeeze().permute(1, 2, 0).cpu().numpy()
    delta = np.clip(np.abs(10 * delta), 0, 1)
    img, img_w = torch_to_np(img), torch_to_np(img_w)

    
    psnr = peak_signal_noise_ratio(img, img_w)
    
    # plot images: original, watermarked, difference
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(img_w)
    plt.title('Image with localized watermark')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(delta), cmap='hot')
    # plt.title('Difference Image (PSNR: {:.2f})'.format(psnr))
    plt.title('Difference image between original and localized wm')
    plt.axis('off')
    plt.show()
    masks_target = mask.squeeze().cpu().numpy()
    mask_preds = mask_pred.squeeze().cpu().numpy()
    # plot masks
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(masks_target, cmap='gray')
    plt.title('Position of the watermark')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(mask_preds, cmap='gray', vmin=0, vmax=1)
    plt.title('Predicted watermark position')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    if labels is not None:
        full_labels_store = labels
        rgb_image = torch.zeros((3, full_labels_store.shape[-1], full_labels_store.shape[-2]), dtype=torch.uint8)
        # Map each value to its corresponding color
        for value, color in color_map.items():
            mask_ = full_labels_store == value
            for channel, color_value in enumerate(color):
                rgb_image[channel][mask_.squeeze()] = color_value
        rgb_image = resize_ori(rgb_image.float()/255)
        rgb_image = rgb_image.permute(1, 2, 0).numpy()
        # Create a legend with sentences
        legend_labels = [f'{msg2str(centroids[key])}' for key in centroids.keys()]
        legend_colors = [np.array(color_map[key])/255 for key in centroids.keys()]
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]
        plt.legend(handles, legend_labels, loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.imshow(rgb_image)
        plt.title('clusters')
        plt.axis('off')
    plt.show()