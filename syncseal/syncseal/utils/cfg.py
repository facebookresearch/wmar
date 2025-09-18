# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import omegaconf

import torch
import torchvision.transforms as transforms

from syncseal.data.datasets import CocoImageIDWrapper, ImageFolder
from syncseal.models import build_embedder, build_extractor, SyncModel
from syncseal.modules.jnd import JND


def setup_model_from_checkpoint(ckpt_path):

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    args = state_dict["args"]
    args = omegaconf.OmegaConf.create(args)
    if not isinstance(args, omegaconf.DictConfig):
        raise Exception("Expected logfile to contain params dictionary.")

    # Load sub-model configurations
    embedder_cfg = omegaconf.OmegaConf.load(args.embedder_config)
    extractor_cfg = omegaconf.OmegaConf.load(args.extractor_config)

    # Build the embedder model
    embedder_params = embedder_cfg[args.embedder_model]
    embedder = build_embedder(args.embedder_model, embedder_params)
    print(f'embedder: {sum(p.numel() for p in embedder.parameters() if p.requires_grad) / 1e6:.1f}M parameters')

    # Build the extractor model
    extractor_params = extractor_cfg[args.extractor_model]
    extractor = build_extractor(args.extractor_model, extractor_params, args.img_size_proc)
    print(f'extractor: {sum(p.numel() for p in extractor.parameters() if p.requires_grad) / 1e6:.1f}M parameters')

    attenuation_cfg = omegaconf.OmegaConf.load(args.attenuation_config)
    attenuation = JND(**attenuation_cfg[args.attenuation])

    sync_model = SyncModel(embedder, extractor, 
                           None, None, 
                           attenuation, args.scaling_w, args.scaling_i,
                           img_size=args.img_size_proc)
    sync_model = sync_model.eval()
    sync_model.load_state_dict(state_dict["model"])
    return sync_model, args


def setup_dataset(args):
    try:
        dataset_config = omegaconf.OmegaConf.load(f"configs/datasets/{args.dataset}.yaml")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset configuration not found: {args.dataset}")
    # Image dataset
    resize_short_edge = None
    if args.short_edge_size > 0:
        transform_list = [transforms.Resize(args.short_edge_size)]
        if getattr(args, "square_images", False):
            transform_list.append(transforms.CenterCrop(args.short_edge_size))
        resize_short_edge = transforms.Compose(transform_list)
    if dataset_config.val_annotation_file:
        # COCO dataset, with masks
        dataset = CocoImageIDWrapper(
            root = dataset_config.val_dir,
            annFile = dataset_config.val_annotation_file,
            transform = resize_short_edge, 
            mask_transform = resize_short_edge
        )
    else:
        # ImageFolder dataset
        dataset = ImageFolder(
            path = dataset_config.val_dir,
            transform = resize_short_edge
        )  
    print(f"Image dataset loaded from {dataset_config.val_dir}")
    return dataset
