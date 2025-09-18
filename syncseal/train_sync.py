# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Example usage (cluster 2 gpus):
    torchrun --nproc_per_node=2 train_sync.py --local_rank 0
Example usage (cluster 1 gpu):
    torchrun train_sync.py --debug_slurm

Put OMP_NUM_THREADS such that OMP_NUM_THREADS=(number of CPU threads)/(nproc per node) to remove warning messages
        
Example:

    OMP_NUM_THREADS=40 torchrun --nproc_per_node=2 train_sync.py --local_rank 0 
"""

import argparse
import datetime
import json
import os
import time
from typing import List

import numpy as np
import omegaconf

import torch
import torch.distributed as dist
import torch.nn as nn
from torchvision.utils import save_image

import syncseal.utils as utils
import syncseal.utils.dist as udist
import syncseal.utils.logger as ulogger
import syncseal.utils.optim as uoptim
from syncseal.augmentation import get_validation_augs, get_validation_augs_subset
from syncseal.augmentation.geometricunified import GeometricAugmenter
from syncseal.augmentation.augmenter import Augmenter, geometric_augs
from syncseal.data.loader import get_dataloader_segmentation
from syncseal.data.transforms import get_resize_transform
from syncseal.evals.metrics import psnr, ssim
from syncseal.losses.sync_loss import SyncLoss
from syncseal.models import SyncModel, build_embedder, build_extractor
from syncseal.modules.jnd import JND
from syncseal.utils.helpers import create_diff_img, warp_image_homography

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Dataset parameters')
    aa("--dataset_config", type=str,
        help="Name of the image dataset.", default="configs/datasets/sa-1b-full-resized.yaml")
    aa("--finetune_detector_start", type=int, default=1e6,
       help="Number of epochs afterwhich the generator is frozen and detector is finetuned")

    group = parser.add_argument_group('Experiments parameters')
    aa("--output_dir", type=str, default="output/",
       help="Output directory for logs and images (Default: /output)")

    group = parser.add_argument_group('Embedder and extractor config')
    aa("--embedder_config", type=str, default="configs/embedder.yaml",
       help="Path to the embedder config file")
    aa("--extractor_config", type=str, default="configs/extractor.yaml",
       help="Path to the extractor config file")
    aa("--attenuation_config", type=str, default="configs/attenuation.yaml",
       help="Path to the attenuation config file")
    aa("--embedder_model", type=str, default="convnext_tiny",
       help="Name of the extractor model")
    aa("--extractor_model", type=str, default="sam_tiny",
       help="Name of the extractor model")

    group = parser.add_argument_group('Augmentation parameters')
    aa("--augmentation_config", type=str, default="configs/all_augs.yaml",
       help="Path to the augmentation config file")
    aa("--num_augs", type=int, default=2,
       help="Number of augmentations to apply for geometric augmenter")
    aa("--num_augs_valuem", type=int, default=2,
       help="Number of augmentations to apply for valuemetric augmenter")
    aa("--per_img_aug", type=utils.bool_inst, default=True,
       help="If True, apply per-image augmentation, otherwise batch augmentation")

    group = parser.add_argument_group('Image and watermark parameters')
    aa("--img_size", type=int, default=256,
       help="Size of the input images for data preprocessing, used at loading time for training.")
    aa("--img_size_val", type=int, default=256,
       help="Size of the input images for data preprocessing, used at loading time for validation.")
    aa("--img_size_proc", type=int, default=256, 
       help="Size of the input images for interpolation in the embedder/extractor models")
    aa("--resize_only", type=utils.bool_inst, default=False,
         help="If True, only resize the image no crop is applied at loading time (without preserving aspect ratio)")
    aa("--attenuation", type=str, default="jnd_1_1", help="Attenuation model to use")
    aa("--scaling_w", type=float, default=0.2,
       help="Scaling factor for the watermark in the embedder model")
    aa("--scaling_w_schedule", type=str, default=None,
       help="Scaling factor for the watermark in the embedder model. Ex: 'Linear,scaling_min=0.025,epochs=100,start_epoch=0'")
    aa("--scaling_i", type=float, default=1.0,
       help="Scaling factor for the image in the embedder model")
    aa("--lowres_attenuation", type=utils.bool_inst, default=False,
       help="Apply attenuation at low resolution for high-res images (more memory efficient)")
    aa("--rounding", type=utils.bool_inst, default=True,
       help="If True, rounding the output images to 8-bit (Default: True)")

    group = parser.add_argument_group('Optimizer parameters')
    aa("--optimizer", type=str, default="AdamW,lr=1e-4",
       help="Optimizer (default: AdamW,lr=1e-4)")
    aa("--scheduler", type=str, default="None",
       help="Scheduler (default: None)")
    aa('--epochs', default=300, type=int,
       help='Number of total epochs to run')
    aa('--iter_per_epoch', default=500, type=int,
       help='Number of iterations per epoch, made for very large datasets')
    aa('--iter_per_valid', default=None, type=int,
       help='Number of iterations per eval, made for very large eval datasets if None eval on all dataset')
    aa('--resume_from', default=None, type=str,
       help='Path to the checkpoint to resume from')
    aa('--resume_disc', type=utils.bool_inst, default=False,
       help='If True, also load discriminator weights when resuming from checkpoint')
    aa('--resume_optimizer_state', type=utils.bool_inst, default=False,
       help='If True, also load optimizer state when resuming from checkpoint')

    group = parser.add_argument_group('Losses parameters')
    aa('--temperature', default=1.0, type=float,
       help='Temperature for the mask loss')
    aa('--lambda_det', default=0.0, type=float,
       help='Weight for the watermark detection loss')
    aa('--lambda_sync', default=1.0, type=float,
       help='Weight for the watermark synchronization loss')
    aa('--lambda_i', default=0.0, type=float, help='Weight for the image loss')
    aa('--lambda_d', default=0.1, type=float,
       help='Weight for the discriminator loss')
    aa('--perceptual_loss', default='mse', type=str,
       help='Perceptual loss to use. e.g., "lpips", "mse"')
    aa('--disc_start', default=25, type=float,
       help='Weight for the discriminator loss')
    aa('--disc_num_layers', default=2, type=int,
       help='Number of layers for the discriminator')
    aa('--disc_in_channels', default=1, type=int,
         help='Number of input channels for the discriminator')
    aa('--random_mask', type=utils.bool_inst, default=False,
       help='For WAM-style prediction -- if True, randomly mask out part of the watermark during training')
    aa('--transform_loss', default='mae', type=str, choices=['mse', 'mae'],
       help='Transform loss type for geometric transformation prediction. Options: "mse", "mae"')

    group = parser.add_argument_group('Loading parameters')
    aa('--batch_size', default=32, type=int, help='Batch size')
    aa('--batch_size_eval', default=32, type=int, help='Batch size for evaluation')
    aa('--workers', default=0, type=int, help='Number of data loading workers')

    group = parser.add_argument_group('Misc.')
    aa('--only_eval', type=utils.bool_inst,
       default=False, help='If True, only runs evaluate')
    aa('--eval_freq', default=5, type=int, help='Frequency for evaluation')
    aa('--full_eval_freq', default=50, type=int,
       help='Frequency for full evaluation')
    aa('--saveimg_freq', default=5, type=int, help='Frequency for saving images')
    aa('--saveckpt_freq', default=50, type=int, help='Frequency for saving ckpts')
    aa('--seed', default=0, type=int, help='Random seed')

    group = parser.add_argument_group('Distributed training parameters')
    aa('--debug_slurm', action='store_true')
    aa('--local_rank', default=-1, type=int)
    aa('--master_port', default=-1, type=int)

    return parser


def main(params):

    params = omegaconf.OmegaConf.create(vars(params))

    # Distributed mode
    udist.init_distributed_mode(params)

    # Set seeds for reproductibility
    seed = params.seed + udist.get_rank()
    # seed = params.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if params.distributed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    # Print the arguments
    print("__git__:{}".format(utils.get_sha()))
    json_params = json.dumps(omegaconf.OmegaConf.to_container(params, resolve=True))
    print("__log__:{}".format(json_params))

    # Copy the config files to the output dir
    if udist.is_main_process():
        os.makedirs(os.path.join(params.output_dir, 'configs'), exist_ok=True)
        os.system(f'cp {params.embedder_config} {params.output_dir}/configs/embedder.yaml')
        os.system(f'cp {params.augmentation_config} {params.output_dir}/configs/augs.yaml')
        os.system(f'cp {params.extractor_config} {params.output_dir}/configs/extractor.yaml')

    # Build the embedder model
    embedder_cfg = omegaconf.OmegaConf.load(params.embedder_config)
    params.embedder_model = params.embedder_model or embedder_cfg.model
    embedder_params = embedder_cfg[params.embedder_model]
    embedder = build_embedder(params.embedder_model, embedder_params)
    print(f'embedder: {sum(p.numel() for p in embedder.parameters() if p.requires_grad) / 1e6:.1f}M parameters')

    # Build the extractor model
    extractor_cfg = omegaconf.OmegaConf.load(params.extractor_config)
    params.extractor_model = params.extractor_model or extractor_cfg.model
    extractor_params = extractor_cfg[params.extractor_model]
    if params.extractor_model.startswith('unet'):
        extractor = build_embedder(params.extractor_model, extractor_params)
    else:
        extractor = build_extractor(params.extractor_model, extractor_params, params.img_size_proc)
    print(f'extractor: {sum(p.numel() for p in extractor.parameters() if p.requires_grad) / 1e6:.1f}M parameters')

    # build attenuation
    if params.attenuation.lower() != "none":
        attenuation_cfg = omegaconf.OmegaConf.load(params.attenuation_config)
        if params.attenuation.lower().startswith("jnd"):
            attenuation_cfg = omegaconf.OmegaConf.load(params.attenuation_config)
            attenuation = JND(**attenuation_cfg[params.attenuation]).to(device)
        else:
            attenuation = None
    else:
        attenuation = None
    print(f'attenuation: {attenuation}')

    # build the valuemetric augmenter
    augmenter_cfg = omegaconf.OmegaConf.load(params.augmentation_config)
    for aug_key in augmenter_cfg['augs']:
        if aug_key in geometric_augs:
            augmenter_cfg['augs'][aug_key] = 0
    valuem_augmenter = Augmenter(**augmenter_cfg).to(device)
    valuem_augmenter.num_augs = params.num_augs_valuem

    # build the geometric augmenter
    augmenter_cfg = omegaconf.OmegaConf.load(params.augmentation_config)
    for aug_key in augmenter_cfg['augs']:
        if aug_key not in geometric_augs:
            augmenter_cfg['augs'][aug_key] = 0
    geom_augmenter = GeometricAugmenter(**augmenter_cfg).to(device)
    geom_augmenter.num_augs = params.num_augs
    print(f'augmenter: {geom_augmenter}')

    # build the complete model
    sync_model = SyncModel(embedder, extractor, 
                geom_augmenter, valuem_augmenter, 
                attenuation, params.scaling_w, params.scaling_i,
                img_size=params.img_size_proc,
                rounding=params.rounding)
    sync_model = sync_model.to(device)
    # print(sync_model)

    # build losses
    sync_loss = SyncLoss(
        disc_weight=params.lambda_d, percep_weight=params.lambda_i,
        detect_weight=params.lambda_det, sync_weight=params.lambda_sync,
        disc_start=params.disc_start, disc_num_layers=params.disc_num_layers, disc_in_channels=params.disc_in_channels,
        percep_loss=params.perceptual_loss, transform_loss=params.transform_loss,
    ).to(device)
    print(sync_loss)

    # Build the scaling schedule. Default is none
    if params.scaling_w_schedule is not None and params.scaling_w_schedule.lower() != "none":
        scaling_w_schedule = uoptim.parse_params(params.scaling_w_schedule)
        scaling_scheduler = uoptim.ScalingScheduler(
            obj=sync_model.blender, attribute="scaling_w", scaling_o=params.scaling_w,
            **scaling_w_schedule
        )
    else:
        scaling_scheduler = None

    # Build optimizer and scheduler
    model_params = list(embedder.parameters()) + list(extractor.parameters())
    optim_params = uoptim.parse_params(params.optimizer)
    optimizer = uoptim.build_optimizer(model_params, **optim_params)
    scheduler_params = uoptim.parse_params(params.scheduler)
    scheduler = uoptim.build_lr_scheduler(optimizer, **scheduler_params)
    print('optimizer: %s' % optimizer)
    print('scheduler: %s' % scheduler)

    # discriminator optimizer
    optim_params_d = uoptim.parse_params(params.optimizer)
    optimizer_d = uoptim.build_optimizer(
        model_params=[*sync_loss.discriminator.parameters()],
        **optim_params_d
    )
    scheduler_d = uoptim.build_lr_scheduler(optimizer=optimizer_d, **scheduler_params)
    print('optimizer_d: %s' % optimizer_d)
    print('scheduler_d: %s' % scheduler_d)

    # Data loaders
    dataset_config = omegaconf.OmegaConf.load(params.dataset_config)
    train_transform, train_mask_transform = get_resize_transform(params.img_size, resize_only=params.resize_only)
    val_transform, val_mask_transform = get_resize_transform(params.img_size_val)
    train_loader = get_dataloader_segmentation(dataset_config.train_dir,
                                                dataset_config.train_annotation_file,
                                                transform=train_transform,
                                                mask_transform=train_mask_transform,
                                                batch_size=params.batch_size,
                                                num_workers=params.workers, shuffle=True,
                                                random_mask=params.random_mask)
    val_loader = get_dataloader_segmentation(dataset_config.val_dir,
                                                dataset_config.val_annotation_file,
                                                transform=val_transform,
                                                mask_transform=val_mask_transform,
                                                batch_size=params.batch_size_eval,
                                                num_workers=params.workers,
                                                shuffle=False,
                                                random_nb_object=False)

    # optionally resume training
    if params.resume_from is not None:
        components_to_load = {'model': sync_model}
        if params.resume_disc:
            components_to_load['discriminator'] = sync_loss.discriminator
        if params.resume_optimizer_state:
            components_to_load['optimizer'] = optimizer
            components_to_load['optimizer_d'] = optimizer_d
        uoptim.restart_from_checkpoint(
            params.resume_from,
            **components_to_load
        )

    to_restore = {
        "epoch": 0,
    }
    uoptim.restart_from_checkpoint(
        os.path.join(params.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=sync_model,
        discriminator=sync_loss.discriminator,
        optimizer=optimizer,
        optimizer_d=optimizer_d,
        scheduler=scheduler,
        scheduler_d=scheduler_d
    )
    start_epoch = to_restore["epoch"]
    for param_group in optimizer.param_groups:
        param_group['lr'] = optim_params['lr']
    for param_group in optimizer_d.param_groups:
        param_group['lr'] = optim_params_d['lr']
    optimizers = [optimizer, optimizer_d]

    # specific thing to do if distributed training
    if params.distributed:
        # if model has batch norm convert it to sync batchnorm in distributed mode
        sync_model = nn.SyncBatchNorm.convert_sync_batchnorm(sync_model)

        sync_model_ddp = nn.parallel.DistributedDataParallel(
            sync_model, device_ids=[params.local_rank])
        sync_loss.discriminator = nn.parallel.DistributedDataParallel(
            sync_loss.discriminator, device_ids=[params.local_rank])
        sync_model = sync_model_ddp.module
    else:
        sync_model_ddp = sync_model

    dummy_img = torch.ones(3, params.img_size_val, params.img_size_val)
    validation_masks = valuem_augmenter.mask_embedder.sample_representative_masks(
        dummy_img)  # n 1 h w, full of ones or random masks depending on config

    # evaluation only
    if params.only_eval and udist.is_main_process():
        augs = get_validation_augs()
        val_stats = eval_one_epoch(sync_model, val_loader, sync_loss, 0, augs, validation_masks, params)
        with open(os.path.join(params.output_dir, f'log_only_eval.txt'), 'a') as f:
            f.write(json.dumps(val_stats) + "\n")
        return

    # start training
    print('training...')
    start_time = time.time()
    for epoch in range(start_epoch, params.epochs):

        # scheduler
        if scheduler is not None:
            scheduler.step(epoch)
            scheduler_d.step(epoch)
        if scaling_scheduler is not None:
            scaling_scheduler.step(epoch)

        if params.distributed:
            train_loader.sampler.set_epoch(epoch)

        # prepare if freezing the generator and finetuning the detector
        if epoch >= params.finetune_detector_start:
            # remove the grads from embedder
            sync_model.embedder.requires_grad_(False)
            sync_model.embedder.eval()
            
            # rebuild DDP with unused parameters
            sync_model_ddp = nn.parallel.DistributedDataParallel(
                sync_model, device_ids=[params.local_rank], find_unused_parameters=True)

            # set to 0 the weights of the perceptual losses
            params.lambda_i = 0.0
            params.lambda_d = 0.0
            params.balanced = False
            sync_loss.percep_weight = 0.0
            sync_loss.disc_weight = 0.0

        # train and log
        train_stats = train_one_epoch(sync_model_ddp, optimizers, train_loader, sync_loss, epoch, params)
        log_stats = {
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_stats.items()}
        }

        if epoch % params.eval_freq == 0:
            if (epoch % params.full_eval_freq == 0 and epoch > 0) or (epoch == params.epochs-1):
                augs = get_validation_augs()
            else:
                augs = get_validation_augs_subset()
            val_stats = eval_one_epoch(sync_model, val_loader, sync_loss, epoch, augs, validation_masks, params)
            log_stats = {**log_stats, **{f'val_{k}': v for k, v in val_stats.items()}}

        if udist.is_main_process():
            with open(os.path.join(params.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + "\n")
        if udist.is_dist_avail_and_initialized():
            dist.barrier()  # Ensures all processes wait until the main node finishes validation

        print("Saving Checkpoint..")
        discrim_no_ddp = sync_loss.discriminator.module if params.distributed else sync_loss.discriminator
        save_dict = {
            'epoch': epoch + 1,
            'model': sync_model.state_dict(),
            'discriminator': discrim_no_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'scheduler_d': scheduler_d.state_dict() if scheduler_d is not None else None,
            'args': omegaconf.OmegaConf.to_yaml(params),
        }
        udist.save_on_master(save_dict, os.path.join(
            params.output_dir, 'checkpoint.pth'))
        if params.saveckpt_freq and epoch % params.saveckpt_freq == 0:
            udist.save_on_master(save_dict, os.path.join(
                params.output_dir, f'checkpoint{epoch:03}.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total time {}'.format(total_time_str))


def train_one_epoch(
    sync_model: SyncModel,
    optimizers: List[torch.optim.Optimizer],
    train_loader: torch.utils.data.DataLoader,
    sync_loss: SyncLoss,
    epoch: int,
    params: argparse.Namespace,
) -> dict:
    sync_model.train()

    header = f'Train - Epoch: [{epoch}/{params.epochs}]'
    metric_logger = ulogger.MetricLogger(delimiter="  ")

    for it, batch_items in enumerate(metric_logger.log_every(train_loader, 10, header, params.iter_per_epoch)):
        if it >= params.iter_per_epoch:
            break

        # some data loaders return batch_data, masks, frames_positions as well
        imgs, masks = batch_items[0], batch_items[1]
        
        if params.lambda_d == 0:  # no disc, optimize embedder/extractor only
            optimizer_ids_for_epoch = [0]
        else:
            optimizer_ids_for_epoch = [1, 0]

        imgs = imgs.to(device, non_blocking=True)

        for optimizer_idx in optimizer_ids_for_epoch:
            optimizers[optimizer_idx].zero_grad()

        # forward
        outputs = sync_model(imgs, masks, per_img_aug=params.per_img_aug)
        outputs["preds"] = outputs["preds"] / params.temperature

        # index 1 for discriminator, 0 for embedder/extractor
        for optimizer_idx in optimizer_ids_for_epoch:
            loss, logs = sync_loss(
                imgs, outputs["imgs_w"], 
                outputs["preds"], outputs["target_pts"],
                optimizer_idx, epoch,
            )
            # ignore loss if it is nan
            if torch.isnan(loss).any().item():
                print(f"Warning: NaN loss detected at epoch {epoch}, iteration {it}. Skipping this batch.", flush=True)
                continue
            loss.backward()

            # log stats
            log_stats = {
                **logs,
                'psnr': psnr(outputs["imgs_w"], imgs).mean().item(),
                'ssim': ssim(outputs["imgs_w"], imgs).mean().item(),
                'lr': optimizers[0].param_groups[0]['lr'],
            }
            det_preds = outputs["preds"][:, 0:1]  # b 1
            corner_preds = outputs["preds"][:, 1:]  # b nparams h w (or global if not pixelwise)

            torch.cuda.synchronize()
            for name, value in log_stats.items():
                metric_logger.update(**{name: value})

            # save images on training
            if (epoch % params.saveimg_freq == 0) and it == 0:
            # if (epoch % params.saveimg_freq == 0) and (it % 50) == 0:
                ori_path = os.path.join(
                    params.output_dir, f'{epoch:03}_{it:03}_train_0_ori.png')
                wm_path = os.path.join(
                    params.output_dir, f'{epoch:03}_{it:03}_train_1_wm.png')
                diff_path = os.path.join(
                    params.output_dir, f'{epoch:03}_{it:03}_train_2_diff.png')
                aug_path = os.path.join(
                    params.output_dir, f'{epoch:03}_{it:03}_train_3_aug.png')
                warp_path = os.path.join(
                    params.output_dir, f'{epoch:03}_{it:03}_train_5_warp.png')
                warp_diff_path = os.path.join(
                    params.output_dir, f'{epoch:03}_{it:03}_train_6_warpdiff.png')

                if udist.is_main_process():
                    save_image(imgs, ori_path, nrow=8)
                    save_image(outputs["imgs_w"], wm_path, nrow=8)
                    save_image(create_diff_img(imgs, outputs["imgs_w"]), diff_path, nrow=8)
                    save_image(outputs["imgs_aug"], aug_path, nrow=8)
                    if params.lambda_sync > 0:
                        # For training, original_size is the same as current image size
                        B, C, H, W = outputs["imgs_aug"].shape
                        original_size = (H, W)
                        img_size = getattr(sync_model, 'img_size', 256)
                        warped_image = warp_image_homography(
                            images=outputs["imgs_aug"], 
                            corner_points=corner_preds,
                            original_size=original_size,
                            img_size=img_size
                        )
                        save_image(warped_image, warp_path, nrow=8)
                        save_image(create_diff_img(outputs["imgs_w"], warped_image), warp_diff_path, nrow=8)
                    
        grad_norm = torch.nn.utils.clip_grad_norm_(sync_model.parameters(), max_norm=1000, norm_type=2.0, error_if_nonfinite=False)
        if torch.isnan(grad_norm).any().item():
            print(f"Warning: NaN gradient norm detected at epoch {epoch}, iteration {it}. Skipping this batch.", flush=True)
            continue

        # add optimizer step
        for optimizer_idx in optimizer_ids_for_epoch:
            optimizers[optimizer_idx].step()

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('train'), metric_logger)
    train_logs = {k: meter.global_avg for k,
                  meter in metric_logger.meters.items()}

    return train_logs


@ torch.no_grad()
def eval_one_epoch(
    sync_model: SyncModel,
    val_loader: torch.utils.data.DataLoader,
    sync_loss: SyncLoss,
    epoch: int,
    validation_augs: List,
    validation_masks: torch.Tensor,
    params: argparse.Namespace,
) -> dict:
    """
    Evaluate the model on the validation set, with different augmentations

    Args:
        sync_model (SyncModel): the model
        val_loader (torch.utils.data.DataLoader): the validation loader
        sync_loss (SyncLoss): the loss function
        epoch (int): the current epoch
        validation_augs (List): list of augmentations to apply
        validation_masks (torch.Tensor): the validation masks, full of ones for now
        params (argparse.Namespace): the parameters
    """
    if torch.is_tensor(validation_masks):
        validation_masks = list(torch.unbind(validation_masks, dim=0))

    sync_model.eval()

    header = f'Val - Epoch: [{epoch}/{params.epochs}]'
    metric_logger = ulogger.MetricLogger(delimiter="  ")

    for it, batch_items in enumerate(metric_logger.log_every(val_loader, 10, header)):
        if params.iter_per_valid is not None and it >= params.iter_per_valid:
            break

        # some data loaders return batch_data, masks, frames_positions as well
        imgs, _ = batch_items[0], batch_items[1]
            
        # forward embedder
        embed_time = time.time()
        outputs = sync_model.embed(imgs, lowres_attenuation=params.lowres_attenuation)
        embed_time = (time.time() - embed_time) / imgs.shape[0]
        imgs_w = outputs["imgs_w"]  # b c h w

        if (epoch % params.saveimg_freq == 0) and it == 0 and udist.is_main_process():
            base_name = os.path.join(params.output_dir, f'{epoch:03}_img_val')
            ori_path = base_name + '_0_ori.png'
            wm_path = base_name + '_1_wm.png'
            diff_path = base_name + '_2_diff.png'
            save_image(imgs, ori_path, nrow=8)
            save_image(imgs_w, wm_path, nrow=8)
            save_image(create_diff_img(imgs, imgs_w), diff_path, nrow=8)

        # quality metrics
        metrics = {}
        metrics['psnr'] = psnr(imgs_w, imgs).mean().item()
        metrics['ssim'] = ssim(imgs_w, imgs).mean().item()
        metrics['embed_time'] = embed_time
        torch.cuda.synchronize()
        metric_logger.update(**metrics)

        extract_times = []
        for mask_id, masks in enumerate(validation_masks):
            # watermark masking
            masks = masks.to(imgs.device)  # 1 h w
            if len(masks.shape) < 4:
                masks = masks.unsqueeze(0).repeat(imgs_w.shape[0], 1, 1, 1)  # b 1 h w
            imgs_masked = imgs_w * masks + imgs * (1 - masks)

            for transform_instance, strengths in validation_augs:

                for strength in strengths:
                    imgs_aug, masks_aug = transform_instance(
                            imgs_masked, masks, strength)
                    selected_aug = str(transform_instance) + f"_{strength}"
                    selected_aug = selected_aug.replace(", ", "_")

                    # extract watermark
                    extract_time = time.time()
                    outputs = sync_model.detect(imgs_aug)
                    extract_time = time.time() - extract_time
                    extract_times.append(extract_time / imgs_aug.shape[0])
                    preds = outputs["preds"]
                    det_preds = preds[:, 0:1]  # b 1 ...
                    param_preds = preds[:, 1:]  # b nparams ...

                    aug_log_stats = {}

                    current_key = f"aug={selected_aug}"
                    aug_log_stats = {f"{k}_{current_key}": v for k,
                                        v in aug_log_stats.items()}

                    torch.cuda.synchronize()
                    metric_logger.update(**aug_log_stats)

            metrics['extract_time'] = np.mean(extract_times)
            torch.cuda.synchronize()
            metric_logger.update(**metrics)

    metric_logger.synchronize_between_processes()
    print("Averaged {} stats:".format('val'), metric_logger)
    valid_logs = {k: meter.global_avg for k,
                  meter in metric_logger.meters.items()}
    return valid_logs


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
