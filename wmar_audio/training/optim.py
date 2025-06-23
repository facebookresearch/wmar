# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import logging

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from contextlib import contextmanager

def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, 
        warmup_steps: int, 
        training_steps: int,
        min_mult: float = 1e-2,
        cycles: float = 0.5,
        last_epoch: int = -1
    ):
    """
    Create a schedule with a learning rate that first increases linearly during warmup,
    then decreases following a cosine curve.

    Args:
        optimizer (Optimizer): PyTorch optimizer.
        warmup_steps (int): Number of steps for linear warmup.
        training_steps (int): Total number of training steps.
        min_mult (float): Minimum multiplier for the learning rate (default is 1e-3).
        cycles (float): Number of cosine cycles (default is 0.5, i.e., decay to 0).
        last_epoch (int): The index of the last epoch.

    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(current_step: int):
        # Linear warmup
        if current_step < warmup_steps:
            progress = current_step / warmup_steps
            return min_mult + (1 - min_mult) * progress
        elif current_step > training_steps:
            return min_mult
        else:
            # Cosine decay
            progress = (current_step - warmup_steps) / (max(1, training_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * cycles * 2.0 * progress))
            return max(min_mult, min_mult + (1 - min_mult) * cosine_decay)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    Args:
        ckp_path: path to the checkpoint
        run_variables: dictionary of variables to re-load
        kwargs: dictionary of objects to re-load. The key is the name of the object in the checkpoint file, the value is the object to load.
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=True)

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                try:
                    msg = value.load_state_dict(checkpoint[key], strict=True)
                except:
                    checkpoint[key] = {k.replace("module.", ""): v for k, v in checkpoint[key].items()}
                    msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))
    print(flush=True)

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]
