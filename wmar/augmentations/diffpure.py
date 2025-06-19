# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.​

import yaml
from deps.saberi_wmr.utils import GuidedDiffusion, dict2namespace

"""
    Wraps the DiffPure from deps.saberi_wmr/
"""


class DiffPure:
    # [0,1] -> [0,1]
    # B=1
    def __init__(self, steps=0.4):
        with open("deps/saberi_wmr/imagenet.yml", "r") as f:
            config = yaml.safe_load(f)
        self.config = dict2namespace(config)

        # NOTE: make sure you downloaded the model (see README.md)
        model_dir = "checkpoints/"
        t = int(steps * int(self.config.model.timestep_respacing))
        print(f"Using t = {t} from steps = {steps}")
        self.runner = GuidedDiffusion(self.config, t=t, model_dir=model_dir, device="cuda")
        self.steps = steps

    def __call__(self, imgs, steps_override=None):
        orig_device = imgs.device
        if steps_override is not None:
            t = int(steps_override * int(self.config.model.timestep_respacing))
            self.runner.t = t
            self.steps = steps_override
        imgs_pured, _ = self.runner.image_editing_sample((imgs - 0.5) * 2)
        imgs_pured = (imgs_pured.to(imgs.dtype).to(orig_device) + 1) / 2
        imgs_pured = imgs_pured.clamp(0, 1)
        return imgs_pured

    def __repr__(self) -> str:
        return f"diffpure-{self.steps}"
