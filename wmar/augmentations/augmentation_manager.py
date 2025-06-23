# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.​

from wmar.augmentations.diffpure import DiffPure
from wmar.augmentations.geometric import (
    HorizontalFlip,
    Rotate,
    UpperLeftCropWithResizeBack,
)
from wmar.augmentations.neuralcompression import NeuralCompression
from wmar.augmentations.valuemetric import JPEG, Brightness, GaussianBlur, GaussianNoise

"""AugmentationManager handles the setup of various image augmentations that we use in evaluation.

This class manages different types of image augmentations including:
- Geometric transformations (rotation, flipping, cropping)
- Value-based transformations (blur, noise, brightness, JPEG compression) 
- Neural compression (optional)
- DiffPure augmentations (optional)

All augmentations expect input images to be in the range [0,1].

Args:
    include_neural_compress (bool): Whether to include neural compression augmentations
    include_diffpure (bool): Whether to include DiffPure augmentations
    load_augs (bool): Whether to initialize the augmentations (useful to skip sometimes)
"""


class AugmentationManager:
    def __init__(self, include_neural_compress, include_diffpure, load_augs):
        self.include_neural_compress = include_neural_compress
        self.include_diffpure = include_diffpure

        self.augs = [
            (
                "gaussian-blur",
                None if not load_augs else lambda x, kernel_size: GaussianBlur()(x, kernel_size),
                [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
            ),
            (
                "gaussian-noise",
                None if not load_augs else lambda x, std: GaussianNoise()(x, std),
                [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2],
            ),
            (
                "jpeg",
                None if not load_augs else lambda x, quality: JPEG()(x, quality),
                [100, 95, 85, 75, 65, 55, 45, 35, 25, 15, 5],
            ),
            (
                "brightness",
                None if not load_augs else lambda x, brightness: Brightness()(x, brightness),
                [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3],
            ),
            (
                "rotation",
                None if not load_augs else lambda x, angle: Rotate()(x, angle),
                [-20, -15, -10, -5, 0, 5, 10, 15, 20],
            ),
            ("flip-h", None if not load_augs else lambda x, do: HorizontalFlip()(x) if do else x, [0, 1]),
            (
                "upperleft-crop",
                None if not load_augs else lambda x, factor: UpperLeftCropWithResizeBack()(x, factor),
                [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5],
            ),
        ]

        if self.include_neural_compress:
            print(f"AUGS: Including Neural Compress with load: {load_augs}")
            self.neural_compressor_names = [
                "bmshj2018-factorized-q=1",
                "bmshj2018-factorized-q=3",
                "bmshj2018-factorized-q=6",
                "bmshj2018-hyperprior-q=1",
                "bmshj2018-hyperprior-q=3",
                "bmshj2018-hyperprior-q=6",
                "mbt2018-mean-q=1",
                "mbt2018-mean-q=3",
                "mbt2018-mean-q=6",
                "mbt2018-q=1",
                "mbt2018-q=3",
                "mbt2018-q=6",
                "cheng2020-anchor-q=1",
                "cheng2020-anchor-q=3",
                "cheng2020-anchor-q=6",
                "cheng2020-attn-q=1",
                "cheng2020-attn-q=3",
                "cheng2020-attn-q=6",
                "diffusers-sd-vae-ft-ema",
                "diffusers-sd-vae-fp16",
                "diffusers-deep-compression",
                "diffusers-flux",
            ]
            if load_augs:
                self.compressors = {name: NeuralCompression.from_name(name) for name in self.neural_compressor_names}

            self.augs.append(
                (
                    "neural-compress",
                    None if not load_augs else lambda x, name: self.compressors[name](x).clamp(0, 1),
                    self.neural_compressor_names,
                )
            )
        else:
            print("AUGS: Including no neural compress")

        if self.include_diffpure:
            print(f"AUGS: Including DiffPure with load: {load_augs}")
            if load_augs:
                self.diffpure = DiffPure(steps=0.0001)
            self.augs.append(
                (
                    "diffpure",
                    None if not load_augs else lambda x, steps: self.diffpure(x, steps_override=steps),
                    [0.01, 0.05, 0.1, 0.2, 0.3],
                )
            )
        else:
            print("AUGS: Including no diffpure")
