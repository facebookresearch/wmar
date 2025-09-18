# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .sequential import Sequential
from .valuemetric import Identity, JPEG, Brightness, Contrast, GaussianBlur, Grayscale, Hue

def get_validation_augs_subset() -> list:
    """
    Get the validation augmentations.
    """
    augs = [
        (Identity(),                [0]),  # No parameters needed for identity
        (Brightness(),              [0.5]),
        (JPEG(),                    [60]),
        (Sequential(JPEG(), Brightness()), [(60, 0.5)]),
    ]
    return augs


def get_combined_augs() -> list:
    """
    Get only the combined augmentations for validation.
    """
    augs = [
        (Identity(),          [0]),  # Always include identity for baseline
        (Sequential(JPEG(), Brightness()), [(40, 0.5)]),
    ]
    return augs


def get_validation_augs(
    only_identity: bool = False,
    only_combined: bool = False,
    only_valuemetric: bool = False,
) -> list:
    """
    Get the validation augmentations.
    Args:
        only_identity (bool): Whether to only use identity augmentation
        only_combined (bool): Whether to only use combined augmentations
    """
    if only_identity:
        augs = [
            (Identity(),          [0]),  # No parameters needed for identity
        ]
    elif only_combined:
        augs = get_combined_augs()
    elif only_valuemetric:
        augs = [
            (Identity(),                        [0]),  # No parameters needed for identity
            (Brightness(),                      [0.5, 1.5, 2.0]),
            (Contrast(),                        [0.5, 1.5, 2.0]),
            (Hue(),                             [-0.2, -0.1, 0.1, 0.2]),
            (Grayscale(),                       [-1]),  # No parameters needed
            (JPEG(),                            [20, 40, 60, 80]),
            (GaussianBlur(),                    [3, 9, 17]),
            (Sequential(JPEG(), Brightness()),  [(40, 2.0)]),
            (Sequential(JPEG(), Brightness()),  [(80, 2.0)]),
        ]
    else:
        augs = [
            (Identity(),          [0]),  # No parameters needed for identity
            (Brightness(),        [0.5, 1.5, 2.0]),
            (Contrast(),          [0.5, 1.5, 2.0]),
            (Hue(),               [-0.2, -0.1, 0.1, 0.2]),
            (Grayscale(),         [-1]),  # No parameters needed
            (JPEG(),              [20, 40, 60, 80]),
            (GaussianBlur(),      [3, 9, 17]),
            (Sequential(JPEG(), Brightness()), [(40, 2.0)]),
            (Sequential(JPEG(), Brightness()), [(80, 2.0)]),
        ]
    return augs
