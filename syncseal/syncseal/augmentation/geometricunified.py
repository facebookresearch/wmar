# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Test with:
    python -m syncseal.augmentation.geometricunified
"""

import random
import math

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def rotate_point(pt, angle, cx, cy):
    """
    Rotate a point (x, y) around a center (cx, cy) by a given angle in radians.
    """
    x, y = pt
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x_new = cos_a * (x - cx) - sin_a * (y - cy) + cx
    y_new = sin_a * (x - cx) + cos_a * (y - cy) + cy
    return x_new, y_new

def perspective_point(pt, pt_b, pt_c, perturb_ab, perturb_ac):
    """
    Distort a point (x, y) by a random perturbation along directions pt_b->pt and pt_c->pt.
    """
    x, y = pt
    perturb_x = perturb_ab * (pt_b[0] - x) + perturb_ac * (pt_c[0] - x)
    perturb_y = perturb_ab * (pt_b[1] - y) + perturb_ac * (pt_c[1] - y)
    return x + perturb_x, y + perturb_y


class GeometricAugmenter(nn.Module):
    """
    Unified geometric augmenter that applies one or several geometric augmentations
    (rotate, resize, crop, perspective, hflip, identity) sequentially based on given probabilities.
    """
    def __init__(
        self,
        augs: dict,
        augs_params: dict,
        num_augs: int = 1,
        **kwargs: dict
    ) -> None:
        super().__init__()
        self.num_augs = num_augs

        # Populate lists of augmentation names, probabilities, and parameters.
        self.aug_names = []
        self.aug_probs = []
        self.aug_params = []
        self.possible_classes = ['identity', 'rotate', 'crop', 'perspective', 'hflip']
        for name, prob in augs.items():
            if prob > 0:
                if name not in self.possible_classes:
                    raise ValueError(f"Unknown geometric augmentation: {name}")
                self.aug_names.append(name)
                self.aug_probs.append(prob)
                self.aug_params.append(augs_params.get(name, {}))

        # Normalize probabilities.
        total = sum(self.aug_probs)
        self.aug_probs = torch.tensor([p / total for p in self.aug_probs])

    def _get_random_pts(self, pts: list[tuple[int]]) -> tuple[list[tuple[int]], dict]:
        """
        Get new points representing the new topleft, topright, botright, botleft corners.
        Returns tuple of (new_pts, aug_info)
        """
        index = torch.multinomial(self.aug_probs, 1).item()
        selected_aug = self.aug_names[index]
        params = self.aug_params[index]

        if selected_aug == 'identity':
            return pts, {'aug_type': 'identity'}
        elif selected_aug == 'rotate':
            return self._apply_rotation(pts, params)
        elif selected_aug == 'crop':
            return self._apply_crop(pts, params)
        elif selected_aug == 'perspective':
            return self._apply_perspective(pts, params)
        elif selected_aug == 'hflip':
            return self._apply_hflip(pts)
        else:
            return pts, {'aug_type': 'unknown'}

    def _apply_rotation(self, pts: list[tuple[float, float]], params: dict) -> tuple[list[tuple[float, float]], dict]:
        """
        Apply rotation to the 4 corner points.

        Args:
            pts (list): A list of four (x, y) tuples defining the corners of the source quadrilateral.
            params (dict): A dictionary of parameters for the rotation.
                - min_angle (float): Minimum rotation angle in degrees.
                - max_angle (float): Maximum rotation angle in degrees.
                - do90 (bool): Whether to include 90-degree rotations.
                - fill (bool): If True, the image is zoomed after rotation to fill the
                  frame and avoid black borders. Defaults to False.
                  :warning: Not implemented yet.

        Returns:
            A tuple containing the new corner points and augmentation info.
        """
        min_angle = params.get('min_angle', -30)
        max_angle = params.get('max_angle', 30)
        assert -90 <= min_angle <= 90, f"min_angle must be between -90 and 90 degrees. Got {min_angle}"
        assert -90 <= max_angle <= 90, f"max_angle must be between -90 and 90 degrees. Got {max_angle}"
        assert min_angle <= max_angle, f"min_angle must be less than max_angle. Got {min_angle} and {max_angle}"

        do90 = params.get('do90', False)
        fill = params.get('fill', False)  # Check for the 'fill' option

        # Calculate the center of the quadrilateral
        cx = sum(pt[0] for pt in pts) / 4
        cy = sum(pt[1] for pt in pts) / 4

        if do90:
            base_angles = [-90, 0, 90]
            base_angle = random.choice(base_angles)
            fine_angle = random.uniform(min_angle, max_angle)
            total_angle = base_angle + fine_angle  # Only apply fine angle after permutation
        else:
            total_angle = random.uniform(min_angle, max_angle)
            # if fine angle is closer to 90 or -90, we can treat it as a 90-degree rotation
            if abs(total_angle - 90) % 360 < 45:
                base_angle = 90
                fine_angle = total_angle - base_angle
            elif abs(total_angle + 90) % 360 < 45:
                base_angle = -90
                fine_angle = total_angle - base_angle
            else:
                base_angle = 0
                fine_angle = total_angle

        # For 90/-90, permute the points before applying fine rotation
        if base_angle == 90:
            # tl, tr, br, bl = tr, br, bl, tl
            pts = [pts[1], pts[2], pts[3], pts[0]]
        elif base_angle == -90:
            # tl, tr, br, bl = bl, tl, tr, br
            pts = [pts[3], pts[0], pts[1], pts[2]]

        angle_rad = math.radians(fine_angle)
        new_pts = [rotate_point(pt, angle_rad, cx, cy) for pt in pts]
        if fill:
            pass

        aug_info = {
            'aug_type': 'rotate',
            'angle_degrees': total_angle,
            'center': (cx, cy),
            'fill': fill
        }

        return new_pts, aug_info

    def _apply_crop(self, pts, params):
        """Apply crop by selecting a random rectangle with aspect ratio between 1/2 and 2, inside the quadrilateral.
        If params['topleft'] is True, crop is always a square from the top-left corner of the bounding box.
        """
        min_size = params.get('min_size', 0.5)
        max_size = params.get('max_size', 0.9)
        min_ar = 0.5
        max_ar = 2.0
        topleft = params.get('topleft', False)

        # Compute the bounding box of the quadrilateral
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y

        if topleft:
            # Always crop a square from the top-left corner
            scale = random.uniform(min_size, max_size)
            side = min(width, height) * scale
            x0, y0 = min_x, min_y
            x1, y1 = x0 + side, y0 + side
            new_pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            aug_info = {
                'aug_type': 'crop',
                'scale': scale,
                'center': (cx, cy),
                'rect': (x0, y0, x1, y1),
                'aspect_ratio': 1.0,
                'topleft': True
            }
            return new_pts, aug_info

        # Try up to 10 times to find a valid crop
        # Random area scale
        scale = random.uniform(min_size, max_size)
        area = width * height * scale
        for _ in range(10):
            # Random aspect ratio
            aspect = random.uniform(min_ar, max_ar)
            crop_w = math.sqrt(area * aspect)
            crop_h = math.sqrt(area / aspect)

            if crop_w > width or crop_h > height:
                continue

            # Random top-left corner within bounds
            x0 = random.uniform(min_x, max_x - crop_w)
            y0 = random.uniform(min_y, max_y - crop_h)
            x1 = x0 + crop_w
            y1 = y0 + crop_h

            # Rectangle corners: TL, TR, BR, BL
            crop_pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

            # Check if all corners are inside the quadrilateral using a point-in-polygon test
            def point_in_quad(pt, quad):
                # Ray casting algorithm for convex quad
                from matplotlib.path import Path
                return Path(quad).contains_point(pt)

            if all(point_in_quad(cpt, pts) for cpt in crop_pts):
                new_pts = crop_pts
                break
        else:
            # Fallback: use a square crop
            aspect = 1.0
            crop_w = crop_h = math.sqrt(area)
            x0 = random.uniform(min_x, max_x - crop_w)
            y0 = random.uniform(min_y, max_y - crop_h)
            x1 = x0 + crop_w
            y1 = y0 + crop_h
            new_pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

        # Center of the crop rectangle
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        aug_info = {
            'aug_type': 'crop',
            'scale': scale,
            'center': (cx, cy),
            'rect': (x0, y0, x1, y1),
            'aspect_ratio': (x1-x0)/(y1-y0) if (y1-y0) != 0 else None,
            'topleft': False
        }

        return new_pts, aug_info

    def _apply_perspective(self, pts, params):
        """Apply perspective distortion to the 4 corner points."""
        max_scale = params.get('max_distortion_scale', 0.1)

        new_pts = []
        perturbations = []
        for i, pt in enumerate(pts):
            prev_pt = pts[(i - 1) % 4]
            next_pt = pts[(i + 1) % 4]
            perturb_prev = max_scale * random.uniform(-1, 1)
            perturb_next = max_scale * random.uniform(-1, 1)
            new_pt = perspective_point(pt, prev_pt, next_pt, perturb_prev, perturb_next)
            new_pts.append(new_pt)
            perturbations.append({'prev': perturb_prev, 'next': perturb_next})

        aug_info = {
            'aug_type': 'perspective',
            'max_distortion_scale': max_scale,
            'perturbations': perturbations
        }

        return new_pts, aug_info

    def _apply_hflip(self, pts):
        """Apply horizontal flip to the 4 corner points."""
        # Calculate the bounding box to determine flip axis
        min_x = min(pt[0] for pt in pts)
        max_x = max(pt[0] for pt in pts)
        center_x = (min_x + max_x) / 2

        new_pts = []
        for pt in pts:
            x, y = pt
            new_x = 2 * center_x - x  # Flip horizontally around center
            new_pts.append((new_x, y))

        aug_info = {
            'aug_type': 'hflip',
            'flip_center_x': center_x
        }

        return new_pts, aug_info

    def forward(self, image, mask=None):
        """
        image: a tensor of shape (B, C, H, W) or (C, H, W)
        mask: (optional) tensor with the same spatial dimensions as image.
        """
        H, W = image.shape[-2:]

        # start_pts are the original corners of the image
        end_pts = [(0, 0), (W - 1, 0), (W - 1, H - 1), (0, H - 1)]  # topleft, topright, botright, botleft
        start_pts = end_pts.copy()
        applied_transformations = []

        for aug_idx in range(self.num_augs):  # Apply num_augs augmentations sequentially
            aug_probs_saved = self.aug_probs.clone()
            # Drop everything but crop and identity for the first augmentation.
            if aug_idx == 0 and self.num_augs > 1:
                crop_idx = self.aug_names.index('crop')
                identity_idx = self.aug_names.index('identity')
                for idx in range(len(self.aug_probs)):
                    if idx == crop_idx:
                        self.aug_probs[idx] = 1
                    elif idx == identity_idx:
                        self.aug_probs[idx] = 1
                    elif idx != crop_idx and idx != identity_idx:
                        self.aug_probs[idx] = 0
            # remove crops for other augmentations
            if aug_idx > 0 and 'crop' in self.aug_names:
                crop_idx = self.aug_names.index('crop')
                self.aug_probs[crop_idx] = 0
            # Apply random augmentation.
            start_pts, aug_info = self._get_random_pts(start_pts)
            applied_transformations.append(aug_info)
            # Restore original probabilities after the first augmentation.
            self.aug_probs = aug_probs_saved

        # Apply the transformation to the image.
        image = TF.perspective(image, start_pts, end_pts, interpolation=InterpolationMode.BILINEAR)
        if mask is not None:
            mask = TF.perspective(mask, start_pts, end_pts, interpolation=InterpolationMode.NEAREST)

        # Return the augmented image, mask, and transformation info.
        info = {
            'startpoints': start_pts,
            'endpoints': end_pts,
            'applied_transformations': applied_transformations
        }
        return image, mask, info

    def __repr__(self):
        return f"UnifiedGeometricAugmenter(augs={self.aug_names}, probs={self.aug_probs})"


if __name__ == "__main__":
    import os
    from PIL import Image
    from torchvision.transforms import ToTensor
    from torchvision.utils import save_image

    from syncseal.utils.helpers import display_wrap

    # seed
    random.seed(42)
    torch.manual_seed(42)

    augs = {
        'identity': 1,
        'rotate': 1,
        'crop': 0.5,
        'perspective': 1,
        'hflip': 1,
    }
    augs_params = {
        'crop': {'min_size': 0.3, 'max_size': 1.0},
        'rotate': {'min_angle': -40, 'max_angle': 40, 'do90': True, 'fill': True},
        'perspective': {'max_distortion_scale': 0.2},
    }

    img_paths = [
        "images/squirrel.png",
        "images/squirrel.png"
    ]
    imgs = [ToTensor()(Image.open(path)) for path in img_paths]
    imgs = torch.stack(imgs)

    augmenter = GeometricAugmenter(augs, augs_params, num_augs=2)
    print("Augmenter:", augmenter)

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    # Apply 3 sequential augmentations.
    for ii in range(10):
        imgs_aug, _, info = augmenter(imgs)
        # Visualize points for the first image in the batch, side by side
        triplet_img = display_wrap(
            imgs[0], imgs_aug[0],
            info['startpoints'], info['endpoints'],
            orig_color="red", aug_color="green"
        )
        triplet_img.save(os.path.join(output_dir, f"imgs_aug_{ii}_points.png"))
        # save_image(imgs_aug.clamp(0, 1),
        #            os.path.join(output_dir, f"imgs_aug_{ii}.png"), nrow=2)
        # print(f"Saved: imgs_aug_{ii}.png, info: {info}")

    # Test each augmentation independently
    print("\n=== Testing individual augmentations ===")
    individual_augs = ['identity', 'rotate', 'crop', 'perspective', 'hflip']

    for aug_name in individual_augs:
        print(f"\nTesting {aug_name} augmentation...")
        single_aug = {aug_name: 1.0}
        single_augmenter = GeometricAugmenter(single_aug, augs_params)

        for ii in range(3):
            imgs_aug, _, info = single_augmenter(imgs)
            save_image(imgs_aug.clamp(0, 1),
                      os.path.join(output_dir, f"{aug_name}_aug_{ii}.png"), nrow=2)
            print(f"  Saved: {aug_name}_aug_{ii}.png, start_pts: {info['startpoints']}")
            triplet_img = display_wrap(
                imgs[0], imgs_aug[0],
                info['startpoints'], info['endpoints'],
                orig_color="red", aug_color="green"
            )
            triplet_img.save(os.path.join(output_dir, f"{aug_name}_aug_{ii}_points.png"))
