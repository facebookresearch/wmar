# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np

def compute_corner_error(
    corner_points_norm: torch.Tensor,
    orig_points: list,
    original_size: tuple[int, int],
    img_size: int = 256,
    return_pts: bool = False,
    device: torch.device = None
) -> float | tuple[float, torch.Tensor]:
    """
    Compute corner error matching the scripted model's end_points computation.
    
    Args:
        corner_points_norm: Predicted corner points (B, 8) in [-1, 1] format from model
        orig_points: Target corner points as list of [x, y] coordinates
        original_size: (H, W) of the original image
        img_size: Model image size of processing (default 256)
        return_pts: If True, also return the detected corner points tensor
        device: Device for computation
        
    Returns:
        Mean corner error in pixels
    """
    if device is None:
        device = corner_points_norm.device
        
    # Follow the exact computation from scripted.py unwarp method
    center = torch.tensor(
        [img_size / 2.0, img_size / 2.0], 
        device=device, dtype=corner_points_norm.dtype
    )  # tensor([128, 128]) for img_size=256
    
    H, W = original_size
    
    # Take first batch item
    pts = corner_points_norm[0].view(4, 2)  # (4,2) normalized
    end_pts = (pts * center + center).round()  # denormalize to [0, img_size-1]
    
    # Scale to original image size
    end_pts[..., 0] = end_pts[..., 0] * ((W - 1) / (img_size - 1))  # scale to [0, W-1]
    end_pts[..., 1] = end_pts[..., 1] * ((H - 1) / (img_size - 1))  # scale to [0, H-1]
    
    # Convert to same format as orig_points for comparison
    detected_points_array = end_pts.detach().cpu()
    orig_points_array = torch.tensor(orig_points).float()
    
    # Compute L2 distance for each corner and take mean
    corner_error = torch.norm(orig_points_array - detected_points_array, dim=1).mean().item()
    
    if return_pts:
        return corner_error, detected_points_array
    else:
        return corner_error


def compute_corner_error_legacy(
    corner_points_norm: torch.Tensor, 
    orig_points: list,
    center_pt: torch.Tensor
) -> float:
    """
    Legacy corner error computation (kept for reference).
    
    Args:
        corner_points_norm: Normalized corner points (B, 8) in [-1, 1] format from model
        orig_points: Original corner points as list of [x, y] coordinates
        center_pt: Center point tensor for denormalization
        
    Returns:
        Mean corner error in pixels
    """
    B = corner_points_norm.shape[0]
    
    # Convert corner points back to pixel coordinates
    corner_points_pixels = corner_points_norm.view(B, 4, 2).clone()  # b 8 -> b 4 2
    corner_points_pixels = corner_points_pixels * center_pt.view(1, 1, 2) + center_pt.view(1, 1, 2)
    corner_points_pixels = corner_points_pixels.detach().cpu()
    
    orig_points_array = torch.tensor(orig_points).float()
    detected_points_array = corner_points_pixels[0]
    corner_error = torch.norm(orig_points_array - detected_points_array, dim=1).mean().item()
    
    return corner_error


def warp_image_homography(
    images: torch.Tensor, 
    pixel_map: torch.Tensor = None,
    corner_points: torch.Tensor = None,
    homography_matrix: torch.Tensor = None,
    original_size: tuple[int, int] = None,
    img_size: int = 256
) -> torch.Tensor:
    """
    Warp a batch of images using a homography transformation specified by either a pixel map, corner points, or a homography matrix.

    Args:
        images (torch.Tensor): Batch of images of shape (B, C, H, W).
        pixel_map (torch.Tensor, optional): Tensor of shape (B, 3, H, W) containing pixel weights and coordinates for homography estimation. The first channel is the weight, and the next two channels are the x and y coordinates in the original image, normalized to [0, 1].
        corner_points (torch.Tensor, optional): Tensor of shape (B, 8) or (8) representing the four corners of the image in the original image coordinates, normalized to [-1, 1].
        homography_matrix (torch.Tensor, optional): Tensor of shape (B, 9) or (B, 3, 3) representing the homography matrix for each image.
        original_size (tuple[int, int], optional): Original size (H, W) for corner point scaling. If None, uses image size.
        img_size (int): Model image size for normalization (default 256).

    Returns:
        torch.Tensor: Batch of warped images of shape (B, C, H, W).
    """
    assert pixel_map is not None or corner_points is not None or homography_matrix is not None, "Either pixel_map or corner_points or homography_matrix must be provided."
    B, _, H, W = images.shape

    if corner_points is not None:
        # Use original_size if provided, otherwise fall back to image size
        if original_size is not None:
            H_orig, W_orig = original_size
        else:
            H_orig, W_orig = H, W
            
        # Follow scripted model's computation approach
        center = torch.tensor(
            [img_size / 2.0, img_size / 2.0], 
            device=images.device, dtype=images.dtype
        )
        start_pts = torch.tensor(
            [[0.0, 0.0], [W_orig - 1.0, 0.0], [W_orig - 1.0, H_orig - 1.0], [0.0, H_orig - 1.0]],
            device=images.device, dtype=images.dtype
        )
        
        images_transformed = []
        for ii in range(B):
            img_i = images[ii:ii + 1]
            corner_points_i = corner_points[ii] if corner_points.dim() == 2 else corner_points
            
            # Convert from normalized coordinates following scripted model approach
            pts = corner_points_i.view(4, 2)  # (4,2) normalized [-1, 1]
            end_pts = (pts * center + center).round()  # denormalize to [0, img_size-1]
            
            # Scale to original image size
            end_pts[..., 0] = end_pts[..., 0] * ((W_orig - 1) / (img_size - 1))  # scale to [0, W_orig-1]
            end_pts[..., 1] = end_pts[..., 1] * ((H_orig - 1) / (img_size - 1))  # scale to [0, H_orig-1]
            
            # Check for reasonable values
            if torch.linalg.norm(end_pts - start_pts) > 1e6:
                # if corner_points and coords are too far apart, default to images
                images_transformed.append(img_i)
            else:
                # Convert to lists for torchvision perspective
                start_pts_list = start_pts.long().tolist()
                end_pts_list = end_pts.long().tolist()
                
                # Resize image to original size if needed
                img_rect = img_i.clone()
                if img_rect.shape[-2:] != (H_orig, W_orig):
                    img_rect = torch.nn.functional.interpolate(
                        img_rect, size=(H_orig, W_orig),
                        mode="bilinear", align_corners=False, antialias=True
                    )
                
                try:
                    warped_img = TF.perspective(
                        img_rect[0], start_pts_list, end_pts_list, 
                        interpolation=InterpolationMode.BILINEAR, fill=None
                    ).unsqueeze(0)
                    images_transformed.append(warped_img)
                except Exception as e:
                    print(f"Error in perspective transform: {e}, using original image.")
                    images_transformed.append(img_i)
                
        images_transformed = torch.cat(images_transformed, dim=0)

    else:
        try:
            import kornia
        except:
            import inspect
            frame = inspect.currentframe()
            print(f"File: {frame.f_code.co_filename}, line: {frame.f_lineno}")
            print("Please install kornia if you encounter issue in homography operations, with pip install kornia")

        if pixel_map is not None:
            # computation of homography matrix is done on cpu to prevent OOM on GPU
            pixel_map = pixel_map.view(B, 3, H * W)[..., ::10].detach().cpu()
            pixel_weight, pixel_map = pixel_map[:, 0], pixel_map[:, 1:] * torch.tensor([W, H], device=pixel_map.device).view(1, 2, 1)

            coords = torch.stack([
                torch.arange(W, dtype=torch.float32).view(1, W).repeat(H, 1),
                torch.arange(H, dtype=torch.float32).view(H, 1).repeat(1, W)
            ], 2).view(1, -1, 2)[:, ::10].repeat(B, 1, 1)

            homography_mat = kornia.geometry.homography.find_homography_dlt(coords, pixel_map.transpose(1, 2), pixel_weight)
            images_transformed = kornia.geometry.transform.warp_perspective(images, homography_mat.to(images.device), dsize=(H, W))

        elif homography_matrix is not None:
            homography_mat = homography_matrix.view(B, 9)
            homography_mat = (homography_mat / (homography_mat[:, -1:] + 1e-8)).view(B, 3, 3)
            homography_mat = torch.inverse(homography_mat)
            images_transformed = kornia.geometry.transform.warp_perspective(images, homography_mat, dsize=(H, W))

    return images_transformed


def create_diff_img(img1, img2):
    """
    Create a difference image between two images.

    Parameters:
        img1 (torch.Tensor): The first image tensor of shape 3xHxW.
        img2 (torch.Tensor): The second image tensor of shape 3xHxW.

    Returns:
        torch.Tensor: The difference image tensor of shape 3xHxW.
    """
    diff = img1 - img2
    # diff = 0.5 + 10*(img1 - img2)
    # normalize the difference image
    diff = (diff - diff.min()) / ((diff.max() - diff.min()) + 1e-6)
    diff = 2*torch.abs(diff - 0.5)
    # diff = 20*torch.abs(diff)
    return diff.clamp(0, 1)


def display_wrap(orig_tensor_img, aug_tensor_img, orig_points, aug_points, 
                        save_path=None, orig_color="red", aug_color="blue", radius=5, connect=True,
                        left_title=None, right_title=None):
    """
    Return a side-by-side PIL image with points overlaid on both the original and augmented images.
    Handles points outside the image by adding padding.
    Adds corner labels ("TL", "TR", "BR", "BL") with white outline and black text.
    If save_path is provided, also saves the image.
    Optionally adds titles above the left and right images.
    """

    # Convert tensors to PIL images
    if hasattr(orig_tensor_img, "dim") and orig_tensor_img.dim() == 4:
        orig_tensor_img = orig_tensor_img[0]
    if hasattr(aug_tensor_img, "dim") and aug_tensor_img.dim() == 4:
        aug_tensor_img = aug_tensor_img[0]
    orig_img = TF.to_pil_image(orig_tensor_img.cpu())
    aug_img = TF.to_pil_image(aug_tensor_img.cpu())

    # Gather all points to compute bounding box
    all_points = list(orig_points) + list(aug_points)
    xs = [x for x, y in all_points]
    ys = [y for x, y in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Image sizes
    w, h = orig_img.size
    letter_pad = 28
    pad_left = max(0, int(-min(min_x, 0)) + letter_pad)
    pad_top = max(0, int(-min(min_y, 0)) + letter_pad)
    pad_right = max(0, int(max(max_x, w-1) - (w-1)) + letter_pad)
    pad_bottom = max(0, int(max(max_y, h-1) - (h-1)) + letter_pad)

    new_w = w + pad_left + pad_right
    new_h = h + pad_top + pad_bottom

    corner_labels = [f"p\u2081", f"p\u2082", f"p\u2083", f"p\u2084"]

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    # For titles, use a larger font if available
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        title_font = font

    def draw_label(draw, x, y, text):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw.text((x+dx, y+dy), text, font=font, fill="black")
        draw.text((x, y), text, font=font, fill="white")

    def pad_and_draw(img, points, color, labels):
        canvas = Image.new("RGB", (new_w, new_h), (255,255,255))
        canvas.paste(img, (pad_left, pad_top))
        draw = ImageDraw.Draw(canvas)
        shifted_pts = [(x + pad_left, y + pad_top) for (x, y) in points]
        # Draw ellipses and lines first
        for idx, (x, y) in enumerate(shifted_pts):
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=2)
        if connect and len(shifted_pts) == 4:
            draw.line([shifted_pts[0], shifted_pts[1], shifted_pts[2], shifted_pts[3], shifted_pts[0]], fill=color, width=2)
        # Draw labels last so they are on top
        for idx, (x, y) in enumerate(shifted_pts):
            label = labels[idx] if idx < len(labels) else ""
            draw_label(draw, x+radius+2, y-radius-2, label)
        return canvas

    orig_canvas = pad_and_draw(orig_img, orig_points, orig_color, corner_labels)
    aug_canvas = pad_and_draw(aug_img, aug_points, aug_color, corner_labels)

    # Optionally add titles above the images
    title_pad = 0
    if left_title or right_title:
        # Add vertical space for the title
        title_pad = 32
        new_h_with_title = new_h + title_pad
        orig_canvas_with_title = Image.new("RGB", (new_w, new_h_with_title), (255,255,255))
        aug_canvas_with_title = Image.new("RGB", (new_w, new_h_with_title), (255,255,255))
        orig_canvas_with_title.paste(orig_canvas, (0, title_pad))
        aug_canvas_with_title.paste(aug_canvas, (0, title_pad))
        draw_left = ImageDraw.Draw(orig_canvas_with_title)
        draw_right = ImageDraw.Draw(aug_canvas_with_title)
        if left_title:
            # Use font.getbbox for text size (Pillow >=8.0), fallback to font.getsize
            try:
                bbox = title_font.getbbox(left_title)
                w_txt, h_txt = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                w_txt, h_txt = title_font.getsize(left_title)
            draw_left.text(((new_w - w_txt) // 2, (title_pad - h_txt) // 2), left_title, font=title_font, fill="black")
        if right_title:
            try:
                bbox = title_font.getbbox(right_title)
                w_txt, h_txt = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                w_txt, h_txt = title_font.getsize(right_title)
            draw_right.text(((new_w - w_txt) // 2, (title_pad - h_txt) // 2), right_title, font=title_font, fill="black")
        orig_canvas = orig_canvas_with_title
        aug_canvas = aug_canvas_with_title
        new_h = new_h_with_title

    combined = Image.new("RGB", (new_w * 2, new_h), (255,255,255))
    combined.paste(orig_canvas, (0, 0))
    combined.paste(aug_canvas, (new_w, 0))
    if save_path is not None:
        combined.save(save_path)
    return combined


def display_wrap_matplotlib(orig_tensor_img, aug_tensor_img, orig_points, aug_points, 
                            save_path=None, orig_color="red", aug_color="blue", radius=5, connect=True,
                            left_title=None, right_title=None):
    """
    Display a side-by-side image using matplotlib with points and vector lines.
    Points are labeled p₁, p₂, etc. and lines are drawn as vectors (not pixeled).
    Optionally saves the figure if save_path is provided.
    """
    # Convert tensors to numpy arrays for matplotlib
    def to_numpy_img(tensor_img):
        if hasattr(tensor_img, "dim") and tensor_img.dim() == 4:
            tensor_img = tensor_img[0]
        img = tensor_img.detach().cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        return img

    orig_img = to_numpy_img(orig_tensor_img)
    aug_img = to_numpy_img(aug_tensor_img)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, img, points, color, title in zip(
        axes, [orig_img, aug_img], [orig_points, aug_points], [orig_color, aug_color], [left_title, right_title]
    ):
        ax.imshow(img)
        points = np.array(points)
        # Draw points
        ax.scatter(points[:, 0], points[:, 1], s=radius**2, c=color, edgecolors='black', zorder=3)
        # Draw lines as vectors
        if connect and len(points) == 4:
            for i in range(4):
                start = points[i]
                end = points[(i+1)%4]
                ax.annotate(
                    '', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='-', color=color, lw=2),
                    zorder=2
                )
        # Draw labels
        subscript_nums = ['₁', '₂', '₃', '₄']
        for idx, (x, y) in enumerate(points):
            label = f"p{subscript_nums[idx]}" if idx < len(subscript_nums) else f"p{idx+1}"
            # Draw white border for better visibility
            ax.text(x + 6, y - 6, label, color='white', fontsize=14, weight='bold', zorder=4, 
                    path_effects=[patheffects.withStroke(linewidth=1, foreground='white')])

            ax.text(x + 6, y - 6, label, color='black', fontsize=14, weight='bold', zorder=5)
        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def display_triplet_matplotlib(
    img1_tensor, img2_tensor, img3_tensor,
    pts1, pts2, pts3,
    save_path=None,
    color1="red", color2="green", color3="blue",
    radius=5, connect=True,
    title1=None, title2=None, title3=None
):
    """
    Display three images side-by-side with points and vector lines.
    Points are labeled p₁, p₂, ... on the first image, q₁, q₂, ... on the second, and 𝑝̂₁, 𝑝̂₂, ... on the third.
    Optionally saves the figure if save_path is provided.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as patheffects
    import numpy as np

    def to_numpy_img(tensor_img):
        if hasattr(tensor_img, "dim") and tensor_img.dim() == 4:
            tensor_img = tensor_img[0]
        img = tensor_img.detach().cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        return img

    imgs = [to_numpy_img(img1_tensor), to_numpy_img(img2_tensor), to_numpy_img(img3_tensor)]
    pts = [np.array(pts1), np.array(pts2), np.array(pts3)]
    colors = [color1, color2, color3]
    titles = [title1, title2, title3]
    label_prefixes = ["p", "q", r"$\hat{p}$"]
    subscript_nums = ['₁', '₂', '₃', '₄']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, points, color, title, prefix in zip(axes, imgs, pts, colors, titles, label_prefixes):
        ax.imshow(img)
        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], s=radius**2, c=color, edgecolors='black', zorder=3)
            if connect and len(points) == 4:
                for i in range(4):
                    start = points[i]
                    end = points[(i+1)%4]
                    ax.annotate(
                        '', xy=end, xytext=start,
                        arrowprops=dict(arrowstyle='-', color=color, lw=2),
                        zorder=2
                    )
            for idx, (x, y) in enumerate(points):
                if prefix == r"$\hat{p}$":
                    label = r"$\hat{p}_{" + f"{idx+1}" + r"}$"
                else:
                    label = f"{prefix}{subscript_nums[idx]}" if idx < len(subscript_nums) else f"{prefix}{idx+1}"
                ax.text(
                    x + 6, y - 6, label, color='white', fontsize=24, zorder=4,
                    path_effects=[patheffects.withStroke(linewidth=1, foreground='white')]
                )
                ax.text(
                    x + 6, y - 6, label, color='black', fontsize=24, zorder=5
                )
        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)
