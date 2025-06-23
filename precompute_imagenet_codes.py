# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.​

import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from wmar.models.armm_wrapper import load_model
from wmar.utils.utils import chw_to_pillow

# python3 precompute_imagenet_codes.py --model taming --modelpath checkpoints/2021-04-03T19-39-50_cin_transformer/ \
# --imagenet_root data/imagenet/061417/ --outdir out/imagenet_taming

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["taming", "chameleon7b", "rar"], help="model to use")
parser.add_argument("--modelpath", type=str, help="path to the model (see README.md)")
parser.add_argument(
    "--imagenet_root",
    type=str,
    help="imagenet root e.g., data/imagenet/061417/, containing labels.txt, train/, val/, test/; each of the last 3 with subdirs of the form n01440764/",
)
parser.add_argument("--outdir", type=str, help="path to the output directory, e.g., out/imagenet_taming")
args = parser.parse_args()

size = 512 if args.model == "chameleon7b" else 256

# Load VQGAN only
if args.model == "taming":
    vqgan_config_path = os.path.join(args.modelpath, "configs", "vqgan.yaml")
    vqgan_ckpt_path = os.path.join(args.modelpath, "checkpoints", "vqgan.ckpt")
elif args.model == "chameleon7b":
    vqgan_config_path = os.path.join(args.modelpath, "tokenizer", "vqgan.yaml")
    vqgan_ckpt_path = os.path.join(args.modelpath, "tokenizer", "vqgan_patched.ckpt")
    # NOTE: make sure to patch it first!
    vqgan = load_model(vqgan_config_path, vqgan_ckpt_path, device="cuda")
elif args.model == "rar":
    vqgan_config_path = "deps/rar/configs/training/generator/rar.yaml"
    vqgan_ckpt_path = None  # downloaded
else:
    raise ValueError(f"Model {args.model} not supported")
vqgan_codebase = "rar" if args.model == "rar" else "taming"
vqgan = load_model(vqgan_config_path, vqgan_ckpt_path, vqgan_codebase=vqgan_codebase)

# Load imagenet labels
with open(os.path.join(args.imagenet_root, "labels.txt"), "r") as f:
    labels = f.readlines()
labels = [label.strip().split(",")[0] for label in labels]

# Get number of images to precompute per class
if size == 512:
    # (if size is 512 we need a custom split since there's not enough images from some classes)
    imagenet_512_split_50k_path = os.path.join("assets", "imagenet_512_split_50k.txt")
    with open(imagenet_512_split_50k_path, "r") as f:
        cnt_per_label = [line.strip().split(",") for line in f.readlines()]
        cnt_per_label = {k: int(v) for k, v in cnt_per_label}
else:
    # otherwise just use 50 per class
    cnt_per_label = {k: 50 for k in labels}

# Get paths to precompute
paths = {}
print("Going through all labels")
for label in labels:
    cls_dir = os.path.join(args.imagenet_root, "train", label)
    cls_paths = [os.path.join(cls_dir, p) for p in os.listdir(cls_dir)]
    paths[label] = np.random.choice(cls_paths, size=cnt_per_label[label], replace=False)
    np.random.shuffle(paths[label])
total_paths = sum(len(paths[label]) for label in labels)
assert total_paths == 50_000, "Total paths should be 50,000"

# n01734418 -> 56 (king snake)
imagenet_class_index_path = os.path.join("assets", "imagenet_class_index.json")
with open(imagenet_class_index_path, "r") as f:
    imagenet_class_index = json.load(f)
label_to_idx = {}
for idx, val in imagenet_class_index.items():
    label, _ = val
    label_to_idx[label] = idx

# Start encoding
outdir_codes = os.path.join(args.outdir, "codes")
os.makedirs(outdir_codes, exist_ok=True)
outdir_images = os.path.join(args.outdir, "images")
os.makedirs(outdir_images, exist_ok=True)

transform = transforms.Compose(
    [
        transforms.Resize(size),
        transforms.RandomCrop((size, size)),
        transforms.ToTensor(),
        lambda x: 2.0 * x - 1.0,  # normalize to [-1, 1]
    ]
)

# For RAR we use the function that expects [-1, 1] like the Taming one
encode_fn = vqgan.encode_like_taming if args.model == "rar" else vqgan.encode

# Finally generate all codes and also save resized/cropped images
for i, (label, curr_paths) in enumerate(paths.items()):
    conditioning = label_to_idx[label]
    if int(conditioning) not in [0, 999]:
        continue
    print(conditioning)
    print(curr_paths)
    for count, path in enumerate(curr_paths):
        print(count)
        if count > 1:
            continue
        img = Image.open(path)
        if not img.mode == "RGB":
            img = img.convert("RGB")
        img = transform(img).to("cuda")
        z_q, _, (_, _, z_indices) = encode_fn(img.unsqueeze(0))
        code = z_indices.view(-1).cpu().numpy()

        chw_to_pillow(img).save(os.path.join(outdir_images, f"{conditioning}:{count:04}.png"))
        np.save(os.path.join(outdir_codes, f"{conditioning}:{count:04}.npy"), code)
