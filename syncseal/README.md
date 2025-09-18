# SyncSeal: Geometric Image Synchronization

Self-contained code for robust image synchronization through watermark embedding, developed to improve the performance and robustness of Watermarking Autoregressive Image Generation (WMAR). This folder is standalone and does not depend on the WMAR repository. It was built upon the [Meta Video Seal codebase](https://github.com/facebookresearch/videoseal).

See paper: [`arXiv`](TODO_add_arXiv_link)


## Quickstart: TorchScript inference

We provide a scripted model so you can run the model out-of-the-box, without heavy setup.
Download it here: [`syncmodel.jit.pt`](https://dl.fbaipublicfiles.com/wmar/syncseal/paper/syncmodel.jit.pt), or through:
```bash
wget -O syncmodel.jit.pt https://dl.fbaipublicfiles.com/wmar/syncseal/paper/syncmodel.jit.pt
```

Minimal usage:
```python
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scripted = torch.jit.load("syncmodel.jit.pt").to(device).eval()

# Load an RGB image in [0,1]
img = Image.open("/path/to/image.jpg").convert("RGB")
img_pt = to_tensor(img).unsqueeze(0).to(device)

with torch.no_grad():
    emb = scripted.embed(img_pt)            # {'preds_w', 'imgs_w'}
    det = scripted.detect(emb["imgs_w"])    # {'preds', 'preds_pts'} where preds_pts is Bx8 corners in [-1,1]

# Optional: rectify the image using the detected corners
pred_pts = det["preds_pts"]
imgs_unwarped = scripted.unwarp(emb["imgs_w"], pred_pts, original_size=img_pt.shape[-2:])
```

For an end-to-end example (including simple augmentations and visualization), see `notebooks/standalone.ipynb`.


## Setup

### Requirements

First clone the root repository:
```bash
git clone https://github.com/facebookresearch/wmar.git
cd wmar/syncseal/
```

Python 3.10 is recommended. PyTorch should be installed to match your system (CPU or CUDA 12.1). We provide both pip and uv instructions.

PyTorch (choose one):
- CUDA 12.1 wheels via pip:
	```bash
	pip install --index-url https://download.pytorch.org/whl/cu121 \
			torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
	```
- CPU-only wheels via pip:
	```bash
	pip install --index-url https://download.pytorch.org/whl/cpu \
			torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
	```

Other dependencies with pip:
```bash
pip install -r requirements.txt
```

Copy WAM-synchronization used as baseline
```bash
cp -r ../deps/watermark_anything deps/watermark_anything
```

### Using uv (optional, fast installer)

This repo ships a `pyproject.toml` compatible with uv. 

1) Create and activate a virtualenv (examples):
```bash
uv venv --python 3.10
source .venv/bin/activate
```

2) Install deps with uv:
```bash
uv sync
```


### Data

Datasets are configured via YAML under `configs/datasets/`. 
For instance, in `configs/datasets/your-dataset.yaml`:
```
train_dir: /path/to/your-dataset/sa-1b/train/
val_dir: /path/to/your-dataset/sa-1b/val/
```
where train and val are directories containing images.

In the paper, we train with SA-1B which you can download in https://segment-anything.com/dataset/index.html.
We sort the filenames and use the first 1k images for validation, the following 1k for testing, and the resize all remaining images to 256x256 for training.


### Models

We release the following:

| Model Type | Checkpoint | TorchScript | Parameters | Training Logs | Console Outputs |
|------------|------------|-------------|----------|---------------|-----------------|
| Paper model (full) | [checkpoint.pth](https://dl.fbaipublicfiles.com/wmar/syncseal/paper/checkpoint.pth) | [syncmodel.jit.pt](https://dl.fbaipublicfiles.com/wmar/syncseal/paper/syncmodel.jit.pt) | [expe.json](https://dl.fbaipublicfiles.com/wmar/syncseal/paper/expe.json) | [log.txt](https://dl.fbaipublicfiles.com/wmar/syncseal/paper/log.txt) | [log.stdout](https://dl.fbaipublicfiles.com/wmar/syncseal/paper/log.stdout) |
| Codebase model (faster reproduction) | [checkpoint.pth](https://dl.fbaipublicfiles.com/wmar/syncseal/reproduction/checkpoint.pth) | [syncmodel.jit.pt](https://dl.fbaipublicfiles.com/wmar/syncseal/reproduction/syncmodel.jit.pt) | [expe.json](https://dl.fbaipublicfiles.com/wmar/syncseal/reproduction/expe.json) | [log.txt](https://dl.fbaipublicfiles.com/wmar/syncseal/reproduction/log.txt) | [log.stdout](https://dl.fbaipublicfiles.com/wmar/syncseal/reproduction/log.stdout) |

Remark about the main difference between the two models, w.r.t. the way geometric augmentations are applied during training:
- The paper's model first selected crop or identity (as done in `syncseal/augmentation/geometricunified.py`), but could also choose crops in subsequent augmentations, which led to a non-uniform sampling, and to sampling some extreme crops in terms of area. 
- The new code instead first selects crop or identity, but then removes crops from the pool of augmentations for subsequent augmentations, leading to a more uniform sampling of geometric transformations, and to a more stable training, which was found to obtain similar results in 4x fewer steps per epoch.
- Overall, the way augmentations are applied could be further tuned if you want to improve performance.


## Training

Use `train_sync.py` to train a synchronization model. 
Below is an example:
```bash
# Example (2 GPUs)
OMP_NUM_THREADS=40 torchrun --nproc_per_node=2 train_sync.py --local_rank 0
```

Notes
- Datasets are configured via YAML under `configs/datasets/`. Example: `configs/datasets/your-dataset.yaml`.
- To see the specific parameters used in the models released above, see `expe.json` files linked in the Models section.


### Checkpoint loading (non-TorchScript)

If you want to load a training checkpoint and run Python inference with the native modules, use the convenience function in `syncseal/utils/cfg.py`:
```python
from syncseal.utils.cfg import setup_model_from_checkpoint
model, cfg = setup_model_from_checkpoint("/path/to/checkpoint.pth")
model.eval().to("cuda")
```

### TorchScript export

After training, script your own checkpoint to a single `.pt` file with:
```bash
python -m syncseal.models.scripted \
	--checkpoint /path/to/checkpoint.pth
```

This will create `syncmodel.jit.pt` in the current directory which can be loaded as shown in the Quickstart section.


## Evaluation

Evaluate synchronization accuracy and image quality under geometric and value-metric augmentations:

```bash
python -m syncseal.evals.eval_sync \
	--checkpoint /path/to/checkpoint.pth \
	--dataset your-dataset \
	--num_samples 100 \
	--short_edge_size 512 \
	--square_images true \
	--output_dir outputs/sync_eval
```

Baseline options
- `--checkpoint baseline/sift` uses a SIFT+Lowe matching baseline.
- `--checkpoint baseline/wam` runs a WAM-based baseline.

The script writes two CSV files in the output directory:
- `sync_metrics.csv`: sync error per augmentation setup (includes detection and unwarp timing)
- `image_quality_metrics.csv`: PSNR/SSIM/LPIPS between original and watermarked images


## Notebooks

- Standalone quickstart and visualization: `notebooks/standalone.ipynb`


## License

Please see the LICENSE file in the root of the main repository.


## Citation

If you find this repository useful, please consider giving a star ⭐ and please cite as:

```
@article{fernandez2025geometric,
    title={Geometric Image Synchronization with Deep Watermarking},
    author={Fernandez, Pierre and Sou\v{c}ek, Tom\'{a}\v{s} and Jovanovi\'{c}, Nikola and Elsahar, Hady and Rebuffi, Sylvestre-Alvise and Lacatusu, Valeriu and Tran, Tuan and Mourachko, Alexandre},
    journal={arXiv preprint arXiv:TODO},
    year={2025}
}
```
