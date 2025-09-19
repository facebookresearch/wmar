# Watermarking Autoregressive Image Generation 🖼️💧

<p align="center">
  <img src="https://dl.fbaipublicfiles.com/wmar/gimmick.gif" alt="Watermarking Demo" width="600"/>
</p>

Official implementation of [Watermarking Autoregressive Image Generation](https://arxiv.org/pdf/2506.16349).
This repository provides a framework for watermarking autoregressive image models, and includes the code to reproduce the main results from the paper. 
In [`wmar_audio`](https://github.com/facebookresearch/wmar/tree/main/wmar_audio) we also provide the code accompanying our case study on Audio (see Section 5 in the paper). 

[[`arXiv`](https://arxiv.org/abs/2506.16349)]
[[`Colab`](https://colab.research.google.com/github/facebookresearch/wmar/blob/main/notebooks/colab.ipynb)]

## 📰 News

- **19th September 2024**: New work on watermark synchronization! Check out [SyncSeal](https://arxiv.org/abs/2506.16349) - an active method for synchronizing images and improving robustness against desynchronization attacks. Code is available in the [`syncseal/`](11-autoregressive/wmar/syncseal) folder.
- **18th September 2024**: Our paper has been accepted to NeurIPS 2025! 🎉


## 💿 Installation

### 1️⃣ Environment
First, clone the repository and enter the directory:
```bash
git clone https://github.com/facebookresearch/wmar
cd wmar
```

Then, set up a conda environment as follows:
```bash
conda create --name wmar python=3.12
conda activate wmar
```

Finally, install xformers (which will include Torch 2.7.0 CUDA 12.6) and other dependencies, and override the triton version (needed for compatibility with Chameleon).
```bash
pip install -U xformers --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install triton==3.1.0 
```

We next describe how to load all autoregressive models, finetuned tokenizer deltas, and other requirements. The simplest way to start is to execute `notebooks/colab.ipynb` (also hosted on [`Colab`](https://colab.research.google.com/github/facebookresearch/wmar/blob/main/notebooks/colab.ipynb)) which downloads only the necessary components from below.
We assume that all checkpoints will be placed under `checkpoints/`.

### 2️⃣ Autoregressive Models

Instructions to download each of the three models evaluated in the paper are given below.

- **Taming**. You need to manually download the transfomer and VQGAN weights following the instructions from the [official repo](https://github.com/CompVis/taming-transformers). In particular, download `cin_transformer` from https://app.koofr.net/links/90cbd5aa-ef70-4f5e-99bc-f12e5a89380e and `VQGAN ImageNet (f=16), 16384` from https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/ and set up the following folder structure under e.g., `checkpoints/2021-04-03T19-39-50_cin_transformer`:
  ```
  checkpoints/
      net2net.ckpt 
      vqgan.ckpt
  configs/
      net2net.yaml
      vqgan.yaml
  ```
  This directory should be also set as `--modeldir` when executing the code (see below).
  To adapt the model configs to the paths in our codebase execute:
  ```bash
  sed -i 's/ taming\./ deps.taming./g' checkpoints/2021-04-03T19-39-50_cin_transformer/configs/vqgan.yaml
  sed -i 's/ taming\./ deps.taming./g' checkpoints/2021-04-03T19-39-50_cin_transformer/configs/net2net.yaml
  ```

- **Chameleon**. Our runs can be reproduced with the open-source alternative Anole, following [these instructions](https://github.com/GAIR-NLP/anole?tab=readme-ov-file). In particular, in your `checkpoints/` run:
  ```bash
  git lfs install
  git clone https://huggingface.co/GAIR/Anole-7b-v0.1
  ```
  And then set `--modelpath` flag when running the models to `checkpoints/Anole-7b-v0.1`. 
  Before this, patch Anole to make it compatible with the Taming codebase (you need to also download Taming above for this step): 
  ```bash
  python -c 'from wmar.utils.utils import patch_chameleon; patch_chameleon("checkpoints/Anole-7b-v0.1")'
  cp assets/chameleon_patched_config.yaml checkpoints/Anole-7b-v0.1/tokenizer/vqgan.yaml
  ```

- **RAR**. RAR-XL is downloaded automatically on the first run; set `--modelpath` to the directory where you want to save the tokenizer and model weights, e.g., `checkpoints/rar`.

### 3️⃣ Deltas of Finetuned Tokenizers

We provide links to weight deltas of the tokenizers finetuned for reverse-cycle-consistency (RCC) that we used in our evaluation in the paper:
| Model | Finetuned | Finetuned+Augmentations |
| --- | --- | --- |
| Taming |  [Encoder](https://dl.fbaipublicfiles.com/wmar/finetunes/taming_encoder_ft_noaug_delta.pth) / [Decoder](https://dl.fbaipublicfiles.com/wmar/finetunes/taming_decoder_ft_noaug_delta.pth) | [Encoder](https://dl.fbaipublicfiles.com/wmar/finetunes/taming_encoder_ft_delta.pth) / [Decoder](https://dl.fbaipublicfiles.com/wmar/finetunes/taming_decoder_ft_delta.pth) |
| Chameleon/Anole | [Encoder](https://dl.fbaipublicfiles.com/wmar/finetunes/chameleon7b_encoder_ft_noaug_delta.pth) / [Decoder](https://dl.fbaipublicfiles.com/wmar/finetunes/chameleon7b_decoder_ft_noaug_delta.pth) | [Encoder](https://dl.fbaipublicfiles.com/wmar/finetunes/chameleon7b_encoder_ft_delta.pth) / [Decoder](https://dl.fbaipublicfiles.com/wmar/finetunes/chameleon7b_decoder_ft_delta.pth) |
| RAR | [Encoder](https://dl.fbaipublicfiles.com/wmar/finetunes/rar_encoder_ft_noaug_delta.pth) / [Decoder](https://dl.fbaipublicfiles.com/wmar/finetunes/rar_decoder_ft_noaug_delta.pth) | [Encoder](https://dl.fbaipublicfiles.com/wmar/finetunes/rar_encoder_ft_delta.pth) / [Decoder](https://dl.fbaipublicfiles.com/wmar/finetunes/rar_decoder_ft_delta.pth) |

To use them, download the files and place them in e.g., `checkpoints/finetunes/`, setting `--encoder_ft_ckpt` and `--decoder_ft_ckpt` flags accordingly when running the code (see below). 
These deltas should be added to the original encoder/decoder weights, which is automatically handled by our loading functions.

Alternatively, you can:
- download them automatically by running: 
  ```bash
  mkdir -p checkpoints/finetunes && cd checkpoints/finetunes && wget -nc https://dl.fbaipublicfiles.com/wmar/finetunes/taming_encoder_ft_noaug_delta.pth https://dl.fbaipublicfiles.com/wmar/finetunes/taming_decoder_ft_noaug_delta.pth https://dl.fbaipublicfiles.com/wmar/finetunes/taming_encoder_ft_delta.pth https://dl.fbaipublicfiles.com/wmar/finetunes/taming_decoder_ft_delta.pth https://dl.fbaipublicfiles.com/wmar/finetunes/chameleon7b_encoder_ft_noaug_delta.pth https://dl.fbaipublicfiles.com/wmar/finetunes/chameleon7b_decoder_ft_noaug_delta.pth https://dl.fbaipublicfiles.com/wmar/finetunes/chameleon7b_encoder_ft_delta.pth https://dl.fbaipublicfiles.com/wmar/finetunes/chameleon7b_decoder_ft_delta.pth https://dl.fbaipublicfiles.com/wmar/finetunes/rar_encoder_ft_noaug_delta.pth https://dl.fbaipublicfiles.com/wmar/finetunes/rar_decoder_ft_noaug_delta.pth https://dl.fbaipublicfiles.com/wmar/finetunes/rar_encoder_ft_delta.pth https://dl.fbaipublicfiles.com/wmar/finetunes/rar_decoder_ft_delta.pth && cd - 
  ```
- or use the `finetune.py` script to finetune the models yourself (see below).

### 4️⃣ Other Requirements

To use watermark synchronization, download [WAM](https://github.com/facebookresearch/watermark-anything): 
```
wget https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth -P checkpoints/
```

To evaluate watermark robustness, download the [DiffPure](https://diffpure.github.io/) model:
```
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt -P checkpoints/
```

## 🎮 Usage

### 1️⃣ Quickstart 

The notebook `colab.ipynb` ([`open in Colab`](https://colab.research.google.com/github/facebookresearch/wmar/blob/main/notebooks/colab.ipynb)) is a good starting point. 
It downloads the necessary components to run watermarked generation with RAR (RAR, finetuned deltas, WAM) and illustrates the robustness of the watermark to transformations.

### 2️⃣ Large-scale generation and evaluation
We describe how to start a larger generation run and the follow-up evaluation and plotting that follows our experimental setup from the paper and reproduces our main results.
We focus on the Taming model, aiming to reproduce Figures 5, 6 and Table 2 in the paper. 
Before starting make sure to follow the relevant parts of the setup above.

For each of the 4 variants evaluated in the paper (_Base_, _FT_, _FT+Augs_, _FT+Augs+Sync_), we generate 1000 watermarked images and apply all the transformations using `generate.py`. 
The 4 corresponding runs are documented in a readable form in `configs/taming_generate.json`. 
For Taming, we provide the corresponding 4 commands in `configs/taming_generate.sh`.
For example, to run _FT+Augs+Sync_, execute:
```bash
python3 generate.py --seed 1 --model taming \
--decoder_ft_ckpt checkpoints/finetunes/taming_decoder_ft_delta.pth \
--encoder_ft_ckpt checkpoints/finetunes/taming_encoder_ft_delta.pth  \
--modelpath checkpoints/2021-04-03T19-39-50_cin_transformer/ \
--wam True --wampath checkpoints/wam_mit.pth \
--wm_method gentime --wm_seed_strategy linear --wm_delta 2 --wm_gamma 0.25 \
--wm_context_size 1 --wm_split_strategy stratifiedrand \
--include_diffpure True --include_neural_compress True \
--top_p 0.92 --temperature 1.0 --top_k 250 --batch_size 5 \
--conditioning 1,9,232,340,568,656,703,814,937,975 \
--num_samples_per_conditioning 100 \
--chunk_id 0 --num_chunks 1 \
--outdir out/0617_taming_generate/_wam=True_decoder_ft_ckpt=2_encoder_ft_ckpt=2
```
Evaluation can be speed up by increasing the batch size, and parallelizing the evaluation using `chunk_id` and `num_chunks` (see `configs/rar_generate.json` for an example).
Each such run will save the outputs under `out/0617_taming_generate`, that we can parse, aggregate, and plot as follows:
```python
from wmar.utils.analyzer import Analyzer
outdir = "out/0617_taming_generate"
watermark = "linear-stratifiedrand-h=1-d=2.0-g=0.25"
methods = {
    # "name": (outdir, relevant_dir_prefix, watermark_as_str)
    "original": (outdir, "_wam=False_decoder_ft_ckpt=0", watermark),
    "finetuned_noaugs": (outdir, "_wam=False_decoder_ft_ckpt=1", watermark),
    "finetuned_augs": (outdir, "_wam=False_decoder_ft_ckpt=2", watermark),
    "finetuned_augs+sync": (outdir, "_wam=True_decoder_ft_ckpt=2", watermark)
}
analyzer = Analyzer(methods, cache_path="assets/cache.json")
analyzer.set_up_latex()
analyzer.plot_l0_hist(save_to=f"{outdir}/l0_hist.png")
analyzer.plot_auc(save_to=f"{outdir}/auc.png")
analyzer.plot_robustness(save_to=f"{outdir}/robustness.png")
```
The same code is also placed in `notebooks/analyze.ipynb` that also shows the result after a successful run, i.e., figures similar to Fig. 5 and Fig. 6 in our paper, and Table 2.

To do the same for other models refer to other config files provided in `configs/`.

### 3️⃣ Finetuning

To repeat the RCC finetuning procedure (instead of using our deltas above), first precompute the tokenized version of the finetuning dataset ([ImageNet](https://image-net.org/download.php)) using the following command (for Taming, adapt first two args for other models):
```bash
python3 precompute_imagenet_codes.py --model taming \
--modelpath checkpoints/2021-04-03T19-39-50_cin_transformer/ \
--imagenet_root data/imagenet/061417/ --outdir out/imagenet_taming
```
where `data/imagenet/061417` points to the ImageNet root which contains `train/`, `val/` and `test/` directories within. The resulting data will be saved to `out/imagenet_taming`.

After this, run `finetune.py` using arguments such as documented in `configs/taming_ft.json`. For Taming, an example command that runs finetuning with DDP on 2 local GPUs using `torchrun` is:
```bash
OMP_NUM_THREADS=40 torchrun --standalone --nnodes=1 --nproc_per_node=2 finetune.py \
--master_port -1 --model taming --modelpath checkpoints/2021-04-03T19-39-50_cin_transformer/ \ 
--dataset codes-imagenet --datapath out/imagenet_taming/codes --dataset_size 50000 \
--mode newenc-dec --nb_epochs 10 --augs_schedule 1,1,4,4 \ 
--optimizer adam --lr 0.0001 --batch_size_per_gpu 4 \ 
--disable_gan --idempotence_loss_weight 1.0 --idempotence_loss_weight_factor 1.0 \ 
--loss hard-to-soft-with-ae --augs all+geom \ 
--outdir out/0617_taming_ft
```
Note that this results in a smaller total batch size than the one we used for the paper, where we train on 16 GPUs.
The finetuning script also downloads the LPIPS checkpoint to `checkpoints/lpips` automatically (needed for perceptual loss).
Final checkpoints will be saved under `outdir` and can be used in evaluation by setting `encoder_ft_ckpt` and `decoder_ft_ckpt` flags as above. 

We provide an example log of a successful finetuning run with Taming in `logs/0620_taming_ft_stdout.txt`.

## 🧾 License

The code is licensed under an [MIT license](LICENSE).
It relies on code and models from other repositories. See the next [Acknowledgements](#acknowledgements) section for the licenses of those dependencies.

## 🫡 Acknowledgements

Some root directories are adapted versions of other repos:
- [Chameleon](https://github.com/facebookresearch/chameleon) in `deps/chameleon/`.
- [RAR](https://github.com/bytedance/1d-tokenizer) in `deps/rar/`.
- [Watermark Robustness](https://github.com/mehrdadsaberi/watermark_robustness) in `deps/saberi_wmr/` (for DiffPure).
- [Taming](https://github.com/CompVis/taming-transformers) in `deps/taming/`.
- [Watermark-Anything](https://github.com/facebookresearch/watermark-anything/) in `deps/watermark_anything/`.
- [Moshi](https://github.com/kyutai-labs/moshi/) in `wmar_audio/moshi/`.

The modifications are primarily done to introduce watermarking and enable finetuning.

Additionally, within `wmar_audio` and `wmar` (marked on top of each file in the latter) some code is taken from:
- [VideoSeal](https://github.com/facebookresearch/videoseal)
- [AudioCraft](https://github.com/facebookresearch/audiocraft/)
- [Moshi](https://github.com/kyutai-labs/moshi)

All of these dependencies are licensed under their respective licenses: 
- MIT license for Taming, Moshi, Audiocraft, VideoSeal, and Watermark-Anything,
- Apache 2.0 for RAR,
- UMD Software Salient ImageNet Copyright (C) 2024 University of Maryland for Watermark Robustness,
- Chameleon License for Chameleon and Anole

Each of the repositories provides their own license for model weights, which are not included in this repository.
We refer to the original repositories for more details on these.


## 🤝 Contributing

See [contributing](.github/CONTRIBUTING.md) and the [code of conduct](.github/CODE_OF_CONDUCT.md).

## 📞 Contact 

Nikola Jovanović, nikola.jovanovic@inf.ethz.ch

Pierre Fernandez, pfz@meta.com

## ✍️ Citation

If you find this repository useful, please consider giving a star :star: and please cite as:

```bibtex
@article{jovanovic2025wmar,
  title={Watermarking Autoregressive Image Generation},
  author={Jovanovi\'{c}, Nikola and Labiad, Ismail and Sou\v{c}ek, Tom\'{a}\v{s} and Vechev, Martin and Fernandez, Pierre},
  journal={arXiv preprint arXiv:2506.16349},
  year={2025}
}
```

