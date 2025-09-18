# WMAR Audio

We present the code for the audio study of the paper [Watermarking Autoregressive Image Generation](https://arxiv.org/abs/2506.16349). 
The analysis and code of this folder is based on Moshi, a framework for duplex text-speech models: [[paper](https://kyutai.org/Moshi.pdf)],
[[main_repo](https://github.com/kyutai-labs/moshi)] 

## Requirements

The audio code requires specific requirements to run properly, which can be installed via:
```bash
pip install -r requirements.txt
```


## Data and models preparation

### Audio prompts for Moshi

The data preparation process consists of two main steps: generating text prompts and converting them to audio prompts.

**Step 1: Generate Text Prompts**

Use the `textprompts.py` script to generate diverse text prompts using an LLM.
It creates monologue topics that usually leads to answers >10 seconds.
Example:
```bash
python scripts/textprompts.py \
    --num_prompts 1000 \
    --output_dir /path/to/text_prompts \
    --seed 42
```

The script automatically filters out similar prompts using ROUGE-L similarity scoring to ensure diversity in the generated dataset (some topics are still quite similar).
It should output a file named `prompts.txt` in the specified output directory.

**Step 2: Convert Text to Audio Prompts**

Use the `audioprompts.py` script to convert the generated text prompts into audio using SeamlessM4T-v2:
```bash
python scripts/audioprompts.py \
    --prompt_file /path/to/text_prompts/prompts.txt \
    --output_dir /path/to/audio_prompts \
    --device cuda
```

For large datasets, you can use chunking which will process only some prompts (and run the commands in parallel):
```bash
python scripts/audioprompts.py \
    --prompt_file /path/to/text_prompts/prompts.txt \
    --output_dir /path/to/audio_prompts \
    --device cuda \
    --chunk_idx 0 \ # ... and same cmd for chunks 1, 2 and 3
    --total_chunks 4
```

It should output audio files and text files in the specified output directory, with the following naming convention:
- Audio files: `prompt_00000.wav`, `prompt_00001.wav`, etc. (16kHz WAV format)
- Text files: `prompt_00000.txt`, `prompt_00001.txt`, etc. (corresponding prompts)


### VoxPopuli dataset

The VoxPopuli dataset is used for finetuning the Mimi model and for some evaluation of the paper.
It can be downloaded from the [VoxPopuli repository](https://github.com/facebookresearch/voxpopuli).


### Deltas of finetuned tokenizers

We provide links to delta weights of the tokenizers finetuned for reverse-cycle-consistency (RCC) used in the paper: 
| Finetuned | Finetuned with augmentations |
| --- | --- |
| [Encoder/Decoder](https://dl.fbaipublicfiles.com/wmar/finetunes/mimi_ft_noaug_delta.pth) | [Encoder/Decoder](https://dl.fbaipublicfiles.com/wmar/finetunes/mimi_ft_delta.pth) |

These deltas are the difference between the finetuned and original encoder/decoder weights.
To use them, download the files and put them in `checkpoints/finetunes/`, then update the weights of the original decoder/encoder by adding the delta weights to the original ones.
To do this, you can use the `apply_deltas.py` script:

```bash
# Download the delta files first (example URLs from the table above)
mkdir -p checkpoints/finetunes
wget https://dl.fbaipublicfiles.com/wmar/finetunes/mimi_ft_delta.pth -O checkpoints/finetunes/mimi_ft_delta.pth
wget https://dl.fbaipublicfiles.com/wmar/finetunes/mimi_ft_noaug_delta.pth -O checkpoints/finetunes/mimi_ft_noaug_delta.pth

# Apply both encoder and decoder deltas to reconstruct the finetuned model
python -m training.apply_deltas \
    --delta_path checkpoints/finetunes/mimi_deltas_ft_noaug.pth \
    --output_dir checkpoints/finetunes/ \
    --output_name mimi_ft_noaug.pth
python -m training.apply_deltas \
    --delta_path checkpoints/finetunes/mimi_deltas_ft.pth \
    --output_dir checkpoints/finetunes/ \
    --output_name mimi_ft.pth
```

The script will automatically download the original MIMI model, apply the specified deltas, and save the reconstructed model state dictionary in the `checkpoints/finetunes/` directory.



## Main evaluations

Main evaluations of the paper are given in the `evals` folder.

### Watermark evaluation

The watermark evaluation script (`evals/main_wm.py`) generates audio with watermarks and evaluates detection performance.

**Basic watermark evaluation:**
```bash
python -m evals.main_wm \
    --output_dir outputs/wm_eval \
    --audio_dir /path/to/audio_prompts \
    --nsamples 100 \
    --batch_size q \
    --temperature 1.0 \
    --steps 200 \
    --wm_method maryland \
    --wm_streams 1 2 3 4 \
    --wm_delta 2.0 \
    --mimi_weight checkpoints/finetunes/mimi_ft.pth
```

*Key parameters:*
- `--wm_method`: Watermarking method (`maryland`, `gumbel`, `none`)
- `--wm_streams`: Which audio streams to watermark (choose between 1 and 8, 0 represents the text stream)
- `--wm_delta`: Watermark strength (higher = stronger watermark, lower audio quality)
- `--wm_ngram`: N-gram context for watermarking (0 = no context)
- `--wm_seed`: Random seed for watermark generation
- `--steps`: Number of generation steps (200 steps is around 10 seconds of audio)
- `--temperature`: Sampling temperature for generation

> [!TIP]
> At the moment, we recommand using batch size of 1 as larger batch sizes may lead to excessive silence in the prompt and to out of distribution samples for Moshi (since we pad  the audio to the maximum length in the batch).


### Token-match evaluation

The token match evaluation script (`evals/token_match.py`) tests how well models preserve tokens through decode-encode.

We provide two modes:
- `moshi`: evaluation of token match on sequences generated by the Moshi model
- `mimi`:  evaluation of token match on sequences generated by the Mimi tokenizer, from audio dataset (e.g. VoxPopuli)

Examples:
```bash
python -m evals.token_match \
    --mode moshi \
    --output_dir outputs/tm \
    --audio_dir /path/to/audio_prompts \
    --nsamples 50 \
    --batch_size 1 \
    --temperature 1.0 \
    --steps 200

python -m evals.token_match \
    --mode mimi \
    --output_dir outputs/idem_mimi \
    --audio_dir /path/to/voxpopuli \
    --nsamples 100 \
    --mimi_weight checkpoints/finetunes/mimi_ft.pth \
    --eval_aug true
```

*Key parameters:*
- `--mode`: Model to evaluate (`moshi` for full model, `mimi` for tokenizer only)
- `--eval_aug`: Whether to test robustness with audio augmentations
- `--save_audio`: Number of audio samples to save for inspection
- `--duration_sec`: Duration of audio clips to process


## Tokenizer finetuning

The finetuning script (`training/finetune_mimi.py`) trains the Mimi tokenizer for better reverse-cycle-consistency.

Example:
```bash
torchrun --nproc_per_node=2 -m training.finetune_mimi \
    --audio_dir /path/to/voxpopuli \
    --output_dir outputs/finetune \
    --target_duration 10.0 \
    --learning_rate 1e-5 \
    --epochs 10 \
    --batch_size 32 \
    --code_loss_weight 1.0 \
    --audio_loss_weight 1e-3
```

or with augmentations:
```bash
torchrun --nproc_per_node=2 -m training.finetune_mimi \
    --audio_dir /path/to/voxpopuli \
    --output_dir outputs/finetune_aug \
    --augs '{"identity":1,"lowpass_filter":1,"highpass_filter":1,"noise_injection":1,"pink_noise":1}' \
    --augs_params '{"lowpass_filter":{"min_cutoff_freq":2000,"max_cutoff_freq":6000},"noise_injection":{"min_noise_std":0.005,"max_noise_std":0.015}}'
```

*Key parameters:*
- `--target_duration`: Audio clip duration (should be multiple of 80ms)
- `--code_loss_weight`: Weight for code reconstruction loss (main objective)
- `--audio_loss_weight`: Weight for audio reconstruction loss
- `--code_target_type`: Code loss target (`pre_q`, `post_q`, or `indices`)
- `--audio_loss_type`: Audio loss function (`mrstft`, `stft`, `sisnr`)
- `--finetune_encoder`: Whether to finetune encoder (default: true)
- `--augs`: JSON string defining augmentation types and weights
- `--resume_from`: Path to checkpoint for resuming training

We give other commands in the header of the `training/finetune_mimi.py` script.
