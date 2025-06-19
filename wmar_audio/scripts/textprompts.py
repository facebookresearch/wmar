# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Run with:
    python scripts/textprompts.py --num_prompts 1000 --output_dir /path/to/output
"""

import argparse
import re
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoProcessor

PROMPT = """
You are a creative assistant helping design interesting, in-depth topics for an audio AI to monologue about. The goal is to generate prompts that lead the AI to speak in long, thoughtful, and informative ways, like a mini podcast or TED talk.

Here are some examples of such prompts:

What are the best things to see in Paris?
Tell me about the history of the internet.
Explain how black holes work in simple terms.
Give me a beginner's guide to meditation and mindfulness.

Now, generate 1000 more prompts that would lead to equally engaging AI monologues. 
- Each prompt must be unique and cover a different topic. 
- Each prompt must be a short single sentence.
- Each prompt must start with a new line.
- Each prompt must start with a verb (like "describe", "explain", "talk about", etc.).
- Do not include anything else in the answer, just the prompts.
"""


def parse_prompts(text):
    # Extract lines that look like prompts (non-empty lines that don't contain instructions)
    lines = text.strip().split('\n')
    prompts = [
            line.strip() for line in lines if line.strip() and 
               not line.startswith(('-', '#', '•', '*')) and 
               not any(x in line.lower() for x in ['generate', 'prompt', 'example']) and
               not len(line) < 10 and 
               not len(line) > 100
            ]
    # Remove leading numbers like "1.", "17."
    processed_prompts = []
    for p in prompts:
        # Remove leading number, dot, and optional space
        p_cleaned = re.sub(r"^\d+\.\s*", "", p)
        processed_prompts.append(p_cleaned)
    # remove last line which may be incomplete
    prompts = processed_prompts[:-1] if len(processed_prompts) > 1 else processed_prompts
    return prompts

def generate_prompts(model, processor, prompt_text, max_gen_len, temperature=1.0):
    # Create messages format
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a creative assistant helping generate interesting monologue topics."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}]
        }
    ]
    
    # Apply chat template
    inputs = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=True,
        return_dict=True, 
        return_tensors="pt"
    ).to(model.device)
    # inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    # Get the input length for later slicing
    input_len = inputs["input_ids"].shape[-1]
    
    # Generate output
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            do_sample=True,
            top_p=0.9,
            temperature=temperature,
            pad_token_id=processor.eos_token_id,
        )
        generation = output[0][input_len:]  # Only get the generated part
    
    # Decode and return
    return processor.decode(generation, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_prompts', type=int, required=True, help='Target number of prompts to generate')
    parser.add_argument('--max_gen_len', type=int, default=1024, help='Number of tokens to generate in each batch')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save output files')
    parser.add_argument('--similarity_threshold', type=float, default=0.7, help='Maximum similarity allowed between prompts (0-1)')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for similarity calculation')
    parser.add_argument('--model_id', type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                            help='Model ID to use (e.g., "google/gemma-3-4b-it", "meta-llama/Llama-3.1-70B-Instruct", "meta-llama/Llama-3.1-8B-Instruct)')
    parser.add_argument('--quantize', type=lambda x: x.lower() == 'true', default=False, 
                            help='Whether to quantize the model (True/False)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (higher = more random)')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory to cache downloaded models (optional)')
    args = parser.parse_args()

    # Validate arguments
    if args.num_prompts <= 0:
        raise ValueError("num_prompts must be positive")
    if not 0 < args.similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "prompts.txt"
    
    model_id = args.model_id

    # Configure quantization if requested
    quantization_config = None
    if args.quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # Load model and processor
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        quantization_config=quantization_config
    )
    
    all_prompts = set()
    all_prompt_tokens = []

    pbar = tqdm(total=args.num_prompts, desc="Generating prompts")
    last_count = 0
    total_generated = 0
    total_filtered = 0

    with open(output_path, 'w') as f:
        while len(all_prompts) < args.num_prompts:
            generated_text = generate_prompts(model, processor, PROMPT, args.max_gen_len, args.temperature)
            new_prompts = parse_prompts(generated_text)
            total_generated += len(new_prompts)
            
            # Process each new prompt
            for prompt in new_prompts:
                if prompt in all_prompts or len(all_prompts) >= args.num_prompts:
                    continue
                    
                # Tokenize new prompt
                new_prompt_tokens = tokenizer.tokenize(prompt)
                
                # If we have existing prompts, check similarity
                if all_prompt_tokens:
                    with Pool(args.workers) as p:
                        rouge_scores = p.map(
                            partial(rouge_scorer._score_lcs, new_prompt_tokens),
                            all_prompt_tokens
                        )
                    rouge_scores = [score.fmeasure for score in rouge_scores]
                    max_similarity = max(rouge_scores) if rouge_scores else 0
                    
                    # Skip if too similar to existing prompts
                    if max_similarity > args.similarity_threshold:
                        total_filtered += 1
                        continue
                
                # If we reach here, prompt is unique enough to keep
                all_prompts.add(prompt)
                all_prompt_tokens.append(new_prompt_tokens)
                f.write(prompt + '\n')
                f.flush()
            
            # Update progress bar
            new_count = len(all_prompts)
            if new_count > last_count:
                pbar.update(new_count - last_count)
                last_count = new_count
            
            pbar.set_postfix({
                'unique': len(all_prompts), 
                'generated': total_generated,
                'filtered': total_filtered,
                'keep_rate': f"{len(all_prompts)/max(1, total_generated):.2f}"
            })

    pbar.close()
    print(f"\nGenerated {total_generated} prompts, filtered {total_filtered} similar prompts")
    print(f"Kept {len(all_prompts)} diverse prompts (keep rate: {len(all_prompts)/max(1, total_generated):.2f})")
    print(f"Saved to {output_path}")

if __name__ == "__main__":

    # remove warnings linked to // in HF tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main()
