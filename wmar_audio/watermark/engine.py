# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
 
from moshi.utils.sampling import sample_token

GENERATOR = torch.Generator(device="cpu")

def get_wm_window_hash(ngrams: torch.Tensor = None, seed: int = 0) -> torch.Tensor:
    """Get watermarking window hash."""
    # Get the hash of the ngrams
    batch_size, wm_ngram = ngrams.shape
    if wm_ngram == 0:
        return torch.full((batch_size,), seed, dtype=torch.int64)
    else:
        window_hash = torch.zeros(batch_size, dtype=torch.int64)
        for bsz in range(batch_size):
            GENERATOR.manual_seed(seed)
            window_hash[bsz] = torch.randint(0, 2**31 - 1, (1,), GENERATOR=GENERATOR).item()
            for ii in range(wm_ngram):
                window_hash[bsz] ^= ngrams[bsz, ii].item()
        return window_hash


def gumbel_sample(
    logits: torch.Tensor, 
    window_hash: torch.Tensor,  # b
    use_sampling: bool = False,
    temp: float = 1.0,
    top_p: float = 0.0,
    top_k: int = 0,
) -> torch.Tensor:
    """Aaronson-style watermarking sampling method."""
    if not (use_sampling and temp > 0.0):
        return torch.argmax(logits, dim=-1)
        
    probs = torch.softmax(logits / temp, dim=-1)
    if top_p > 0.0:
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = probs_sort
        need_remap = True
    elif top_k > 0:
        topk_probs, topk_idx = torch.topk(probs, min(top_k, probs.shape[-1]), dim=-1)
        probs = torch.full_like(probs, 1e-6)
        probs.scatter_(-1, topk_idx, topk_probs)
        probs.div_(probs.sum(dim=-1, keepdim=True))
        need_remap = False
        probs_idx = None
    else:
        need_remap = False
        probs_idx = None
    
    # Create batched random values using different seeds
    batch_size = logits.shape[0]
    rps = torch.empty_like(probs)  # b v
    for bsz in range(batch_size):
        GENERATOR.manual_seed(window_hash[bsz].item())
        rs = torch.rand(probs[bsz].shape, generator=GENERATOR).to(probs.device)
        if need_remap:
            rs = torch.gather(rs, -1, probs_idx[bsz])
        rps[bsz] = torch.pow(rs, 1/probs[bsz])  # v
    
    # Select token per batch
    next_token = torch.argmax(rps, dim=-1)
    if need_remap:
        next_token = torch.stack([probs_idx[b, next_token[b]] for b in range(batch_size)])
    return next_token


def maryland_sample(
    logits: torch.Tensor, 
    window_hash: torch.Tensor,  # shape: (b,)
    use_sampling: bool = False,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    gamma: float = 0.5, 
    delta: float = 1.0
) -> torch.Tensor:
    """Maryland-style watermarking sampling method."""
    vocab_size = logits.shape[-1]
    batch_size = logits.shape[0]

    # Create batch-specific greenlist and bias
    bias = torch.zeros_like(logits)  # b 1 1 v
    for bsz in range(batch_size):
        GENERATOR.manual_seed(window_hash[bsz].item())
        vocab_perm = torch.randperm(vocab_size, generator=GENERATOR).to(logits.device)
        greenlist = vocab_perm[:int(gamma * vocab_size)]
        deltas = torch.zeros(vocab_size, device=logits.device)  # v
        deltas[greenlist] = delta
        bias[bsz] = deltas  # v --> 1 1 v
    
    # Sample using modified logits
    modified_logits = logits + bias
    return sample_token(modified_logits, use_sampling, temp, top_k, top_p)


def maryland_score_tok(
    tokens: torch.Tensor, 
    window_hash: torch.Tensor,  # shape: (b,)
    vocab_size: int,
    gamma: float = 0.5,
) -> torch.Tensor:
    """Maryland-style watermarking detection method."""
    scores = torch.zeros_like(tokens)  # b
    for bsz in range(tokens.shape[0]):
        GENERATOR.manual_seed(window_hash[bsz].item())
        vocab_perm = torch.randperm(vocab_size, generator=GENERATOR).to(tokens.device)
        greenlist = vocab_perm[:int(gamma * vocab_size)]
        scores[bsz] = tokens[bsz] in greenlist
    return scores


def gumbel_score_tok(
    tokens: torch.Tensor, 
    window_hash: torch.Tensor,  # shape: (b,)
    vocab_size: int,
) -> torch.Tensor:
    """gumbel-style watermarking detection method."""
    scores = torch.zeros_like(tokens)  # b
    for bsz in range(tokens.shape[0]):
        GENERATOR.manual_seed(window_hash[bsz].item())
        rs = torch.rand(vocab_size, generator=GENERATOR) # n
        scores[bsz] = -(1 - rs).log()[tokens[bsz]]
    return scores


def wm_sample_token(
    logits: torch.Tensor,  # b 1 1 v
    use_sampling: bool = False,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    method: str = "gumbel",
    window_hash: torch.Tensor = None,  # b
    aux_params: dict = None,
) -> torch.Tensor:
    """Given logits of shape [*, Card], returns a LongTensor of shape [*]."""
    if window_hash is None:
        assert method == "none", f"window_hash is required for {method} sampling"
    if method == "gumbel":
        return gumbel_sample(logits, window_hash, use_sampling, temp, top_p, top_k)
    elif method == "maryland":
        gamma = aux_params.get("gamma", 0.5)
        delta = aux_params.get("delta", 1.0)
        return maryland_sample(logits, window_hash, use_sampling, temp, top_k, top_p, gamma, delta)
    else:
        return sample_token(logits, use_sampling, temp, top_k, top_p)

