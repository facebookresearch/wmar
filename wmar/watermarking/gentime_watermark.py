# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.​

import collections
import math
from enum import Enum
from functools import lru_cache, partial
from itertools import chain, tee
from typing import Union  # for Python <3.10 compatibility

import numpy as np
import torch
from loguru import logger
from scipy import special
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from deps.taming.modules.vqvae.quantize import VectorQuantizer


##########################################################################
# Ngram iteration from nltk, extracted to remove the dependency
# Natural Language Toolkit: Utility functions
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Eric Kafe <kafe.eric@gmail.com> (acyclic closures)
# URL: <https://www.nltk.org/>
# For license information, see https://github.com/nltk/nltk/blob/develop/LICENSE.txt
##########################################################################
def ngrams(sequence, n, pad_left=False, pad_right=False, pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (pad_symbol,) * (n - 1))
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
    return zip(*iterables)  # Unpack and flattens the iterables.


def spatial_ngrams(sequence, n):
    """
    Generate spatial n-grams from a sequence by treating it as a square image.
    For n=4, treats each 2x2 subsquare in the 16x16 image as a 4-gram.

    Args:
        sequence: Input sequence of length 256
        n: Must be 4 for 2x2 spatial blocks
        pad_left, pad_right, pad_symbol: Kept for API compatibility but not used

    Returns:
        Iterator yielding 4-tuples representing 2x2 spatial blocks
    """
    if n not in [2, 4]:
        raise ValueError("This spatial n-gram implementation only supports n=4 (2x2 blocks) or n = 2 (1x2 blocks)")

    sqrt = int(math.sqrt(len(sequence)))
    assert sqrt * sqrt == len(sequence), "Sequence must be a square"

    # Reshape the sequence into 16x16 grid
    grid = [sequence[i : i + sqrt] for i in range(0, len(sequence), sqrt)]

    if n == 2:
        # Generate all 2x2 blocks
        for i in range(0, sqrt):  # -1 because we need room for 2x2 block
            for j in range(0, sqrt):
                if j == 0:
                    if i > 0:
                        yield (grid[i - 1][j], grid[i][j])
                else:
                    yield (grid[i][j - 1], grid[i][j])
    else:
        # Generate all 2x2 blocks
        for i in range(0, sqrt - 1):  # -1 because we need room for 2x2 block
            for j in range(0, sqrt - 1):
                # Yield the 2x2 block as a 4-tuple in row-major order
                yield (
                    grid[i][j],  # top-left
                    grid[i][j + 1],  # top-right
                    grid[i + 1][j],  # bottom-left
                    grid[i + 1][j + 1],  # bottom-right
                )


# NOTE: This is needed to load this from notebooks

if "SeedStrategy" not in globals():

    class SeedStrategy(Enum):
        FIXED = "fixed"
        LINEAR = "linear"
        SPATIAL = "spatial"


if "SplitStrategy" not in globals():

    class SplitStrategy(Enum):
        RANDOM = "rand"
        RANDOM_STRATIFIED = "stratifiedrand"
        CLUSTERING = "clustering"


class GentimeWatermark:
    def __init__(
        self,
        vq: Union[VectorQuantizer, dict],  # sometimes it's a dict
        vocab_size: int,
        seed_strategy: SeedStrategy,
        split_strategy: SplitStrategy,
        context_size: int,
        delta: float,
        gamma: float,
        device="cpu",
        spatial_dim=16,
        salt_key=15485863,
    ) -> None:
        self.device = device

        # VQ params
        self.vocab_size = vocab_size  # just VQ for taming, full vocab for chameleon
        if isinstance(vq, dict):
            self.alive_ids = vq["alive_ids"].to(device)
            self.dead_ids = vq["dead_ids"].to(device)
            self.embedding = vq["embedding"].to(device)
            self.embedding_dim = self.embedding.shape[1]
        else:
            self.alive_ids = vq.alive_ids.to(device)
            self.dead_ids = vq.dead_ids.to(device)
            self.embedding = vq.embedding.weight.to(device)  # [vocab_size x embedding_dim]
            self.embedding_dim = vq.embedding.weight.shape[1]

        # WM params
        self.salt_key = salt_key
        self.seed_strategy = seed_strategy
        self.split_strategy = split_strategy
        self.context_size = context_size
        self.delta = delta
        self.gamma = gamma
        self.greenlist_size = int(self.vocab_size * self.gamma)
        print(f"Greenlist size: {self.greenlist_size} for vocab sz {self.vocab_size} and gamma {self.gamma}")
        self.rng = torch.Generator(device="cpu")  # always and then move
        if self.seed_strategy == SeedStrategy.FIXED:
            self.fixed_greenlist = self._split_with_seed(0)  # Always 0
        else:
            self.fixed_greenlist = None

        # 16 for Taming/RAR and 32 for Cham
        self.spatial_dim = spatial_dim

    def __str__(self):
        ret = f"{self.seed_strategy.value}-{self.split_strategy.value}-"
        ret += f"h={self.context_size}-d={self.delta:.1f}-g={self.gamma:.2f}"
        return ret

    def _split_with_seed(self, seed: int) -> torch.LongTensor:
        self.rng.manual_seed(seed)
        if self.split_strategy is SplitStrategy.RANDOM:
            self.vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng).to(self.device)
            greenlist_ids = self.vocab_permutation[: self.greenlist_size]
        elif self.split_strategy is SplitStrategy.RANDOM_STRATIFIED:
            alive_ids_shuf = self.alive_ids[
                torch.randperm(len(self.alive_ids), generator=self.rng, device="cpu").tolist()
            ]
            dead_ids_shuf = self.dead_ids[torch.randperm(len(self.dead_ids), generator=self.rng, device="cpu").tolist()]
            real_nb_green_alive_at_gamma = len(alive_ids_shuf) * self.gamma
            nb_green_alive = int(real_nb_green_alive_at_gamma)
            nb_green_dead = self.greenlist_size - nb_green_alive
            greenlist_ids = torch.cat([alive_ids_shuf[:nb_green_alive], dead_ids_shuf[:nb_green_dead]])
        elif self.split_strategy is SplitStrategy.CLUSTERING:
            logger.info("Splitting with clustering strategy")
            assert seed == 0 and self.seed_strategy == SeedStrategy.FIXED, "Clustering only with fixed seeding"

            # Get alive embeddings
            alive_embeddings = self.embedding[self.alive_ids]
            alive_embeddings_flat = alive_embeddings.view(alive_embeddings.shape[0], -1).detach().cpu().numpy()
            tsne = TSNE(n_components=2, random_state=42)
            alive_embeddings_tsne = tsne.fit_transform(alive_embeddings_flat)

            # Initialize KMeans
            kmeans = KMeans(n_clusters=100, random_state=42)
            kmeans.fit(alive_embeddings_tsne)

            # Kmeans centers
            centers = kmeans.cluster_centers_
            labels = np.arange(len(centers))

            # Sort by y
            ysort = np.argsort(centers[:, 1])
            centers = centers[ysort]
            labels = labels[ysort]

            # Take groups of 10, sort by x and alternate
            centers = centers.reshape(-1, 10, 2)
            labels = labels.reshape(-1, 10)
            curr = 0
            label_to_color = {}
            for i in range(centers.shape[0]):
                curr = 1 - curr
                xsort = np.argsort(centers[i, :, 0])
                centers[i] = centers[i, xsort]
                labels[i] = labels[i][xsort]
                for lab, cen in zip(labels[i], centers[i]):
                    label_to_color[lab] = curr
                    curr = 1 - curr

            # Add alives from even clusters and just even deads
            greenlist_ids = [idd for i, idd in enumerate(self.alive_ids) if label_to_color[int(kmeans.labels_[i])] == 1]
            greenlist_ids += [idd for idd in self.dead_ids if idd % 2 == 0]
            greenlist_ids = torch.tensor(greenlist_ids, dtype=torch.long).to(self.device)
        return greenlist_ids

    # Accepts the context
    def _get_greenlist_ids_for_context(self, context: torch.LongTensor) -> tuple[torch.LongTensor, int]:
        assert context.ndim <= 1, "context must be a non-batched tensor"
        assert len(context) == self.context_size, f"context must be of length {self.context_size}"
        if self.seed_strategy is SeedStrategy.FIXED:
            return self.fixed_greenlist
        else:
            seed = (self.salt_key * context.sum().item()) % (2**64 - 1)
            return self._split_with_seed(seed)

    # past_ids: [B, len], logits: [B, vocab_size]
    def _process_logits(self, past_ids: torch.LongTensor, logits: torch.Tensor) -> torch.Tensor:
        assert logits.shape[-1] == self.vocab_size, f"Logits shape mismatch: {logits.shape} vs {self.vocab_size}"
        for b_idx in range(past_ids.shape[0]):
            try:
                # Get the context
                if self.seed_strategy is SeedStrategy.FIXED:
                    context = torch.LongTensor([]).to(self.device)
                elif self.seed_strategy is SeedStrategy.LINEAR:
                    if len(past_ids[b_idx]) < self.context_size:
                        raise ValueError(
                            f"past_ids size {len(past_ids)} is smaller than the context size {self.context_size}"
                        )
                    context = past_ids[b_idx][-self.context_size :]
                elif self.seed_strategy is SeedStrategy.SPATIAL:
                    assert self.context_size in [1, 3], "Spatial seeding only implemented for context size in [1,3]"
                    if self.context_size == 3:
                        if len(past_ids[b_idx]) < self.spatial_dim + 1:
                            raise ValueError(
                                f"past_ids size {len(past_ids)} is smaller than needed {self.spatial_dim+1}"
                            )
                        idxs = [-self.spatial_dim - 1, -self.spatial_dim, -1]
                        wide_context = past_ids[b_idx][idxs]
                        context = wide_context[-self.context_size :]
                    elif self.context_size == 1:
                        if len(past_ids[b_idx]) < self.context_size:
                            raise ValueError(
                                f"past_ids size {len(past_ids)} is smaller than the context size {self.context_size}"
                            )
                        # Context depends on the location
                        if len(past_ids[b_idx]) % self.spatial_dim == 0:
                            # we are just starting a new row
                            context = past_ids[b_idx][-self.spatial_dim : -self.spatial_dim + 1]
                        else:
                            context = past_ids[b_idx][-1:]
                        print(f"CONTEXT: {context}")
                else:
                    raise ValueError(f"Invalid seed strategy: {self.seed_strategy}")
                greenlist_ids = self._get_greenlist_ids_for_context(context)
                logits[b_idx, greenlist_ids] += self.delta
            except ValueError:
                # Can't apply the watermark here
                continue
        return logits

    # Spawn logit processor
    def spawn_logit_processor(self):
        return partial(self._process_logits)

    # Cached
    @lru_cache(maxsize=2**32)
    def _get_ngram_score_cached(self, context: tuple[int], target: int):
        """Expensive re-seeding and sampling is cached."""
        greenlist_ids = self._get_greenlist_ids_for_context(torch.as_tensor(context, device=self.device))
        return 1 if target in greenlist_ids else 0

    # codes: [len] of ids in [0, vocab_size-1]
    def _score_ngrams_in_passage(self, codes: torch.Tensor, return_mask: bool = False):
        """Core function to gather all ngrams in the input and compute their watermark."""
        if len(codes) - self.context_size < 1:
            raise ValueError(
                f"Must have at least {1} token to score after the first"
                f" min_context_len={self.context_size} tokens required by the seeding scheme."
            )

        # Compute scores for all ngrams contexts in the passage:
        if self.seed_strategy == SeedStrategy.SPATIAL:
            token_ngram_generator = spatial_ngrams(codes.cpu().tolist(), self.context_size + 1)
        else:
            token_ngram_generator = ngrams(codes.cpu().tolist(), self.context_size + 1)
        token_ngram_generator = list(token_ngram_generator)

        frequencies_table = collections.Counter(token_ngram_generator)
        ngram_to_watermark_lookup = {}
        L = len(frequencies_table.keys())
        for idx, ngram_example in enumerate(frequencies_table.keys()):
            context, target = ngram_example[:-1], ngram_example[-1]
            ngram_to_watermark_lookup[ngram_example] = self._get_ngram_score_cached(context, target)

        if return_mask:
            mask = [-1 for _ in range(self.context_size)]
            seen = set()
            for ngram in token_ngram_generator:
                if ngram in seen:
                    mask.append(-1)  # not scored
                    continue
                seen.add(ngram)
                mask.append(ngram_to_watermark_lookup[ngram])
            return ngram_to_watermark_lookup, frequencies_table, mask

        return ngram_to_watermark_lookup, frequencies_table

    # codes: [B, len] of ids in [0, vocab_size-1] -> same lengths
    # returns p-values of greenlist hit
    def detect(self, codes: torch.LongTensor, return_masks: bool = False) -> float:
        pvals = []
        masks = []
        for b_idx in range(codes.shape[0]):
            curr_codes = codes[b_idx]

            if return_masks:
                ngram_to_watermark_lookup, frequencies_table, mask = self._score_ngrams_in_passage(
                    curr_codes, return_masks
                )
                masks.append(mask)
            else:
                ngram_to_watermark_lookup, frequencies_table = self._score_ngrams_in_passage(curr_codes, return_masks)
            n_scored = len(frequencies_table.keys())
            n_green = sum(ngram_to_watermark_lookup.values())

            pval = special.betainc(n_green, 1 + n_scored - n_green, self.gamma)
            pvals.append(pval)

        if return_masks:
            return torch.tensor(pvals).to(self.device), masks
        else:
            return torch.tensor(pvals).to(self.device)


# For example: fixed-stratifiedrand-h=0-d=8.0-g=0.50
def create_watermarker_from_string(vq: Union[VectorQuantizer, dict], vocab_size: int, method: str, device: str) -> GentimeWatermark:
    # split by -
    parts = method.split("-")
    seed_strategy = parts[0]
    split_strategy = parts[1]
    context_size = int(parts[2].split("=")[1])
    delta = float(parts[3].split("=")[1])
    gamma = float(parts[4].split("=")[1])

    return GentimeWatermark(
        vq,
        vocab_size,
        SeedStrategy(seed_strategy),
        SplitStrategy(split_strategy),
        context_size,
        delta,
        gamma,
        device=device,
    )
