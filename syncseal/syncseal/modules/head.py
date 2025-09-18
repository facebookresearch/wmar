# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# SAM

import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        out_dim: int = 1,
        sigmoid_output: bool = False,
    ) -> None:
        """
        Predicts outputs given an image embedding, using a simple MLP.

        Arguments:
            embed_dim (int): the input channel dimension
            out_dim (int): the number of bits to predict (0 for zero-bit)
            sigmoid_output (bool): whether to apply sigmoid to the output
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(embed_dim, 1 + self.out_dim)
        self.sigmoid_output = sigmoid_output
            
    def forward(
        self,
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict outputs given image embeddings.

        Arguments:
            image_embeddings (torch.Tensor): the embeddings from the image encoder

        Returns:
            torch.Tensor: batched predictions (1+out)
        """
        # Average the spatial dimensions
        x = image_embeddings.mean(dim=[-2, -1])  # b c
        preds = self.linear(x)    # b c -> b 1+out

        # Apply sigmoid if needed and return
        if self.sigmoid_output: 
            return F.sigmoid(preds)
        return preds
