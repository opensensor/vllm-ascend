# SPDX-License-Identifier: Apache-2.0
"""Portable MTP input normalization for 310P, which cannot run the Triton op."""

import torch


def mtp_eh_norm(positions, embeddings, previous_hidden, embedding_weight, hidden_weight, eps):
    """Match fused_eh_norm's FP32 math and final activation-dtype conversion."""
    embedded = torch.where(positions[:, None] == 0, 0, embeddings).float()
    hidden = previous_hidden.float()
    embedded = embedded * torch.rsqrt(embedded.square().mean(-1, keepdim=True) + eps)
    hidden = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + eps)
    return torch.cat((embedded * embedding_weight.float(), hidden * hidden_weight.float()), dim=-1).to(embeddings.dtype)
