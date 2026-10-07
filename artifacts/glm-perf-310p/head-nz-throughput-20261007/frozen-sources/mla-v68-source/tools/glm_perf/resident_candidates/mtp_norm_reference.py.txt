"""Reference candidate to qualify patching and recapture without changing math."""

import torch


def normalize(positions, embeddings, previous_hidden, embedding_weight, hidden_weight, eps):
    embedded = torch.where(positions[:, None] == 0, 0, embeddings).float()
    hidden = previous_hidden.float()
    embedded = embedded * torch.rsqrt(embedded.square().mean(-1, keepdim=True) + eps)
    hidden = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + eps)
    return torch.cat((embedded * embedding_weight.float(), hidden * hidden_weight.float()), dim=-1).to(embeddings.dtype)


def replacements():
    # The MTP module imports this function by value. Patch the call-site alias
    # as well as the defining module so the existing model sees the candidate.
    return {
        "vllm_ascend.models.glm5next.mtp:mtp_eh_norm": normalize,
        "vllm_ascend.models.glm5next.ops.mtp_norm:mtp_eh_norm": normalize,
    }
