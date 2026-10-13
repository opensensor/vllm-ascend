# SPDX-License-Identifier: Apache-2.0
"""Owner-local GLM compressed-pool sparse selection with persistent tails."""

HEAD_DIM = 128
POOL_SIZE = 4
MAX_SCORE_ELEMENTS = 1 << 26


def cuda_rotation_enabled(specification):
    value = specification.get("cuda_indexer_hadamard", False)
    if type(value) is not bool:
        raise ValueError("CUDA fused indexer rotation selection must be boolean")
    return value


def hadamard_reference(x, torch):
    value = x.float()
    for stride in (1, 2, 4, 8, 16, 32, 64):
        paired = value.reshape(*value.shape[:-1], HEAD_DIM // (2 * stride), 2, stride)
        value = torch.stack(
            (paired[..., 0, :] + paired[..., 1, :], paired[..., 0, :] - paired[..., 1, :]), -2
        ).reshape_as(value)
    return value * (HEAD_DIM**-0.5)


def select_tokens(scores, positions, topk, torch):
    """Select complete causal pools and append the current incomplete tail."""
    rows, pools = scores.shape
    budget = topk // POOL_SIZE
    completed = (positions + 1) // POOL_SIZE
    pool_ids = torch.arange(pools, device=scores.device)
    masked = scores.masked_fill(pool_ids[None, :] >= completed[:, None], -float("inf"))
    count = min(budget, pools)
    if count:
        selected = torch.topk(masked, count, -1).indices
        valid = (torch.arange(count, device=scores.device)[None, :] < completed[:, None]) & (
            selected < completed[:, None]
        )
        selected = torch.where(valid, selected, -1)
        offsets = torch.arange(POOL_SIZE, device=scores.device)
        expanded = (selected[:, :, None] * POOL_SIZE + offsets).flatten(1)
        expanded = torch.where(selected[:, :, None].expand(-1, -1, POOL_SIZE).flatten(1) >= 0, expanded, -1)
    else:
        expanded = torch.empty((rows, 0), dtype=torch.int64, device=scores.device)
    tail_start = completed * POOL_SIZE
    offsets = torch.arange(POOL_SIZE - 1, device=scores.device)
    tail = tail_start[:, None] + offsets
    tail = torch.where(tail <= positions[:, None], tail, -1)
    return torch.cat((expanded, tail), -1)


class PoolIndexer:
    def __init__(self, weights, config, device, torch, max_context, *, fused_cuda_rotation=False):
        if type(fused_cuda_rotation) is not bool or (fused_cuda_rotation and not device.startswith("cuda")):
            raise ValueError("fused indexer rotation requires a CUDA owner and a boolean selection")
        self.torch, self.device = torch, device
        self.fused_cuda_rotation = fused_cuda_rotation
        self.projection_graph = None
        self.projection_graph_enabled = False
        self.topk = config["index_topk"]
        self.heads = config["index_n_heads"]
        self.weights = weights
        self.pool_count, self.tail_length = 0, 0
        self.pools = torch.empty((max_context // POOL_SIZE, HEAD_DIM), dtype=torch.float16, device=device)
        self.tail_keys = torch.zeros((POOL_SIZE - 1, HEAD_DIM), device=device)
        self.tail_gates = torch.zeros_like(self.tail_keys)
        self.fp32 = {
            name: weights[f"self_attn.indexer.{name}"].float()
            for name in (
                "wk.weight",
                "weights_proj.weight",
                "index_kpool_compress_gate",
                "index_kpool_compress_ape",
                "k_norm.weight",
                "k_norm.bias",
            )
        }
        matrix = torch.ones((1, 1))
        for _ in range(7):
            matrix = torch.cat((torch.cat((matrix, matrix), -1), torch.cat((matrix, -matrix), -1)), -2)
        self.hadamard_matrix = matrix.to(device)

    def reset(self):
        self.pool_count, self.tail_length = 0, 0

    def rotate(self, x):
        t = self.torch
        if self.fused_cuda_rotation:
            # CUDA-only dependencies remain isolated from CPU/NPU owners.
            from .cuda_layer_ops import hadamard

            return hadamard(x)
        # Unnormalized +/-1 operands avoid rounding the normalization constant
        # into 310P FP16 GEMM inputs. Input is already BF16-rounded and finite
        # within the FP16 projection range; materialize that policy explicitly.
        if self.device.startswith("npu"):
            return (x.float() @ self.hadamard_matrix * (HEAD_DIM**-0.5)).to(t.bfloat16).to(t.float16)
        return hadamard_reference(x, t).to(t.bfloat16).to(t.float16)

    def append(self, hidden, projected=None):
        t, w = self.torch, self.fp32
        if projected is None:
            xf = hidden.float()
            keys = t.nn.functional.linear(xf, w["wk.weight"])
            keys = t.nn.functional.layer_norm(keys, (HEAD_DIM,), w["k_norm.weight"], w["k_norm.bias"], 1e-6)
            gates = t.nn.functional.linear(xf, w["index_kpool_compress_gate"])
        else:
            keys, gates = projected
        keys = t.cat((self.tail_keys[: self.tail_length], keys))
        gates = t.cat((self.tail_gates[: self.tail_length], gates))
        complete = keys.shape[0] // POOL_SIZE
        if self.pool_count + complete > self.pools.shape[0]:
            raise ValueError("owner-local compressed indexer cache exceeded")
        if complete:
            full_keys = keys[: complete * POOL_SIZE].reshape(-1, POOL_SIZE, HEAD_DIM)
            full_gates = gates[: complete * POOL_SIZE].reshape(-1, POOL_SIZE, HEAD_DIM)
            probabilities = t.softmax(full_gates + w["index_kpool_compress_ape"], -2)
            compressed = (full_keys * probabilities).sum(-2).to(t.bfloat16).float()
            self.pools[self.pool_count : self.pool_count + complete].copy_(self.rotate(compressed))
            self.pool_count += complete
        self.tail_length = keys.shape[0] - complete * POOL_SIZE
        if self.tail_length:
            self.tail_keys[: self.tail_length].copy_(keys[-self.tail_length :])
            self.tail_gates[: self.tail_length].copy_(gates[-self.tail_length :])

    def select(self, hidden, qr, first_position):
        t = self.torch
        projected = None
        if self.projection_graph_enabled and hidden.shape[0] == 1 and first_position + 1 > self.topk:
            projected = self.projection_graph(hidden, qr)
        self.append(hidden, None if projected is None else projected[:2])
        if first_position + hidden.shape[0] <= self.topk:
            return None
        if projected is None:
            query = t.nn.functional.linear(qr, self.weights["self_attn.indexer.wq_b.weight"])
            query = query.view(-1, self.heads, HEAD_DIM).to(t.bfloat16).float()
            query = self.rotate(query).float()
            weights = t.nn.functional.linear(hidden.float(), self.fp32["weights_proj.weight"])
            weights = weights * (HEAD_DIM**-0.5 * self.heads**-0.5)
        else:
            query, weights = projected[2:]
        keys = self.pools[: self.pool_count].float()
        # Query chunks bound the head-by-query-by-pool FP32 temporary.
        chunk = max(1, MAX_SCORE_ELEMENTS // max(1, self.heads * self.pool_count))
        selections = []
        for start in range(0, hidden.shape[0], chunk):
            end = min(start + chunk, hidden.shape[0])
            logits = query[start:end].reshape(-1, HEAD_DIM) @ keys.t()
            logits = logits.view(end - start, self.heads, -1).relu_()
            scores = logits.mul_(weights[start:end, :, None]).sum(1)
            positions = t.arange(first_position + start, first_position + end, device=self.device)
            selections.append(select_tokens(scores, positions, self.topk, t))
        return t.cat(selections)
