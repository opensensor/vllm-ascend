# SPDX-License-Identifier: Apache-2.0
"""Opt-in uniform padded GDN head shards for nondivisible TP geometry.

Checkpoint heads remain unchanged. Added head groups are zero in both input
and output projections. All ranks allocate the same convolution/recurrent shape.
No device value inspection or runtime host transfer is performed here.
"""

import math
from dataclasses import dataclass

import torch

PARTITION_POLICIES = ("strict", "padded")


def gdn_partition_policy(config):
    metadata = getattr(config, "ascend_expert_quantization", None) or {}
    policy = metadata.get("gdn_head_partition", "strict")
    if policy not in PARTITION_POLICIES:
        raise ValueError("gdn_head_partition must be strict or padded")
    return policy


@dataclass(frozen=True)
class GDNHeadShard:
    key_start: int
    live_key_heads: int
    key_heads: int
    value_heads: int
    value_per_key: int
    key_dim: int
    value_dim: int

    @property
    def value_start(self):
        return self.key_start * self.value_per_key

    @property
    def live_value_heads(self):
        return self.live_key_heads * self.value_per_key

    @property
    def conv_dim(self):
        return 2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim


def gdn_head_shard(key_heads, value_heads, key_dim, value_dim, tp_rank, tp_size, policy="strict"):
    dimensions = (key_heads, value_heads, key_dim, value_dim, tp_size)
    if any(type(n) is not int or n <= 0 for n in dimensions) or type(tp_rank) is not int or not 0 <= tp_rank < tp_size:
        raise ValueError("invalid GDN head partition geometry")
    if value_heads % key_heads or policy not in PARTITION_POLICIES:
        raise ValueError("GDN value/key ratio and partition policy are invalid")
    if policy == "strict" and key_heads % tp_size:
        raise ValueError(f"GDN heads (k={key_heads}, v={value_heads}) not divisible by TP {tp_size}")
    per_rank = (key_heads + tp_size - 1) // tp_size
    first = tp_rank * per_rank
    live = max(0, min(per_rank, key_heads - first))
    ratio = value_heads // key_heads
    return GDNHeadShard(first, live, per_rank, per_rank * ratio, ratio, key_dim, value_dim)


def shard_gdn_tensor(tensor, kind, shard, full_key_heads, full_value_heads):
    """Load-time only: validate shape, select trained heads, zero padded groups."""
    full_k = full_key_heads * shard.key_dim
    full_v = full_value_heads * shard.value_dim
    local_k = shard.key_heads * shard.key_dim
    local_v = shard.value_heads * shard.value_dim
    live_k = shard.live_key_heads * shard.key_dim
    live_v = shard.live_value_heads * shard.value_dim
    start_k = shard.key_start * shard.key_dim
    start_v = shard.value_start * shard.value_dim
    if kind == "norm":
        if tensor.shape != (shard.value_dim,):
            raise ValueError("GDN norm shape mismatch")
        return tensor
    if kind == "out":
        if tensor.ndim != 2 or tensor.shape[1] != full_v:
            raise ValueError("GDN output projection shape mismatch")
        result = tensor.new_zeros((tensor.shape[0], local_v))
        result[:, :live_v].copy_(tensor[:, start_v : start_v + live_v])
        return result
    if kind == "conv" and tensor.ndim == 3 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    expected = {
        "qkv": 2 * full_k + full_v,
        "conv": 2 * full_k + full_v,
        "z": full_v,
        "a": full_value_heads,
        "b": full_value_heads,
        "A_log": full_value_heads,
        "dt_bias": full_value_heads,
    }
    if kind not in expected or tensor.ndim not in (1, 2) or tensor.shape[0] != expected[kind]:
        raise ValueError("GDN checkpoint tensor shape/kind mismatch")
    if kind in ("qkv", "conv"):
        result = tensor.new_zeros((2 * local_k + local_v, *tensor.shape[1:]))
        for source, target, count in (
            (start_k, 0, live_k),
            (full_k + start_k, local_k, live_k),
            (2 * full_k + start_v, 2 * local_k, live_v),
        ):
            result[target : target + count].copy_(tensor[source : source + count])
        return result
    size = local_v if kind == "z" else shard.value_heads
    first = start_v if kind == "z" else shard.value_start
    count = live_v if kind == "z" else shard.live_value_heads
    result = tensor.new_zeros((size, *tensor.shape[1:]))
    result[:count].copy_(tensor[first : first + count])
    return result


def place_padded_gdn_tensor(params, name, tensor, shard, full_key_heads, full_value_heads):
    """Place one checkpoint tensor, clearing all padding even on reload."""
    base = name.replace(".linear_attn.", ".attention.")
    suffixes = (
        (".in_proj_qkv.weight", ".in_proj_qkv", "qkv"),
        (".conv1d.weight", ".conv_weight", "conv"),
        (".in_proj_z.weight", ".in_proj_z", "z"),
        (".in_proj_a.weight", ".in_proj_ba", "a"),
        (".in_proj_b.weight", ".in_proj_ba", "b"),
        (".A_log", ".A_log", "A_log"),
        (".dt_bias", ".dt_bias", "dt_bias"),
        (".norm.weight", ".norm_weight", "norm"),
        (".out_proj.weight", ".out_proj", "out"),
    )
    for source_suffix, target_suffix, kind in suffixes:
        if base.endswith(source_suffix):
            target_name = base[: -len(source_suffix)] + target_suffix
            target = params.get(target_name)
            if target is None:
                return None
            source = shard_gdn_tensor(tensor, kind, shard, full_key_heads, full_value_heads)
            if kind in ("a", "b"):
                target = target[shard.value_heads :] if kind == "a" else target[: shard.value_heads]
            if source.shape != target.shape:
                raise ValueError("GDN destination shape mismatch")
            with torch.no_grad():
                target.copy_(source.to(target.dtype))
            return target_name
    return None


def vocab_partition_padding(config, tp_size, base_padding):
    """Keep trained vocabulary size; pad both embedding and head to TP alignment."""
    if type(tp_size) is not int or tp_size <= 0 or type(base_padding) is not int or base_padding <= 0:
        raise ValueError("invalid vocabulary partition alignment")
    return math.lcm(base_padding, tp_size) if gdn_partition_policy(config) == "padded" else base_padding
