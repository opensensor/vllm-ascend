# SPDX-License-Identifier: Apache-2.0
"""Offline six-chip Qwen TP profile; no launcher, device runtime or capacity claims."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from vllm_ascend.models.qwen4_exp.head_partition import gdn_execution_shard, gdn_head_shard, vocab_partition_padding
from vllm_ascend.models.qwen4_exp.qsa_head_sharding import qsa_head_shard
from vllm_ascend.models.qwen4_exp.shared_partition import shared_expert_range
from vllm_ascend.models.qwen4_exp.w4_moe import w4_config
from vllm_ascend.models.qwen4_exp.weight_mapping import local_expert_range

TP_SIZE = 6
GDN_STATE_DTYPE_BYTES = 4
FP16_BYTES = 2
NATIVE_METADATA_BANKS = 3
GROUP_SIZE = 128
DEFAULT_VOCAB_PADDING = 64


def make_profile(config):
    """Retain checkpoint/vision geometry; opt into padded text heads explicitly."""
    result = copy.deepcopy(config)
    text = result.get("text_config", result)
    required = (
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "num_experts",
        "hidden_size",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "num_hidden_layers",
        "layer_types",
        "vocab_size",
    )
    if any(key not in text for key in required):
        raise ValueError("incomplete Qwen geometry")
    metadata = text.get("ascend_expert_quantization")
    if not isinstance(metadata, dict) or metadata.get("bits") != 4 or metadata.get("group_size") != GROUP_SIZE:
        raise ValueError("six-chip profile requires the packed W4 G128 checkpoint")
    metadata.update(
        gdn_head_partition="padded_compact",
        shared_expert_execution="tp_sharded_uneven",
        backend="cube_310_int4_a8",
        lm_head_execution="float16",
        activation_quantization="int8_per_group",
        grouped_activation="cann_builtin_fp16",
        grouped_finalize="cann_v2",
        grouped_prefill_chunk_tokens=2560,
    )
    policy = SimpleNamespace(**text)
    w4_config(policy)
    padding = vocab_partition_padding(policy, TP_SIZE, DEFAULT_VOCAB_PADDING)
    shards = []
    for rank in range(TP_SIZE):
        gdn = gdn_head_shard(
            policy.linear_num_key_heads,
            policy.linear_num_value_heads,
            policy.linear_key_head_dim,
            policy.linear_value_head_dim,
            rank,
            TP_SIZE,
            "padded_compact",
        )
        live = gdn_execution_shard(gdn)
        shared_start, shared_stop = shared_expert_range(policy.shared_expert_intermediate_size, rank, TP_SIZE)
        qsa = qsa_head_shard(policy.num_attention_heads, policy.num_key_value_heads, policy.head_dim, rank, TP_SIZE)
        first, stop = local_expert_range(policy.num_experts, TP_SIZE, rank)
        hidden, intermediate = policy.hidden_size, policy.moe_intermediate_size
        if hidden % GROUP_SIZE or intermediate % GROUP_SIZE:
            raise ValueError("native projection geometry does not satisfy G128")
        weights = 3 * hidden * intermediate // 2
        meta = 3 * hidden * intermediate // GROUP_SIZE * NATIVE_METADATA_BANKS * FP16_BYTES
        shards.append(
            dict(
                rank=rank,
                experts=[first, stop],
                expert_count=stop - first,
                gdn_live_key_heads=gdn.live_key_heads,
                gdn_allocated_key_heads=gdn.key_heads,
                gdn_allocated_value_heads=gdn.value_heads,
                gdn_conv_channels=gdn.conv_dim,
                gdn_execution_value_heads=live.value_heads,
                gdn_execution_conv_channels=live.conv_dim,
                shared_channels=[shared_start, shared_stop],
                shared_channel_count=shared_stop - shared_start,
                shared_projection_bytes=3 * hidden * (shared_stop - shared_start) * FP16_BYTES,
                qsa_query_heads=qsa.num_query_heads,
                qsa_kv_start=qsa.kv_start,
                qsa_kv_heads=qsa.num_kv_heads,
                routed_backbone_codes_and_metadata_bytes=(stop - first) * policy.num_hidden_layers * (weights + meta),
                recurrent_bytes_per_gdn_state=gdn.value_heads * gdn.value_dim * gdn.key_dim * GDN_STATE_DTYPE_BYTES,
                recurrent_live_bytes_per_gdn_state=live.value_heads
                * live.value_dim
                * live.key_dim
                * GDN_STATE_DTYPE_BYTES,
            )
        )
    receipt = dict(
        tensor_parallel_size=TP_SIZE,
        pipeline_parallel_size=1,
        expert_parallel_size=TP_SIZE,
        mtp_enabled=False,
        requested_limits={
            "max_model_len": 262144,
            "max_num_seqs": 6,
            "max_num_batched_tokens": 2560,
            "cudagraph_capture_sizes": [1, 6],
        },
        requested_limits_validated=False,
        mm_encoder_tp_mode="data",
        images_enabled=True,
        projection_variant="baseline",
        ranks=shards,
        hardware_admission=False,
        total_installed_ram_is_not_context_capacity=True,
        excludes=[
            "dense/router/PLE/vision/embedding weights",
            "KV and convolution caches",
            "prefix checkpoints",
            "graphs/HCCL",
            "prefill scratch/load peaks/allocator reserve",
        ],
        required_gates=[
            "six-rank startup and all-reduce",
            "real-weight output/state parity",
            "image data-parallel encoder",
            "graphs/cache/CoW/cancellation",
            "memory and thermals",
        ],
        gdn_uniform_cache_reserve_retained=True,
        gdn_dummy_heads_executed=False,
        shared_expert_execution="tp_sharded_uneven",
        unsupported=["MTP", "PP greater than one", "external KV transfer"],
        vocabulary_padding_multiple=padding,
        padded_vocab_rows=(policy.vocab_size + padding - 1) // padding * padding,
        config_sha256=hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest(),
    )
    return result, receipt


def checkpoint_overlay(checkpoint, output):
    """Append-only config overlay; link canonical files without rewriting weights."""
    checkpoint, output = Path(checkpoint).resolve(strict=True), Path(output).resolve()
    if not checkpoint.is_dir() or output.is_relative_to(checkpoint):
        raise ValueError("checkpoint overlay must be separate from the source directory")
    config, receipt = make_profile(json.loads((checkpoint / "config.json").read_text()))
    if not (checkpoint / "model.safetensors.index.json").is_file():
        raise ValueError("canonical safetensors index is required; TP-specific prepacked exports are unsupported")
    output.mkdir(parents=True, exist_ok=False)
    assets = []
    for source in sorted(checkpoint.iterdir()):
        if source.name in ("config.json", "profile.json"):
            continue
        (output / source.name).symlink_to(source, target_is_directory=source.is_dir())
        assets.append(dict(path=source.name, target=str(source)))
    receipt["source_checkpoint"] = str(checkpoint)
    receipt["linked_assets"] = assets
    for name, value in (("config.json", config), ("profile.json", receipt)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.checkpoint is not None:
        if args.config.resolve() != (args.checkpoint / "config.json").resolve():
            parser.error("config must belong to the selected canonical checkpoint")
        checkpoint_overlay(args.checkpoint, args.output)
        return
    config, receipt = make_profile(json.loads(args.config.read_text()))
    args.output.mkdir(exist_ok=False, parents=True)
    for name, value in (("config.json", config), ("profile.json", receipt)):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
