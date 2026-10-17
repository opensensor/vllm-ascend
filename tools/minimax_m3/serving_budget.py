# SPDX-License-Identifier: Apache-2.0
"""Manifest-based, conservative TP2/PP3 memory planning; never accesses devices."""

import argparse
import json
import math
from pathlib import Path

from tools.minimax_m3.checkpoint import LAYER_NAME, read_manifest

GIB = 1024**3
TP_SIZE = 2
PP_SIZE = 3
BLOCK_SIZE = 128
DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "U8": 1}
SHARDED_PROJECTIONS = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "index_q_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)


def payload_bytes(record):
    return sum(math.prod(t["shape"]) * DTYPE_BYTES[t["dtype"]] for t in record["tensors"].values())


def placement(name, record):
    """Return owner and per-rank bytes. Replicate all vision as an upper bound."""
    size = payload_bytes(record)
    if not name.startswith("language_model."):
        if name.startswith(("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")):
            return "vision", size
        raise ValueError(f"Unknown weight placement: {name}")
    match = LAYER_NAME.search(name)
    if match:
        if record["expert"]:
            return int(match[1]), size // TP_SIZE
        # Index K is a single replicated index vector, unlike main-attention K.
        if name.endswith("index_k_proj.weight"):
            return int(match[1]), size
        if name.endswith(SHARDED_PROJECTIONS):
            if size % TP_SIZE:
                raise ValueError(f"Cannot shard {name}")
            return int(match[1]), size // TP_SIZE
        if name.endswith(("norm.weight", "gate.weight", "e_score_correction_bias")):
            return int(match[1]), size
        raise ValueError(f"Unknown decoder weight placement: {name}")
    if name.endswith("embed_tokens.weight"):
        return "first", size // TP_SIZE
    if name.endswith("lm_head.weight"):
        return "last", size // TP_SIZE
    if name.endswith("model.norm.weight"):
        return "last", size
    raise ValueError(f"Unknown language weight placement: {name}")


def make_budget(manifest, config, *, context=8192, sequences=1, device_gib=44, utilization=0.90, reserve_gib=3):
    text = config["text_config"]
    layers = text["num_hidden_layers"]
    if layers < PP_SIZE or context <= 0 or sequences <= 0 or device_gib <= 0 or reserve_gib < 0:
        raise ValueError("Invalid capacity arguments")
    if not 0 < utilization <= 1 or context > text["max_position_embeddings"]:
        raise ValueError("Invalid utilization or context length")
    if any(
        text[k] % TP_SIZE for k in ("num_attention_heads", "num_key_value_heads", "num_local_experts", "vocab_size")
    ):
        raise ValueError("TP2 geometry requires divisible heads, experts, and vocabulary")
    totals = {"vision": 0, "first": 0, "last": 0, **dict.fromkeys(range(layers), 0)}
    for name, record in manifest["weights"].items():
        owner, size = placement(name, record)
        totals[owner] += size
    sparse = text["sparse_attention_config"]
    main_kv = 2 * (text["num_key_value_heads"] // TP_SIZE) * text["head_dim"] * 2
    per_token = [main_kv + (sparse["sparse_index_dim"] * 2 if flag else 0) for flag in sparse["sparse_attention_freq"]]
    if len(per_token) != layers or sparse["sparse_block_size"] != BLOCK_SIZE:
        raise ValueError("Unsupported cache layout")
    tokens = math.ceil(context / BLOCK_SIZE) * BLOCK_SIZE * sequences
    reserve = round(reserve_gib * GIB)
    budget = int(device_gib * GIB * utilization)

    def partition(cuts):
        stages = []
        for stage, (start, stop) in enumerate(zip(cuts[:-1], cuts[1:])):
            weight = sum(totals[i] for i in range(start, stop)) + totals["vision"]
            weight += totals["first"] if stage == 0 else 0
            weight += totals["last"] if stage == PP_SIZE - 1 else 0
            kv = sum(per_token[start:stop]) * tokens
            required = weight + kv + reserve
            stages.append(
                dict(
                    stage=stage,
                    ranks=[2 * stage, 2 * stage + 1],
                    start_layer=start,
                    end_layer_exclusive=stop,
                    weight_bytes_per_rank=weight,
                    kv_bytes_per_rank=kv,
                    runtime_reserve_bytes=reserve,
                    planned_bytes_per_rank=required,
                    budget_bytes_per_rank=budget,
                    headroom_bytes_per_rank=budget - required,
                )
            )
        return dict(
            partition=[b - a for a, b in zip(cuts[:-1], cuts[1:])],
            stages=stages,
            peak_bytes_per_rank=max(s["planned_bytes_per_rank"] for s in stages),
            fits_planning_budget=all(s["headroom_bytes_per_rank"] >= 0 for s in stages),
        )

    candidates = [partition((0, a, b, layers)) for a in range(1, layers - 1) for b in range(a + 1, layers)]
    best = min(candidates, key=lambda c: c["peak_bytes_per_rank"])
    return dict(
        hardware_validated=False,
        payload_hashes_verified=False,
        model_tensor_count=manifest["tensor_count"],
        unique_payload_bytes=sum(payload_bytes(r) for r in manifest["weights"].values()),
        assumptions=dict(
            tp=TP_SIZE,
            pp=PP_SIZE,
            ep=TP_SIZE,
            context=context,
            sequences=sequences,
            device_gib=device_gib,
            utilization=utilization,
            reserve_gib=reserve_gib,
            vision="Fully replicated per rank (conservative upper bound)",
            reserve="Assumed allowance for workspaces, communication, alignment, and allocator",
        ),
        cache_bytes_per_token_all_ranks=TP_SIZE * sum(per_token),
        best=best,
        original=partition((0, 21, 41, layers)) if layers == 60 else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument(
        "--device-gib", type=float, default=44, help="Planning assumption, NOT detected device capacity"
    )
    parser.add_argument("--utilization", type=float, default=0.90)
    parser.add_argument("--reserve-gib", type=float, default=3)
    args = parser.parse_args()
    report = make_budget(
        read_manifest(args.model_path),
        json.loads((args.model_path / "config.json").read_text()),
        context=args.context,
        sequences=args.sequences,
        device_gib=args.device_gib,
        utilization=args.utilization,
        reserve_gib=args.reserve_gib,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
