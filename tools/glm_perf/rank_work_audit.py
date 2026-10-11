# SPDX-License-Identifier: Apache-2.0
"""Offline source-backed GLM execution ownership and candidate memory counts."""

import argparse
import json
from pathlib import Path

SELECTION_TILE_COLUMNS = 128
INT32_BYTES = 4


def audit(config, *, tp_size, pp_size, tokens):
    text = config.get("text_config", config)
    layers = text["layer_types"]
    if (
        any(type(value) is not int or value < 1 for value in (tp_size, pp_size, tokens))
        or len(layers) != text["num_hidden_layers"]
        or len(layers) % pp_size
    ):
        raise ValueError("audit requires positive dimensions and evenly partitioned explicit layer types")
    hidden, rank = text["hidden_size"], text["q_lora_rank"]
    heads, dim = text["index_n_heads"], text["index_head_dim"]
    pool, topk = text["index_kpool"], text["index_topk"]
    width = ((topk + pool - 2 + SELECTION_TILE_COLUMNS) // SELECTION_TILE_COLUMNS) * SELECTION_TILE_COLUMNS
    weights = rank * heads * dim + hidden * (dim + heads) + hidden * dim
    per_stage = len(layers) // pp_size
    stages = []
    for stage in range(pp_size):
        sparse = [
            layer
            for layer in range(stage * per_stage, (stage + 1) * per_stage)
            if layers[layer] == "deepseek_sparse_attention"
        ]
        stages.append(
            dict(
                stage=stage,
                sparse_layers=sparse,
                replicated_indexer_executions=len(sparse) * tp_size,
                owner_indexer_executions=len(sparse),
                avoided_projection_ops=len(sparse) * (tp_size - 1) * 2 * weights * tokens,
                # One slab per rank reused by all indexers, not one per layer.
                exchange_bytes_per_rank=tp_size * tokens * width * INT32_BYTES if sparse else 0,
                owner_result_bytes_per_sparse_layer=tokens * width * INT32_BYTES,
            )
        )
    return dict(
        evidence="static source and model geometry; no device timings or memory measurement",
        tp_size=tp_size,
        pp_size=pp_size,
        tokens=tokens,
        selection_width=width,
        indexer_projection_weights=weights,
        projections="wq_b, wk_weights_proj, index_kpool_compress_gate",
        stages=stages,
        avoided_projection_ops=sum(stage["avoided_projection_ops"] for stage in stages),
        score_ops_excluded=True,
        peer_weights_and_caches_retained=True,
        numerical_contract_changed=False,
        latency_model="benefit requires old replicated indexer time > owner indexer time + exchange + peer copy",
        sharding_admission="counts alone do not establish support for this TP/PP geometry",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--pp-size", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=640)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(json.loads(args.config.read_text()), tp_size=args.tp_size, pp_size=args.pp_size, tokens=args.tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
