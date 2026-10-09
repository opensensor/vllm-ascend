# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recompute historical profiler attribution and logical byte estimates offline."""

import csv
import hashlib
import io
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
REPO = AUDIT.parents[1]
TRACE_ROOT = REPO / "artifacts/qwen38-prefill-batch256-20261005/named-prefill-trace"


def host_receipt(raw, source):
    operations = defaultdict(lambda: {"count": 0, "host_self_ms": 0.0, "device_self_ms": 0.0})
    for row in csv.DictReader(io.StringIO(raw.decode())):
        name = row["Name"]
        if not any(
            part in name.lower() for part in ("memcpy", "synchron", "nonzero", "local_scalar", "is_nonzero", "event")
        ):
            continue
        values = [float(row[column]) for column in ("Host Self Duration(us)", "Device Self Duration(us)")]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Invalid profiler duration")
        entry = operations[name]
        entry["count"] += 1
        entry["host_self_ms"] += values[0] / 1000
        entry["device_self_ms"] += values[1] / 1000
    return {"source": source, "sha256": hashlib.sha256(raw).hexdigest(), "operations": dict(sorted(operations.items()))}


def main():
    summary_path = TRACE_ROOT / "summary.json"
    summary = json.loads(summary_path.read_text())
    kernels = []
    for trace in summary["traces"]:
        kernels.append(
            {
                "source": trace["file"],
                "task_span_ms": trace["task_span_ms"],
                "summed_task_ms": trace["summed_task_ms"],
                "categories": trace["categories"],
                "top_operations": trace["top_operations"][:15],
                "projection_shapes": trace["projection_shapes"],
            }
        )
    with tarfile.open(TRACE_ROOT / "selected-trace-csvs.tar.gz", "r:gz") as archive:
        member = archive.extractfile("rank1/operator_details.csv")
        assert member is not None
        prefill = host_receipt(member.read(), "selected-trace-csvs.tar.gz:rank1/operator_details.csv")
    decode = []
    for rank in range(4):
        path = REPO / f"artifacts/qwen38-decode-next-20261005/named-decode/rank{rank}/operator_details.csv"
        decode.append(host_receipt(path.read_bytes(), str(path.relative_to(REPO))))
    report = {
        "date": "2026-10-05",
        "not_incident_capture": True,
        "metric_warning": (
            "Summed tasks overlap; percentages are attribution, not latency, speedup or heat. "
            "Host records omit copy byte sizes. These traces predate the bounded scheduler "
            "and current native HC selection."
        ),
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "kernels": kernels,
        "prefill_host_rank1": prefill,
        "decode_host": decode,
    }
    (AUDIT / "historical-profiler-receipts.json").write_text(json.dumps(report, indent=2) + "\n")

    config = json.loads((AUDIT / "model-config.json").read_text())["text_config"]
    tokens, tile, tp, element_bytes = 2560, 64, 4, 2
    query_heads = config["num_attention_heads"] // tp
    heads_per_kv = config["num_attention_heads"] // config["num_key_value_heads"]
    local_kv_heads = max(1, query_heads // heads_per_kv)
    groups = config["indexer_budget"] // config["indexer_compress_ratio"]
    selected = groups * config["indexer_compress_ratio"] + config["indexer_compress_ratio"]
    padded = math.ceil(selected / 16) * 16
    qsa_layers = config["layer_types"].count("full_attention")
    kv_per_query = 2 * local_kv_heads * config["head_dim"] * padded * element_bytes
    estimates = {
        "not_measured_ddr_or_pcie_bytes": True,
        "conditions": (
            "Fully sparse pure prefill, one request, full 2048-token selection budget, 64-query tile, TP4. "
            "Partial tiles and shorter selected widths differ. Cache reuse can reduce physical reads; "
            "these numbers count logical scratch payload written, excluding scores and indexer work."
        ),
        "tokens_per_chunk": tokens,
        "query_tile": tile,
        "local_query_heads": query_heads,
        "local_kv_heads": local_kv_heads,
        "qsa_layers": qsa_layers,
        "gdn_layers": config["layer_types"].count("linear_attention"),
        "padded_selected_tokens": padded,
        "selected_kv_bytes_per_query": kv_per_query,
        "selected_kv_scratch_bytes_per_tile": kv_per_query * tile,
        "selected_kv_payload_bytes_per_layer_chunk": kv_per_query * tokens,
        "selected_kv_payload_bytes_per_target_chunk_per_rank": kv_per_query * tokens * qsa_layers,
        "gather_calls_per_target_chunk_per_rank": 2 * math.ceil(tokens / tile) * qsa_layers,
        "ple_h2d_bytes_per_chunk_per_rank": tokens * config["ple_embed_dim"] * element_bytes,
        "mtp_counts_bytes_per_draft_forward_per_rank": config["num_experts"] // tp * 8,
        "target_decoder_all_reduce_call_sites": config["num_hidden_layers"] * 2,
        "grouped_routes": tokens * config["num_experts_per_tok"],
        "route_count_comparison_elements_per_moe_chunk": tokens
        * config["num_experts_per_tok"]
        * (config["num_experts"] // tp),
        "gate_up_route_output_bytes_per_moe_chunk": tokens
        * config["num_experts_per_tok"]
        * 2
        * config["moe_intermediate_size"]
        * element_bytes,
        "down_route_output_bytes_per_moe_chunk": tokens
        * config["num_experts_per_tok"]
        * config["hidden_size"]
        * element_bytes,
    }
    (AUDIT / "logical-byte-estimates.json").write_text(json.dumps(estimates, indent=2) + "\n")
    print(
        json.dumps(
            {
                "historical_ranks": len(kernels),
                "qsa_scratch_gib_per_layer_chunk": kv_per_query * tokens / 2**30,
                "qsa_scratch_gib_per_target_chunk_per_rank": kv_per_query * tokens * qsa_layers / 2**30,
                "gather_calls_per_chunk": estimates["gather_calls_per_target_chunk_per_rank"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
