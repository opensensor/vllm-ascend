#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the Qwen4Exp (Qwen3.8-Flash-Next) 300i/310P checkpoint manifest (plan T0.1).

Reads the exported ModelSlim W8A8 checkpoint (config.json, generation_config.json,
conversion_report.json, quant_model_weights.safetensors.index.json, and each
shard's safetensors header) and emits a manifest JSON that the load-time tasks
consume: T3.1 (W8A8 weight mapping), T3.2 (shard placement), T4.2 (n-gram hash).

The manifest records: architecture + config-derived geometry, the exact tensor
naming scheme + per-component counts, observed per-component dtypes (read from
safetensors headers — authoritative, not the config's source dtype), the shard
file list with sizes, and (with --sha256) per-shard SHA256 for pinning.

The held-out perplexity / G2 accuracy threshold is NOT produced here: the export
reports `ascend_inference_validated: false` and computing it needs a working
runtime (hardware wave). It is recorded as `pending` with the reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from collections import Counter
from pathlib import Path

MANIFEST_SCHEMA_VERSION = 1

# Config keys we lift verbatim into the manifest geometry block.
_GEOMETRY_KEYS = (
    "model_type",
    "num_hidden_layers",
    "hidden_size",
    "vocab_size",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "ple_layer_ids",
    "ple_embed_dim",
    "ple_conv_kernel_size",
    "ngram_size",
    "ngram_vocab_size_base",
    "make_ngram_vocab_size_divisible_by",
    "heads_per_ngram",
    "split_ngram_parts",
    "head_dim",
    "num_attention_heads",
    "num_key_value_heads",
    "linear_num_value_heads",
    "linear_num_key_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "indexer_budget",
    "indexer_compress_ratio",
    "indexer_head_dim",
    "indexer_kv_heads",
    "indexer_n_heads",
    "partial_rotary_factor",
    "output_gate_type",
    "hc_count",
    "hc_lowrank",
    "max_position_embeddings",
    "rope_parameters",
    "rms_norm_eps",
    "mamba_ssm_dtype",
    "tie_word_embeddings",
    "dtype",
)

# Component classification by tensor-name regex (first match wins). Order matters.
_COMPONENT_PATTERNS = (
    ("expert_weight", re.compile(r"\.mlp\.experts\.\d+\.\w+\.weight$")),
    ("expert_weight_scale", re.compile(r"\.mlp\.experts\.\d+\.\w+\.weight_scale$")),
    ("expert_weight_offset", re.compile(r"\.mlp\.experts\.\d+\.\w+\.weight_offset$")),
    ("shared_expert", re.compile(r"\.mlp\.shared_expert")),
    ("router", re.compile(r"\.mlp\.gate\.weight$|\.mlp\.router")),
    ("ple_ngram", re.compile(r"\.ple\..*ngram_embedding")),
    ("ple_other", re.compile(r"\.ple\.")),
    ("gdn", re.compile(r"\.linear_attn\.")),
    ("qsa_indexer", re.compile(r"\.self_attn\.indexer\.")),
    ("qsa_attn", re.compile(r"\.self_attn\.")),
    ("embed_tokens", re.compile(r"embed_tokens\.weight$")),
    ("lm_head", re.compile(r"lm_head\.weight$")),
    ("norm", re.compile(r"norm\.weight$|norm\.bias$")),
    ("mtp", re.compile(r"\.mtp\.|\bmtp\b")),
)


def classify(name: str) -> str:
    for label, pat in _COMPONENT_PATTERNS:
        if pat.search(name):
            return label
    return "other"


def read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(header_len))


def sha256_file(path: Path, chunk: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(ckpt_dir: Path, *, with_sha256: bool) -> dict:
    config = json.loads((ckpt_dir / "config.json").read_text())
    text_config = config.get("text_config", config)
    gen = json.loads((ckpt_dir / "generation_config.json").read_text())
    report = json.loads((ckpt_dir / "conversion_report.json").read_text())
    index = json.loads((ckpt_dir / "quant_model_weights.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]

    # Component counts + which shards a component lands in.
    comp_counts: Counter[str] = Counter()
    for name in weight_map:
        comp_counts[classify(name)] += 1

    # Observed dtype per component (sample one tensor per component from headers).
    shard_headers: dict[str, dict] = {}
    comp_sample: dict[str, dict] = {}
    for name, shard in weight_map.items():
        label = classify(name)
        if label in comp_sample:
            continue
        if shard not in shard_headers:
            shard_headers[shard] = read_safetensors_header(ckpt_dir / shard)
        meta = shard_headers[shard].get(name, {})
        comp_sample[label] = {"tensor": name, "dtype": meta.get("dtype"), "shape": meta.get("shape")}

    # Shard file list with sizes (+ optional sha256).
    shard_files = sorted({s for s in weight_map.values()})
    shards = []
    for shard in shard_files:
        p = ckpt_dir / shard
        entry = {"file": shard, "bytes": p.stat().st_size}
        if with_sha256:
            entry["sha256"] = sha256_file(p)
        shards.append(entry)

    geometry = {k: text_config[k] for k in _GEOMETRY_KEYS if k in text_config}

    # PLE placement is authoritative from tensor names (config ple_layer_ids may
    # differ in indexing). Record both.
    ple_layers_in_tensors = sorted(
        {int(m.group(1)) for name in weight_map if (m := re.search(r"layers\.(\d+)\.ple\.", name))}
    )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "checkpoint_dir": str(ckpt_dir),
        "architectures": config.get("architectures"),
        "conversion_report": report,
        "export_dtype_observed": {label: s["dtype"] for label, s in sorted(comp_sample.items())},
        "geometry": geometry,
        "ple_layer_ids_config": text_config.get("ple_layer_ids"),
        "ple_layers_in_tensor_names": ple_layers_in_tensors,
        "eos_token_id": gen.get("eos_token_id"),
        "bos_token_id": gen.get("bos_token_id"),
        "pad_token_id": gen.get("pad_token_id"),
        "tensor_total": len(weight_map),
        "component_counts": dict(sorted(comp_counts.items())),
        "component_samples": comp_sample,
        "naming_scheme": {
            "expert": "layers.{L}.mlp.experts.{E}.{gate_proj|up_proj|down_proj}.{weight|weight_scale|weight_offset}",
            "shared_expert": "layers.{L}.mlp.shared_expert.{gate_proj|up_proj|down_proj}.weight (+ shared_expert_gate)",
            "ple_ngram": "model.language_model.layers.{L}.ple.ple_embedding.ngram_embedding.shard_{S}.weight",
            "gdn": "model.language_model.layers.{L}.linear_attn.{in_proj_a|in_proj_z|out_proj|norm|...}.weight",
            "qsa": "layers.{L}.self_attn.{q_proj|k_proj|v_proj|o_proj|q_norm|k_norm|indexer.*}.weight",
        },
        "num_shards": len(shard_files),
        "shards": shards,
        "sha256_present": with_sha256,
        "g2_accuracy_threshold": {
            "status": "pending",
            "reason": "export reports ascend_inference_validated=false; held-out "
            "perplexity/quality delta requires a working runtime (device wave). "
            "Freeze the G2 threshold before inspecting Ascend output (PRD §8.1).",
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the Qwen4Exp checkpoint manifest (T0.1)")
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--out", required=True, help="manifest JSON output path")
    parser.add_argument("--sha256", action="store_true", help="compute per-shard SHA256 (slow: reads every byte)")
    args = parser.parse_args(argv)

    manifest = build_manifest(Path(args.checkpoint_dir), with_sha256=args.sha256)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out} ({len(manifest['shards'])} shards, sha256={'yes' if args.sha256 else 'no'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
