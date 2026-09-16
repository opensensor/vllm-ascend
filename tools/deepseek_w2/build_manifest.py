#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the DeepSeek V4.1 552B W2-on-310P checkpoint manifest (plan E0.1).

Reads the FP8 source checkpoint (``config.json`` with nested ``text_config`` +
``vision_config``, ``model.safetensors.index.json``, and every shard's
safetensors header) and emits a manifest JSON that the downstream tasks consume:
the W2 expert quantiser, the load-time weight mapper, and the memory-accounting
model.

The manifest records, all from authoritative sources:

* architecture + config-derived geometry (every geometry field the plan lists);
* the quantization block (fp8 / fp4 experts, ``weight_block_size``,
  ``scale_fmt``);
* per-family tensor counts (overall + split by main / mtp / vision block);
* observed per-family dtypes read from the safetensors headers -- authoritative,
  not the config's nominal dtype (expect int8-packed FP4 routed experts, F8_E4M3
  dense/MLA/engram weights, F8_E8M0 ue8m0 scales, BF16 norms/indexer/embeddings);
* the Engram embedding-table shapes, the indexer / MLA family layout, EOS /
  special tokens, and the MTP (num_nextn_predict_layers) layers;
* the **intended target precision per family** for the W2 deployment (W2 routed
  experts, ~W4 Engram, FP16 MLA / indexer / dense / LM-head).

The held-out G2 accuracy threshold is NOT produced here: it needs a working
runtime (device wave) and is recorded as ``pending`` with the reason. Reuses the
safetensors-header + sha256 helpers from ``tools/qwen38_1m/build_manifest.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path

MANIFEST_SCHEMA_VERSION = 1

_INDEX_FILE = "model.safetensors.index.json"


# --- reuse the header/sha256 helpers from the Qwen manifest builder ----------
def _load_qwen_helpers():
    qwen_path = Path(__file__).resolve().parents[1] / "qwen38_1m" / "build_manifest.py"
    try:
        spec = importlib.util.spec_from_file_location("qwen38_1m_build_manifest", qwen_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.read_safetensors_header, module.sha256_file
    except (FileNotFoundError, AttributeError):
        return None, None


_qwen_read_header, _qwen_sha256 = _load_qwen_helpers()


def read_safetensors_header(path: Path) -> dict:
    if _qwen_read_header is not None:
        return _qwen_read_header(path)
    with path.open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(header_len))


def sha256_file(path: Path) -> str:
    if _qwen_sha256 is not None:
        return _qwen_sha256(path)
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- geometry ----------------------------------------------------------------
# text_config keys lifted verbatim into the manifest geometry block.
_GEOMETRY_KEYS = (
    "model_type",
    "vocab_size",
    "hidden_size",
    "moe_intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "qk_rope_head_dim",
    "q_lora_rank",
    "o_lora_rank",
    "o_groups",
    "hidden_act",
    "max_position_embeddings",
    "rope_theta",
    "rope_scaling",
    "n_routed_experts",
    "n_shared_experts",
    "num_experts_per_tok",
    "scoring_func",
    "topk_method",
    "norm_topk_prob",
    "routed_scaling_factor",
    "sliding_window",
    "rms_norm_eps",
    "tie_word_embeddings",
    "compress_ratios",
    "compress_rope_theta",
    "kv_source_layer_ids",
    "index_source_layer_ids",
    "index_n_heads",
    "index_head_dim",
    "index_topk",
    "candidate_source_layer_id",
    "engram_layer_ids",
    "engram_num_embeddings",
    "engram_max_ngram_size",
    "engram_vocab_size",
    "engram_n_heads",
    "engram_head_dim",
    "engram_compressed_vocab_size",
    "num_nextn_predict_layers",
    "dspark_block_size",
    "dspark_target_layer_ids",
    "dspark_markov_rank",
    "dspark_n_routed_experts",
    "dspark_num_experts_per_tok",
)


# --- family classification ---------------------------------------------------
# Component family by tensor-name regex, first match wins -- order matters.
# Family names are prefix-agnostic (main / mtp share semantics); the block split
# (main / mtp / vision) is tracked separately.
_FAMILY_PATTERNS = (
    ("routed_expert_scale", re.compile(r"\.ffn\.experts\.\d+\.w\d+\.scale$")),
    ("routed_expert_weight", re.compile(r"\.ffn\.experts\.\d+\.w\d+\.weight$")),
    ("shared_expert_scale", re.compile(r"\.ffn\.shared_experts\.w\d+\.scale$")),
    ("shared_expert_weight", re.compile(r"\.ffn\.shared_experts\.w\d+\.weight$")),
    ("router_gate", re.compile(r"\.ffn\.gate\.")),
    ("engram_embed", re.compile(r"\.engram\.embed\.")),
    ("engram", re.compile(r"\.engram\.")),
    ("mla_indexer", re.compile(r"\.attn\.indexer\.")),
    ("mla_compressor", re.compile(r"\.attn\.compressor\.")),
    ("mla_attn", re.compile(r"\.attn\.")),
    ("hc_gate", re.compile(r"\.hc_")),
    ("mtp_markov_head", re.compile(r"\.markov_head\.")),
    ("mtp_confidence_head", re.compile(r"\.confidence_head\.")),
    ("mtp_main_proj", re.compile(r"\.main_proj\.")),
    ("layer_norm", re.compile(r"(attn_norm|ffn_norm|main_norm)\.weight$")),
    ("vision_aligner", re.compile(r"^aligner\.")),
    ("vision", re.compile(r"^vision\.")),
    ("vision_image_token", re.compile(r"^image_(start|end|newline)$")),
    ("embed_tokens", re.compile(r"^embed\.weight$")),
    ("lm_head", re.compile(r"^head\.weight$")),
    ("final_norm", re.compile(r"^norm\.weight$|^mtp\.\d+\.norm\.weight$")),
)

# Intended target precision per family for the W2 deployment. Source precisions
# are read from headers; these are the *targets* the quantiser must hit.
_TARGET_PRECISION = {
    "routed_expert_weight": "W2 (2-bit routed experts; source FP4 int8-packed)",
    "routed_expert_scale": "keep (per-32x32-block ue8m0 scale)",
    "shared_expert_weight": "FP16 (dense shared expert)",
    "shared_expert_scale": "FP16 (dequantised with weight)",
    "router_gate": "FP16 (router/gate + bias)",
    "engram_embed": "W4 (~4-bit Engram embedding table)",
    "engram": "W4 (~4-bit Engram q/k/wkv)",
    "mla_indexer": "FP16 (sparse-attention indexer)",
    "mla_compressor": "FP16 (KV compressor)",
    "mla_attn": "FP16 (MLA q/kv/o projections)",
    "hc_gate": "FP16 (hierarchical gate params)",
    "mtp_markov_head": "FP16 (MTP markov head)",
    "mtp_confidence_head": "FP16 (MTP confidence head)",
    "mtp_main_proj": "FP16 (MTP main projection)",
    "layer_norm": "FP16 (block norms)",
    "final_norm": "FP16 (final / mtp norms)",
    "embed_tokens": "FP16 (token embedding)",
    "lm_head": "FP16 (LM head)",
    "vision": "n/a (text-only deployment; vision tower excluded)",
    "vision_aligner": "n/a (text-only deployment; vision tower excluded)",
    "vision_image_token": "n/a (text-only deployment; vision tower excluded)",
    "other": "FP16 (unclassified; review)",
}


def classify(name: str) -> str:
    for label, pattern in _FAMILY_PATTERNS:
        if pattern.search(name):
            return label
    return "other"


def block_of(name: str) -> str:
    if name.startswith("vision.") or name.startswith("aligner.") or name.startswith("image_"):
        return "vision"
    if name.startswith("mtp."):
        return "mtp"
    return "main"


def build_manifest(ckpt_dir: Path, *, with_sha256: bool) -> dict:
    ckpt_dir = Path(ckpt_dir)
    config = json.loads((ckpt_dir / "config.json").read_text())
    text_config = config.get("text_config", config)
    quant_config = config.get("quantization_config", {})
    index = json.loads((ckpt_dir / _INDEX_FILE).read_text())
    weight_map: dict[str, str] = index["weight_map"]

    # Read every shard header once (headers are cheap; the 476 GB payload is
    # never touched). Accumulate authoritative per-family dtype counts + a
    # representative shape sample, plus per-block family counts.
    family_counts: Counter[str] = Counter()
    family_counts_by_block: dict[str, Counter[str]] = defaultdict(Counter)
    block_counts: Counter[str] = Counter()
    family_dtypes: dict[str, Counter[str]] = defaultdict(Counter)
    family_sample: dict[str, dict] = {}
    engram_tables: dict[str, dict] = {}

    shard_files = sorted(set(weight_map.values()))
    shard_headers: dict[str, dict] = {}
    for shard in shard_files:
        shard_headers[shard] = read_safetensors_header(ckpt_dir / shard)

    for name, shard in weight_map.items():
        family = classify(name)
        block = block_of(name)
        meta = shard_headers[shard].get(name, {})
        dtype = meta.get("dtype")
        shape = meta.get("shape")

        family_counts[family] += 1
        family_counts_by_block[block][family] += 1
        block_counts[block] += 1
        if dtype is not None:
            family_dtypes[family][dtype] += 1
        if family not in family_sample:
            family_sample[family] = {"tensor": name, "dtype": dtype, "shape": shape}

    # Engram embedding-table geometry (per engram layer), read from headers.
    engram_layers = sorted(
        {int(m.group(1)) for name in weight_map if (m := re.search(r"layers\.(\d+)\.engram\.", name))}
    )
    for layer_id in engram_layers:
        table: dict[str, dict] = {}
        for sub in ("embed.weight", "embed.scale", "q_weight", "k_weight", "wkv.weight", "wkv.scale"):
            tname = f"layers.{layer_id}.engram.{sub}"
            shard = weight_map.get(tname)
            if shard is None:
                continue
            meta = shard_headers[shard].get(tname, {})
            key = sub.replace(".", "_")
            table[key] = {"tensor": tname, "dtype": meta.get("dtype"), "shape": meta.get("shape")}
        engram_tables[str(layer_id)] = table

    # Shard file list with sizes (+ optional sha256).
    shards = []
    for shard in shard_files:
        path = ckpt_dir / shard
        entry = {"file": shard, "bytes": path.stat().st_size}
        if with_sha256:
            entry["sha256"] = sha256_file(path)
        shards.append(entry)

    geometry = {k: text_config[k] for k in _GEOMETRY_KEYS if k in text_config}

    vision_families = ("vision", "vision_aligner", "vision_image_token")
    vision_count = sum(family_counts.get(f, 0) for f in vision_families)

    target_precision = {fam: _TARGET_PRECISION.get(fam, _TARGET_PRECISION["other"]) for fam in family_counts}

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_dir": str(ckpt_dir),
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "dtype": config.get("dtype"),
        "geometry": geometry,
        "quantization": {
            "quant_method": quant_config.get("quant_method"),
            "activation_scheme": quant_config.get("activation_scheme"),
            "weight_block_size": quant_config.get("weight_block_size"),
            "scale_fmt": quant_config.get("scale_fmt"),
            "expert_dtype": quant_config.get("expert_dtype"),
        },
        "eos_token_id": config.get("eos_token_id"),
        "bos_token_id": config.get("bos_token_id"),
        "pad_token_id": config.get("pad_token_id"),
        "image_token_id": config.get("image_token_id"),
        "mtp": {
            "num_nextn_predict_layers": text_config.get("num_nextn_predict_layers"),
            "dspark_target_layer_ids": text_config.get("dspark_target_layer_ids"),
            "dspark_n_routed_experts": text_config.get("dspark_n_routed_experts"),
            "tensor_count": block_counts.get("mtp", 0),
        },
        "vision": {
            "present": vision_count > 0,
            "included_in_deployment": False,
            "reason": "text-only W2-on-310P deployment; vision tower + aligner recorded but not loaded",
            "tensor_count": vision_count,
            "config": config.get("vision_config"),
        },
        "tensor_total": len(weight_map),
        "block_counts": dict(sorted(block_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "family_counts_by_block": {b: dict(sorted(c.items())) for b, c in sorted(family_counts_by_block.items())},
        "family_dtypes": {fam: dict(sorted(c.items())) for fam, c in sorted(family_dtypes.items())},
        "family_samples": dict(sorted(family_sample.items())),
        "target_precision": dict(sorted(target_precision.items())),
        "engram_tables": engram_tables,
        "naming_scheme": {
            "routed_expert": "layers.{L}.ffn.experts.{E}.{w1|w2|w3}.{weight|scale}",
            "shared_expert": "layers.{L}.ffn.shared_experts.{w1|w2|w3}.{weight|scale}",
            "router": "layers.{L}.ffn.gate.{weight|bias|bias_vl}",
            "mla_attn": "layers.{L}.attn.{wq_a|wq_b|q_norm|wkv|kv_norm|wo_a|wo_b|attn_sink}.{weight|scale}",
            "mla_indexer": "layers.{L}.attn.indexer.{wq_b|wk|k_norm|weights_proj}.{weight|scale}",
            "mla_compressor": "layers.{L}.attn.compressor.{wkv|wgate|norm}.weight",
            "engram": "layers.{L}.engram.{embed.{weight|scale}|q_weight|k_weight|wkv.{weight|scale}}",
            "mtp": "mtp.{N}.{attn|ffn|hc_*|main_proj|main_norm|norm|markov_head|confidence_head}...",
            "embed_tokens": "embed.weight",
            "lm_head": "head.weight",
            "final_norm": "norm.weight",
        },
        "num_shards": len(shard_files),
        "shards": shards,
        "sha256_present": with_sha256,
        "g2_accuracy_threshold": {
            "status": "pending",
            "reason": "held-out perplexity / quality delta for the W2 experts "
            "requires a working runtime (device wave). Freeze the G2 threshold "
            "before inspecting Ascend output.",
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the DeepSeek V4.1 W2 checkpoint manifest (E0.1)")
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--out", required=True, help="manifest JSON output path")
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="compute per-shard SHA256 (slow: reads every byte of the 476 GB payload)",
    )
    args = parser.parse_args(argv)

    manifest = build_manifest(Path(args.checkpoint_dir), with_sha256=args.sha256)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(
        f"wrote {out} ({manifest['tensor_total']} tensors, "
        f"{manifest['num_shards']} shards, {len(manifest['family_counts'])} families, "
        f"sha256={'yes' if args.sha256 else 'no'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
