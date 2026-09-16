#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the GLM-5.3-Flash (glm5_next) W2-on-310P checkpoint manifest.

Reads the FP8 source checkpoint (``config.json`` with a nested ``text_config`` +
``vision_config``, ``model.safetensors.index.json``, and every shard's
safetensors header) and emits a manifest JSON for the downstream W2 converter.

GLM-5.3-Flash is *much* simpler than DeepSeek V4.1 for the W2 deployment: there
is **no Engram** and **no sparse-attention indexer table to quantise** -- only
the routed MoE experts are quantised to 2-bit. Concretely:

* **Routed experts** ``model.language_model.layers.{L}.mlp.experts.{E}.{gate_proj|
  up_proj|down_proj}.weight`` are ``F8_E4M3`` ``[out, in]`` with a companion
  ``.weight_scale_inv`` ``F32`` ``[out/128, in/128]`` -- a **plain float32
  per-[128,128]-block scale** (``weight_block_size = [128, 128]``, NOT ue8m0).
  These are the only tensors routed to **W2**.
* Every other quantised weight (dense MLP of the first ``first_k_dense_replace``
  layers, the shared expert, and the sparse-attention MLA projections
  ``kv_a_proj_with_mqa`` / ``o_proj`` / ``q_a_proj`` / ``q_b_proj``) is also
  ``F8_E4M3`` + ``F32`` block scale, but deploys at **FP16** (dequantised, not
  re-quantised).
* Everything else -- linear-attention projections, norms, router gate, hyper-
  connection params, embeddings, LM head, MTP ``eh_proj`` / ``enorm`` / ``hnorm``
  / ``shared_head`` -- is ``BF16`` or ``F32`` and deploys at **FP16** (cast).
* ``model.visual.*`` (the vision tower + merger + patch embed) is EXCLUDED from
  the text-only deployment.

The manifest records, all from authoritative sources: architecture + config
geometry; the quantization block (``weight_block_size``, ``fmt``); per-family
tensor counts (overall + split by main / mtp / vision block); the per-family
dtypes observed in the safetensors headers (authoritative, not the config's
nominal dtype); a representative shape per family; and the intended target
precision per family for the W2 deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path

MANIFEST_SCHEMA_VERSION = 1

_INDEX_FILE = "model.safetensors.index.json"


def read_safetensors_header(path: Path) -> dict:
    """Read only the JSON header of a safetensors shard (payload untouched)."""
    with path.open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))
    header.pop("__metadata__", None)
    return header


def sha256_file(path: Path) -> str:
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
    "intermediate_size",
    "moe_intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_act",
    "max_position_embeddings",
    "rope_theta",
    "rope_scaling",
    "n_routed_experts",
    "n_shared_experts",
    "num_experts_per_tok",
    "first_k_dense_replace",
    "n_group",
    "topk_group",
    "scoring_func",
    "norm_topk_prob",
    "routed_scaling_factor",
    "rms_norm_eps",
    "tie_word_embeddings",
    "kv_lora_rank",
    "q_lora_rank",
    "index_topk",
    "index_n_heads",
    "index_head_dim",
    "num_nextn_predict_layers",
    "layer_types",
    "mlp_layer_types",
)


# --- family classification ---------------------------------------------------
# Component family by tensor-name regex, first match wins -- order matters.
# The ``.weight_scale_inv`` companion of a quantised weight gets its own
# ``*_scale`` family so it is counted but never confused with a real weight.
_FAMILY_PATTERNS = (
    ("vision", re.compile(r"(^model\.visual\.|\.visual\.)")),
    ("routed_expert_scale", re.compile(r"\.mlp\.experts\.\d+\.\w+_proj\.weight_scale_inv$")),
    ("routed_expert_weight", re.compile(r"\.mlp\.experts\.\d+\.\w+_proj\.weight$")),
    ("shared_expert_scale", re.compile(r"\.mlp\.shared_experts\.\w+_proj\.weight_scale_inv$")),
    ("shared_expert_weight", re.compile(r"\.mlp\.shared_experts\.\w+_proj\.weight$")),
    ("router_gate", re.compile(r"\.mlp\.gate\.")),
    ("dense_mlp_scale", re.compile(r"\.mlp\.\w+_proj\.weight_scale_inv$")),
    ("dense_mlp_weight", re.compile(r"\.mlp\.\w+_proj\.weight$")),
    ("mla_indexer", re.compile(r"\.self_attn\.indexer\.")),
    ("attn_scale", re.compile(r"\.self_attn\..*\.weight_scale_inv$")),
    ("self_attn", re.compile(r"\.self_attn\.")),
    ("hyper_connection", re.compile(r"\.hc_")),
    ("mtp_extra", re.compile(r"\.(eh_proj|enorm|hnorm)\.weight$|\.shared_head\.")),
    ("layer_norm", re.compile(r"\.(input_layernorm|post_attention_layernorm)\.weight$")),
    ("embed_tokens", re.compile(r"\.embed_tokens\.weight$")),
    ("lm_head", re.compile(r"^lm_head\.weight$")),
    ("final_norm", re.compile(r"language_model\.norm\.weight$")),
)

# Intended target precision per family for the W2 deployment. Source precisions
# are read from headers; these are the *targets* the quantiser must hit.
_TARGET_PRECISION = {
    "routed_expert_weight": "W2 (2-bit routed experts; source F8_E4M3 + F32 [128,128] block scale)",
    "routed_expert_scale": "consumed (plain F32 per-[128,128]-block scale, fed to dequant)",
    "shared_expert_weight": "FP16 (dense shared expert; dequant F8_E4M3 + F32 block scale)",
    "shared_expert_scale": "consumed (F32 block scale, fed to dequant)",
    "dense_mlp_weight": "FP16 (first_k_dense_replace MLP; dequant F8_E4M3 + F32 block scale)",
    "dense_mlp_scale": "consumed (F32 block scale, fed to dequant)",
    "router_gate": "FP16 (router gate + e_score_correction_bias)",
    "mla_indexer": "FP16 (sparse-attention indexer; BF16 source)",
    "attn_scale": "consumed (F32 block scale, fed to dequant)",
    "self_attn": "FP16 (MLA + linear-attention projections; F8_E4M3 dequant or BF16/F32 cast)",
    "hyper_connection": "FP16 (hyper-connection base/fn/scale params)",
    "mtp_extra": "FP16 (MTP eh_proj / enorm / hnorm / shared_head)",
    "layer_norm": "FP16 (block norms)",
    "final_norm": "FP16 (final norm)",
    "embed_tokens": "FP16 (token embedding)",
    "lm_head": "FP16 (LM head)",
    "vision": "n/a (text-only deployment; vision tower excluded)",
    "other": "FP16 (unclassified; review)",
}


def classify(name: str) -> str:
    for label, pattern in _FAMILY_PATTERNS:
        if pattern.search(name):
            return label
    return "other"


def block_of(name: str, num_hidden_layers: int | None) -> str:
    """main / mtp / vision block a tensor belongs to.

    The single MTP layer sits at index ``num_hidden_layers`` (glm5_next has
    ``num_nextn_predict_layers == 1``); it also carries the ``eh_proj`` / ``enorm``
    / ``hnorm`` / ``shared_head`` MTP-only tensors.
    """
    if name.startswith("model.visual.") or ".visual." in name:
        return "vision"
    if re.search(r"\.(eh_proj|enorm|hnorm)\.weight$|\.shared_head\.", name):
        return "mtp"
    match = re.search(r"\.layers\.(\d+)\.", name)
    if match and num_hidden_layers is not None and int(match.group(1)) >= num_hidden_layers:
        return "mtp"
    return "main"


def build_manifest(ckpt_dir: Path, *, with_sha256: bool) -> dict:
    ckpt_dir = Path(ckpt_dir)
    config = json.loads((ckpt_dir / "config.json").read_text())
    text_config = config.get("text_config", config)
    quant_config = config.get("quantization_config", {})
    index = json.loads((ckpt_dir / _INDEX_FILE).read_text())
    weight_map: dict[str, str] = index["weight_map"]
    num_hidden_layers = text_config.get("num_hidden_layers")

    # Read every shard header once (headers are cheap; the 306 GB payload is
    # never touched). Accumulate authoritative per-family dtype counts + a
    # representative shape sample, plus per-block family counts.
    family_counts: Counter[str] = Counter()
    family_counts_by_block: dict[str, Counter[str]] = defaultdict(Counter)
    block_counts: Counter[str] = Counter()
    family_dtypes: dict[str, Counter[str]] = defaultdict(Counter)
    family_sample: dict[str, dict] = {}

    shard_files = sorted(set(weight_map.values()))
    shard_headers: dict[str, dict] = {}
    for shard in shard_files:
        shard_headers[shard] = read_safetensors_header(ckpt_dir / shard)

    for name, shard in weight_map.items():
        family = classify(name)
        block = block_of(name, num_hidden_layers)
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

    # A representative routed-expert triple (gate/up/down) with weight + scale
    # shapes, so the downstream converter can validate the F32-block geometry.
    expert_layers = sorted(
        {int(m.group(1)) for name in weight_map if (m := re.search(r"\.layers\.(\d+)\.mlp\.experts\.", name))}
    )
    routed_expert_example: dict[str, dict] = {}
    if expert_layers:
        layer_id = expert_layers[0]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            wname = f"model.language_model.layers.{layer_id}.mlp.experts.0.{proj}.weight"
            sname = f"{wname}_scale_inv"
            wshard = weight_map.get(wname)
            sshard = weight_map.get(sname)
            if wshard is None:
                continue
            wmeta = shard_headers[wshard].get(wname, {})
            smeta = shard_headers[sshard].get(sname, {}) if sshard else {}
            routed_expert_example[proj] = {
                "weight": {"tensor": wname, "dtype": wmeta.get("dtype"), "shape": wmeta.get("shape")},
                "weight_scale_inv": {"tensor": sname, "dtype": smeta.get("dtype"), "shape": smeta.get("shape")},
            }

    # Shard file list with sizes (+ optional sha256).
    shards = []
    for shard in shard_files:
        path = ckpt_dir / shard
        entry = {"file": shard, "bytes": path.stat().st_size}
        if with_sha256:
            entry["sha256"] = sha256_file(path)
        shards.append(entry)

    geometry = {k: text_config[k] for k in _GEOMETRY_KEYS if k in text_config}
    vision_count = family_counts.get("vision", 0)
    target_precision = {fam: _TARGET_PRECISION.get(fam, _TARGET_PRECISION["other"]) for fam in family_counts}

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_dir": str(ckpt_dir),
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "geometry": geometry,
        "quantization": {
            "quant_method": quant_config.get("quant_method"),
            "activation_scheme": quant_config.get("activation_scheme"),
            "fmt": quant_config.get("fmt"),
            "weight_block_size": quant_config.get("weight_block_size"),
            "scale_fmt": quant_config.get("scale_fmt"),
            "scale_dtype": "F32 (plain per-block float32 scale; NOT ue8m0)",
        },
        "eos_token_id": config.get("eos_token_id"),
        "bos_token_id": config.get("bos_token_id"),
        "pad_token_id": config.get("pad_token_id"),
        "image_token_id": config.get("image_token_id"),
        "mtp": {
            "num_nextn_predict_layers": text_config.get("num_nextn_predict_layers"),
            "mtp_layer_index": num_hidden_layers,
            "tensor_count": block_counts.get("mtp", 0),
        },
        "vision": {
            "present": vision_count > 0,
            "included_in_deployment": False,
            "reason": "text-only W2-on-310P deployment; vision tower recorded but not loaded",
            "tensor_count": vision_count,
            "config_present": "vision_config" in config,
        },
        "tensor_total": len(weight_map),
        "block_counts": dict(sorted(block_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "family_counts_by_block": {b: dict(sorted(c.items())) for b, c in sorted(family_counts_by_block.items())},
        "family_dtypes": {fam: dict(sorted(c.items())) for fam, c in sorted(family_dtypes.items())},
        "family_samples": dict(sorted(family_sample.items())),
        "target_precision": dict(sorted(target_precision.items())),
        "routed_expert_example": routed_expert_example,
        "naming_scheme": {
            "routed_expert": "model.language_model.layers.{L}.mlp.experts.{E}."
            "{gate_proj|up_proj|down_proj}.{weight|weight_scale_inv}",
            "shared_expert": "model.language_model.layers.{L}.mlp.shared_experts."
            "{gate_proj|up_proj|down_proj}.{weight|weight_scale_inv}",
            "dense_mlp": "model.language_model.layers.{L}.mlp.{gate_proj|up_proj|down_proj}."
            "{weight|weight_scale_inv} (first_k_dense_replace layers)",
            "router": "model.language_model.layers.{L}.mlp.gate.{weight|e_score_correction_bias}",
            "self_attn": "model.language_model.layers.{L}.self_attn.{...}.{weight|weight_scale_inv}",
            "embed_tokens": "model.language_model.embed_tokens.weight",
            "lm_head": "lm_head.weight",
            "final_norm": "model.language_model.norm.weight",
            "mtp": "model.language_model.layers.{num_hidden_layers}.{eh_proj|enorm|hnorm|shared_head...}",
        },
        "num_shards": len(shard_files),
        "shards": shards,
        "sha256_present": with_sha256,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the GLM-5.3-Flash W2 checkpoint manifest")
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--out", required=True, help="manifest JSON output path")
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="compute per-shard SHA256 (slow: reads every byte of the 306 GB payload)",
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
