# SPDX-License-Identifier: Apache-2.0
"""Load a complete real M3 expert bank and check its TP2/EP2 CPU execution."""

import argparse
import gc
import json
import resource
import time
from pathlib import Path

import torch
from safetensors import safe_open

from tools.minimax_m3.checkpoint import checked_tensors, load_expert_banks, read_manifest
from tools.minimax_m3.export_w4 import local_file
from vllm_ascend._310p.ops.minimax_m3 import PackedM3Experts, clipped_swiglu, route_experts


def load_dense(folder, manifest, name):
    return dict(checked_tensors(folder, manifest["weights"][name]))[name]


def ep_view(full, rank):
    # Meta construction avoids allocating and discarding another multi-GiB bank.
    with torch.device("meta"):
        result = PackedM3Experts(
            full.w1_codes.shape[-1] * 2,
            full.w1_codes.shape[1],
            full.num_local_experts,
            2,
            rank,
            full.top_k,
            prefix=full.prefix,
            alpha=full.alpha,
            beta=full.beta,
            limit=full.limit,
            routed_scale=full.routed_scale,
        )
    start = result.local_expert_offset
    for name, value in full.named_buffers():
        setattr(result, name, value[start : start + result.num_local_experts])
    return result


def source_routed(source, index, prefix, x, logits, bias, top_k):
    weights, ids = route_experts(logits, bias, top_k)
    result = torch.zeros_like(x, dtype=torch.float32)
    for expert in sorted(set(ids.flatten().tolist())):
        projections = {}
        for projection in ("w1", "w3", "w2"):
            name = f"{prefix}.experts.{expert}.{projection}.weight"
            with safe_open(local_file(source, index[name]), framework="pt", device="cpu") as handle:
                weight = handle.get_tensor(name).half().float()
            if projection == "w2":
                act = clipped_swiglu(projections["w1"], projections["w3"]).half()
                output = (act.float() @ weight.T).half().float()
            else:
                projections[projection] = (x.float() @ weight.T).half()
        factor = (weights * (ids == expert)).sum(-1)
        result += output * factor[:, None] * 2
    return result, sorted(set(ids.flatten().tolist()))


def probe(source, folder, layer, tokens):
    manifest = read_manifest(folder)
    config = json.loads((folder / "config.json").read_text())["text_config"]
    if layer not in range(config["num_hidden_layers"]) or not config["moe_layer_freq"][layer]:
        raise ValueError("Choose a real MoE layer")
    if not 1 <= tokens <= 32:
        raise ValueError("Bounded real-bank probe requires 1..32 tokens")
    prefix = f"language_model.model.layers.{layer}.block_sparse_moe"
    started = time.monotonic()
    bank = PackedM3Experts(
        config["hidden_size"],
        config["intermediate_size"],
        config["num_local_experts"],
        1,
        0,
        config["num_experts_per_tok"],
        prefix=prefix + ".experts",
    )
    loaded = load_expert_banks(bank, folder, manifest)
    load_seconds = time.monotonic() - started
    print(f"Loaded {loaded} buffers for layer {layer} in {load_seconds:.1f}s", flush=True)
    x = torch.randn(tokens, config["hidden_size"], generator=torch.Generator().manual_seed(313)).half()
    gate = load_dense(folder, manifest, prefix + ".gate.weight").float()
    bias = load_dense(folder, manifest, prefix + ".e_score_correction_bias").float()
    logits = x.float() @ gate.T
    full = bank(x, logits, bias)
    partials = [ep_view(bank, rank)(x, logits, bias) for rank in range(2)]
    combined = sum(partials)
    torch.testing.assert_close(combined, full, rtol=1e-5, atol=1e-5)
    # Exercise shared MLP TP partitioning with real exported dense weights.
    gate_up = []
    for projection in ("gate_proj", "up_proj"):
        weight = load_dense(folder, manifest, prefix + f".shared_experts.{projection}.weight")
        gate_up.append((x.float() @ weight.float().T).half())
    activation = clipped_swiglu(*gate_up).half()
    down = load_dense(folder, manifest, prefix + ".shared_experts.down_proj.weight").float()
    shared_parts = [(a.float() @ w.T).half().float() for a, w in zip(activation.chunk(2, -1), down.chunk(2, -1))]
    output = (partials[0] + shared_parts[0] + partials[1] + shared_parts[1]).half()
    if not torch.isfinite(output).all():
        raise ValueError("Nonfinite real MoE output")
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    reference, experts = source_routed(source, index, prefix, x, logits, bias, bank.top_k)
    delta = combined - reference
    report = dict(
        layer=layer,
        tokens=tokens,
        expert_count=bank.num_local_experts,
        loaded_buffers=loaded,
        bank_bytes=sum(t.numel() * t.element_size() for t in bank.buffers()),
        load_seconds=load_seconds,
        total_seconds=time.monotonic() - started,
        cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        ep2_max_abs_error=(combined - full).abs().max().item(),
        combined_shared_routed_finite=True,
        output_shape=list(output.shape),
        selected_experts=experts,
        routed_source_cosine=torch.nn.functional.cosine_similarity(
            combined.flatten(), reference.flatten(), dim=0
        ).item(),
        routed_source_nrmse=(delta.norm() / reference.norm()).item(),
        hardware_validated=False,
        full_decoder_validated=False,
        scope="Full real expert bank, router and TP shared MLP; seeded random inputs, no attention or model quality",
    )
    del bank
    gc.collect()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("export", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    with torch.inference_mode():
        result = probe(args.source, args.export, args.layer, args.tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
