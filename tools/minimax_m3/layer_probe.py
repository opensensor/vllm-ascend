# SPDX-License-Identifier: Apache-2.0
"""Real M3.1 attention/router weights with seeded hidden states, CPU versus NPU.

This bounded numerical probe is not a full decoder or a language-quality test.
Prepare the bundle on the download host; run it in an isolated NPU process.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

MAX_OUTPUT_NRMSE = 0.01


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare(model, cache, output, layer):
    # Download-only dependencies are not needed by the isolated device runner.
    from tools.minimax_m3.mixed_checkpoint import REVISION, MixedCheckpoint

    source = MixedCheckpoint(model, cache)
    prefix = f"language_model.model.layers.{layer}."
    names = ["input_layernorm.weight", "block_sparse_moe.gate.weight", "block_sparse_moe.e_score_correction_bias"]
    names += [f"self_attn.{p}_proj.weight" for p in ("q", "k", "v", "o", "index_q", "index_k")]
    names += [f"self_attn.{p}_norm.weight" for p in ("q", "k", "index_q", "index_k")]
    tensors = {}
    for name in names:
        print(f"Fetching {prefix + name}", flush=True)
        tensors[name] = source.weight(prefix + name).float().contiguous()
    output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output / "weights.safetensors", metadata={"revision": REVISION, "layer": str(layer)})
    (output / "config.json").write_text(json.dumps(source.config["text_config"], indent=2) + "\n")
    (output / "provenance.json").write_text(
        json.dumps({"revision": REVISION, "layer": layer, "inputs": "seeded synthetic hidden states"}, indent=2) + "\n"
    )


def rms(x, weight, eps):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps) * (1 + weight.float())).half()


def rotate(x, positions, rotary_dim, theta):
    inv = theta ** (-torch.arange(0, rotary_dim, 2, device=x.device).float() / rotary_dim)
    angles = positions.float()[:, None] * inv[None, :]
    cos, sin = angles.cos().half()[:, None, :], angles.sin().half()[:, None, :]
    left, right = x[..., : rotary_dim // 2], x[..., rotary_dim // 2 : rotary_dim]
    return torch.cat((left * cos - right * sin, right * cos + left * sin, x[..., rotary_dim:]), -1)


def project(weights, hidden, config, queries):
    eps, dim = config["rms_norm_eps"], config["head_dim"]
    x = rms(hidden, weights["input_layernorm.weight"], eps)
    positions = torch.arange(len(x), device=x.device)
    outputs = {}
    for name in ("q", "k", "v", "index_q", "index_k"):
        current = x[-queries:] if name in ("q", "index_q") else x
        projected = (current @ weights[f"self_attn.{name}_proj.weight"].half().T).view(len(current), -1, dim)
        if name != "v":
            projected = rms(projected, weights[f"self_attn.{name}_norm.weight"], eps)
            projected = rotate(projected, positions[-len(current) :], config["rotary_dim"], config["rope_theta"])
        outputs[name] = projected
    # Router comparison is independent of attention/residual/FFN composition.
    outputs["router_logits"] = x[-queries:].float() @ weights["block_sparse_moe.gate.weight"].float().T
    return outputs


def selection(index_q, index_k, length, topk, block_size):
    """Diagnostic block selection with explicit causal, current-block forcing."""
    queries = len(index_q)
    count = math.ceil(length / block_size)
    scores = index_q.float() @ index_k.squeeze(1).float().T
    positions = torch.arange(length, device=scores.device)
    qpos = torch.arange(length - queries, length, device=scores.device)
    scores = scores.masked_fill(positions[None, None, :] > qpos[:, None, None], -torch.inf)
    scores = torch.nn.functional.pad(scores, (0, count * block_size - length), value=-torch.inf)
    scores = scores.view(queries, index_q.shape[1], count, block_size).amax(-1)
    blocks = torch.arange(count, device=scores.device)
    scores = torch.where(blocks[None, None, :] == (qpos // block_size)[:, None, None], 1e29, scores)
    ids = scores.topk(min(topk, count), -1).indices
    return scores, ids


def attention(module, projections, config, length, queries):
    block = config["sparse_attention_config"]["sparse_block_size"]
    count = math.ceil(length / block)
    device = projections["q"].device
    heads, dim = config["num_key_value_heads"], config["head_dim"]
    # Different physical page maps catch accidental shared cache indexing.
    pages = torch.arange(count - 1, -1, -1, device=device)
    index_pages = torch.roll(pages, 1)
    kc = torch.full((count, block, heads, dim), torch.nan, device=device, dtype=torch.float16)
    vc = torch.full_like(kc, torch.nan)
    ic = torch.full((count, block, dim), torch.nan, device=device, dtype=torch.float16)
    pos = torch.arange(length, device=device)
    slots = pages[pos // block] * block + pos % block
    index_slots = index_pages[pos // block] * block + pos % block
    prefix = length - queries
    kc.flatten(0, 1).index_copy_(0, slots[:prefix], projections["k"][:prefix])
    vc.flatten(0, 1).index_copy_(0, slots[:prefix], projections["v"][:prefix])
    ic.flatten(0, 1).index_copy_(0, index_slots[:prefix], projections["index_k"][:prefix, 0])
    return module.paged_sparse_attention(
        projections["q"],
        projections["k"][-queries:],
        projections["v"][-queries:],
        projections["index_q"],
        projections["index_k"][-queries:, 0],
        kc,
        vc,
        ic,
        query_starts=[0, queries],
        seq_lens=torch.tensor([length], device=device),
        block_table=pages[None],
        index_block_table=index_pages[None],
        slot_mapping=slots[-queries:],
        index_slot_mapping=index_slots[-queries:],
        max_seq_len=length,
        topk_blocks=config["sparse_attention_config"]["sparse_topk_blocks"],
        scale=dim**-0.5,
    )


def metrics(actual, expected):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    error = actual - expected
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs": error.abs().max().item(),
        "nrmse": (error.norm() / expected.norm().clamp_min(1e-30)).item(),
        "cosine": torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item(),
    }


def run(bundle, ops, device, length, queries):
    started = time.monotonic()
    if device.startswith("npu:"):
        torch.npu.set_device(device)
    if queries < 1 or length < queries:
        raise ValueError("Invalid sequence/query lengths")
    spec = importlib.util.spec_from_file_location("m3_probe_ops", ops)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((bundle / "config.json").read_text())
    weights = load_file(bundle / "weights.safetensors")
    hidden = torch.randn(length, config["hidden_size"], generator=torch.Generator().manual_seed(20261024)).half()
    cpu = project(weights, hidden, config, queries)
    expected = attention(module, cpu, config, length, queries)
    sparse = config["sparse_attention_config"]
    _, reference_ids = selection(
        cpu["index_q"], cpu["index_k"], length, sparse["sparse_topk_blocks"], sparse["sparse_block_size"]
    )
    router_ref, router_ids_ref = module.route_experts(
        cpu["router_logits"], weights["block_sparse_moe.e_score_correction_bias"], config["num_experts_per_tok"]
    )
    actual_weights = {k: v.to(device) for k, v in weights.items()}
    actual = project(actual_weights, hidden.to(device), config, queries)
    _, selected = selection(
        actual["index_q"], actual["index_k"], length, sparse["sparse_topk_blocks"], sparse["sparse_block_size"]
    )
    result = attention(module, actual, config, length, queries)
    # Isolate attention arithmetic from projection/norm/rope differences.
    isolated = attention(module, {k: v.to(device) for k, v in cpu.items()}, config, length, queries)
    router, router_ids = module.route_experts(
        actual["router_logits"],
        actual_weights["block_sparse_moe.e_score_correction_bias"],
        config["num_experts_per_tok"],
    )
    report = {
        "device": device,
        "length": length,
        "queries": queries,
        "provenance": json.loads((bundle / "provenance.json").read_text()),
        "weights_sha256": digest(bundle / "weights.safetensors"),
        "ops_sha256": digest(ops),
        "probe_sha256": digest(Path(__file__)),
        "torch_version": torch.__version__,
        "output_nrmse_limit": MAX_OUTPUT_NRMSE,
        "projection_metrics": {k: metrics(actual[k], cpu[k]) for k in cpu},
        "selected_block_sets_equal": torch.equal(selected.cpu().sort(-1).values, reference_ids.sort(-1).values),
        "selected_blocks_cpu": reference_ids.tolist(),
        "selected_blocks_device": selected.cpu().tolist(),
        "router_ids_equal": torch.equal(router_ids.cpu(), router_ids_ref),
        "router_weights": metrics(router, router_ref),
        "attention": metrics(result, expected),
        "attention_identical_inputs": metrics(isolated, expected),
        "output_projection": metrics(
            result.flatten(1) @ actual_weights["self_attn.o_proj.weight"].half().T,
            expected.flatten(1) @ weights["self_attn.o_proj.weight"].half().T,
        ),
        "full_decoder_validated": False,
        "service_validated": False,
    }
    report["numerical_gate_passed"] = (
        report["selected_block_sets_equal"]
        and report["router_ids_equal"]
        and all(
            report[k]["finite"] and report[k]["nrmse"] <= MAX_OUTPUT_NRMSE
            for k in ("attention", "attention_identical_inputs", "output_projection")
        )
    )
    report["elapsed_seconds_including_cpu_reference_and_compilation"] = time.monotonic() - started
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--ops", type=Path, default=Path("vllm_ascend/_310p/ops/minimax_m3.py"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--length", type=int, default=2177)
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.prepare:
        prepare(args.model, args.cache, args.bundle, args.layer)
        return
    report = run(args.bundle, args.ops, args.device, args.length, args.queries)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["numerical_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
