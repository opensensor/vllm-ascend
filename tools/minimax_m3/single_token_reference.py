# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU next-token reference for the converted full M3 model.

At context length one, causal attention equals the value projection, so this
gate deliberately does not validate QK/RoPE or sparse block selection. It does
exercise all residuals, dense/shared MLPs, routed experts and the output head.
Only selected experts are unpacked, one projection at a time.
Short sequences also exercise QK normalization, rotary positions and causal
attention. They must fit within the model's sparse block-selection budget.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer

from tools.minimax_m3.checkpoint import checked_tensors, read_manifest
from tools.minimax_m3.layer_probe import rms, rotate
from tools.minimax_m3.mixed_checkpoint import dequant_nvfp4
from vllm_ascend._310p.ops.minimax_m3 import clipped_swiglu, route_experts, unpack_w4


def run(folder, output, token_id, tp, source_experts=None):
    """Trace short prompts with TP rounding and bounded expert reconstruction."""
    manifest = read_manifest(folder)
    config = json.loads((folder / "config.json").read_text())["text_config"]
    token_ids = [token_id] if isinstance(token_id, int) else list(token_id)
    sparse = config["sparse_attention_config"]
    if not token_ids or len(token_ids) > sparse["sparse_topk_blocks"] * sparse["sparse_block_size"]:
        raise ValueError("CPU sequence reference requires all causal blocks to fit the sparse selection budget")
    output.mkdir(parents=True, exist_ok=True)
    source_map = (
        json.loads((source_experts / "model.safetensors.index.json").read_text())["weight_map"]
        if source_experts is not None
        else None
    )

    def weight(name):
        if source_map is not None and manifest["weights"][name]["expert"]:
            values = []
            for suffix in (".weight_packed", ".weight_scale", ".weight_global_scale"):
                key = name.removesuffix(".weight") + suffix
                with safe_open(source_experts / source_map[key], framework="pt") as handle:
                    values.append(handle.get_tensor(key))
            return dequant_nvfp4(*values).half()
        values = dict(checked_tensors(folder, manifest["weights"][name]))
        if "codes" in values:
            return unpack_w4(values["codes"], values["scales"]).half()
        return values[name]

    def linear(x, name):
        return (x.float() @ weight(name).float().T).half()

    def mlp(x, prefix):
        gate = linear(x, prefix + "gate_proj.weight")
        up = linear(x, prefix + "up_proj.weight")
        act = clipped_swiglu(gate, up, config["swiglu_alpha"], limit=config["swiglu_limit"]).half()
        down = weight(prefix + "down_proj.weight")
        return [(a.float() @ w.float().T).half() for a, w in zip(act.chunk(tp, -1), down.chunk(tp, -1))]

    embed = "language_model.model.embed_tokens.weight"
    with safe_open(folder / manifest["weights"][embed]["file"], framework="pt") as handle:
        view = handle.get_slice(embed)
        hidden = torch.cat([view[i : i + 1] for i in token_ids])
    residual = None
    trace = {}
    routing = []
    started = time.monotonic()
    for layer in range(config["num_hidden_layers"]):
        prefix = f"language_model.model.layers.{layer}."
        residual = hidden if residual is None else (hidden + residual).half()
        x = rms(residual, weight(prefix + "input_layernorm.weight"), config["rms_norm_eps"])
        value = linear(x, prefix + "self_attn.v_proj.weight")
        repeats = config["num_attention_heads"] // config["num_key_value_heads"]
        values = value.view(len(x), config["num_key_value_heads"], config["head_dim"])
        if len(x) == 1:
            attention = values.repeat_interleave(repeats, dim=1).flatten(1)
        else:
            qk = []
            for projection in ("q", "k"):
                projected = linear(x, prefix + f"self_attn.{projection}_proj.weight")
                projected = projected.view(len(x), -1, config["head_dim"])
                normalized = rms(
                    projected, weight(prefix + f"self_attn.{projection}_norm.weight"), config["rms_norm_eps"]
                )
                qk.append(rotate(normalized, torch.arange(len(x)), config["rotary_dim"], config["rope_theta"]))
            query = qk[0].view(len(x), config["num_key_value_heads"], repeats, config["head_dim"])
            scores = torch.einsum("qhrd,khd->hrqk", query.float(), qk[1].float()) / config["head_dim"] ** 0.5
            causal = torch.ones(len(x), len(x), dtype=torch.bool).tril()
            probs = scores.masked_fill(~causal, -torch.inf).softmax(-1)
            attention = torch.einsum("hrqk,khd->qhrd", probs, values.float()).flatten(1).half()
        o = weight(prefix + "self_attn.o_proj.weight")
        partials = [(a.float() @ w.float().T).half() for a, w in zip(attention.chunk(tp, -1), o.chunk(tp, -1))]
        hidden = torch.stack(partials).sum(0).half()
        residual = (residual + hidden).half()
        x = rms(residual, weight(prefix + "post_attention_layernorm.weight"), config["rms_norm_eps"])
        if not config["moe_layer_freq"][layer]:
            hidden = torch.stack(mlp(x, prefix + "mlp.")).sum(0).half()
        else:
            prefix += "block_sparse_moe."
            logits = x.float() @ weight(prefix + "gate.weight").float().T
            scores, ids = route_experts(
                logits, weight(prefix + "e_score_correction_bias"), config["num_experts_per_tok"]
            )
            contributions = torch.zeros(len(x), ids.shape[1], x.shape[1], dtype=torch.float32)
            for expert in ids.unique().tolist():
                tokens, routes = torch.where(ids == expert)
                expert_prefix = prefix + f"experts.{expert}."
                gate = linear(x[tokens], expert_prefix + "w1.weight")
                up = linear(x[tokens], expert_prefix + "w3.weight")
                act = clipped_swiglu(gate, up, config["swiglu_alpha"], limit=config["swiglu_limit"]).half()
                contributions[tokens, routes] = (
                    linear(x=act, name=expert_prefix + "w2.weight").float() * scores[tokens, routes, None]
                )
            owners = ids // (config["num_local_experts"] // tp)
            local = [torch.where((owners == rank)[..., None], contributions, 0).sum(1) for rank in range(tp)]
            shared = mlp(x, prefix + "shared_experts.")
            hidden = sum(a * config["routed_scaling_factor"] + b.float() for a, b in zip(local, shared)).half()
            routing.append({"layer": layer, "experts": ids.tolist(), "weights": scores.tolist()})
        if not torch.isfinite(hidden).all():
            raise ValueError(f"Nonfinite CPU reference at layer {layer}")
        trace[f"layer.{layer}.hidden"] = hidden.clone()
        trace[f"layer.{layer}.residual"] = residual.clone()
        print(json.dumps({"layer": layer, "elapsed_seconds": time.monotonic() - started}), flush=True)
    hidden = rms((hidden + residual).half(), weight("language_model.model.norm.weight"), config["rms_norm_eps"])
    hidden = hidden[-1:]
    name = "language_model.lm_head.weight"
    with safe_open(folder / manifest["weights"][name]["file"], framework="pt") as handle:
        view = handle.get_slice(name)
        logits = torch.cat(
            [
                (hidden.float() @ view[start : start + 2048].float().T).half().float()
                for start in range(0, config["vocab_size"], 2048)
            ],
            -1,
        )
    trace["logits"] = logits
    save_file(trace, output / "trace.safetensors")
    values, ids = logits.log_softmax(-1).topk(10, dim=-1)
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    report = {
        "token_id": token_id,
        "token_ids": token_ids,
        "tensor_parallel_rounding": tp,
        "source_experts": str(source_experts) if source_experts is not None else None,
        "elapsed_seconds": time.monotonic() - started,
        "top_tokens": [
            {"id": i, "text": tokenizer.decode([i]), "logprob": v} for i, v in zip(ids[0].tolist(), values[0].tolist())
        ],
        "routing": routing,
        "scope": "FP16 dense layers/activations; excludes sparse block selection and source activation quantization",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "routing"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--token-id", type=int)
    inputs.add_argument("--prompt", type=str)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--source-experts", type=Path, help="Compare decoded source NVFP4 expert weights; FP16 activations"
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    token_ids = args.token_id
    if args.prompt is not None:
        token_ids = Tokenizer.from_file(str(args.model / "tokenizer.json")).encode(args.prompt).ids
    with torch.inference_mode():
        run(args.model, args.output, token_ids, args.tp, args.source_experts)


if __name__ == "__main__":
    main()
