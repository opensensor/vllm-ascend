# SPDX-License-Identifier: Apache-2.0
"""Independent, streamed source-weight reference for complete M3.1 histories.

This diagnostic deliberately imports no Ascend model/operator implementation.
It propagates its own hidden states and computes attention over its own history.
FP16 activations are an explicit comparison convention, not original QAT.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from safetensors import safe_open

from tools.minimax_m3.activation_reference import mxfp8_roundtrip, nvfp4_roundtrip


class SourceWeights:
    def __init__(self, root, device):
        self.root, self.device = Path(root), device
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(self, name, rows=None):
        with safe_open(self.root / self.index[name], framework="pt", device="cpu") as handle:
            value = handle.get_tensor(name) if rows is None else handle.get_slice(name)[rows]
            # The synchronous transfer finishes before releasing the mapped file.
            return value.to(self.device, copy=True)

    def weight(self, name):
        base = name.removesuffix(".weight")
        if base + ".weight_packed" in self.index:
            packed = self.tensor(base + ".weight_packed")
            scale = self.tensor(base + ".weight_scale").float() / self.tensor(base + ".weight_global_scale").float()
            lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device=self.device)
            values = torch.stack((lut[(packed & 15).long()], lut[(packed >> 4).long()]), -1).flatten(-2)
            return (values.view(*scale.shape, 16) * scale[..., None]).flatten(-2).half()
        value = self.tensor(name)
        if base + ".weight_scale_inv" in self.index:
            scale = torch.exp2(self.tensor(base + ".weight_scale_inv").float() - 127)
            return (value.float().view(*scale.shape, 32) * scale[..., None]).flatten(-2).half()
        return value if value.dtype == torch.float32 else value.half()


def normalize(x, weight, epsilon):
    values = x.float()
    return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon) * (1 + weight.float())).half()


def rotary_cache(length, dimensions, theta):
    # Match the source/HF frequency construction on CPU. Algebraically
    # equivalent negative powers on CUDA round differently at long positions.
    inverse = 1.0 / (theta ** (torch.arange(0, dimensions, 2, dtype=torch.float32) / dimensions))
    angles = torch.arange(length, dtype=torch.float32)[:, None] * inverse
    return angles.cos().half()[:, None], angles.sin().half()[:, None]


def rotary(x, dimensions, cosine, sine):
    left, right = x[..., : dimensions // 2], x[..., dimensions // 2 : dimensions]
    return torch.cat((left * cosine - right * sine, right * cosine + left * sine, x[..., dimensions:]), -1)


def activate(gate, up, config):
    gate = gate.float().clamp(max=config["swiglu_limit"])
    up = up.float().clamp(-config["swiglu_limit"], config["swiglu_limit"])
    return (gate * torch.sigmoid(config["swiglu_alpha"] * gate) * (up + config.get("swiglu_beta", 1))).half()


def sparse_attention(query, key, value, index_query, index_key, config, tile=64):
    """Dense token attention mask derived from per-head block selections.

    Unlike the serving gather path, this keeps keys in chronological order and
    masks every unselected token. Only query rows are tiled; history is complete.
    """
    query_length, heads, width = query.shape
    length = len(key)
    offset = length - query_length
    if offset < 0:
        raise ValueError("Attention queries cannot exceed the key history")
    kv_heads = key.shape[1]
    block_size = config["sparse_block_size"]
    blocks = math.ceil(length / block_size)
    tokens = torch.arange(length, device=query.device)
    block_ids = torch.arange(blocks, device=query.device)
    output = torch.empty_like(query)
    selections = torch.empty(
        query_length, kv_heads, min(config["sparse_topk_blocks"], blocks), device=query.device, dtype=torch.long
    )
    for start in range(0, query_length, tile):
        stop = min(query_length, start + tile)
        positions = tokens[start + offset : stop + offset]
        causal = tokens[None, :] <= positions[:, None]
        scores = torch.einsum("qhd,kd->qhk", index_query[start:stop].float(), index_key.float())
        scores = scores.masked_fill(~causal[:, None], -torch.inf)
        scores = torch.nn.functional.pad(scores, (0, blocks * block_size - length), value=-torch.inf)
        scores = scores.reshape(stop - start, kv_heads, blocks, block_size).amax(-1)
        current = positions[:, None] // block_size
        visible = block_ids[None] <= current
        forced = visible & (block_ids[None] > current - config["sparse_local_block"])
        scores = scores.masked_fill(forced[:, None], 1e29)
        scores = scores.masked_fill((visible & (block_ids[None] < config["sparse_init_block"]))[:, None], 1e30)
        selected = scores.topk(selections.shape[-1], -1).indices
        selections[start:stop] = selected
        selected_mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, selected, True)
        allowed = selected_mask.repeat_interleave(block_size, -1)[..., :length] & causal[:, None]
        for head in range(kv_heads):
            begin, end = head * (heads // kv_heads), (head + 1) * (heads // kv_heads)
            logits = torch.einsum(
                "qgd,kd->qgk", query[start:stop, begin:end].float(), key[:, head].float()
            ) / math.sqrt(width)
            probability = logits.masked_fill(~allowed[:, head, None], -torch.inf).softmax(-1)
            output[start:stop, begin:end] = torch.einsum("qgk,kd->qgd", probability, value[:, head].float()).half()
    return output.flatten(1), selections


def difference(actual, expected):
    a, b = actual.float().cpu().flatten(), expected.float().cpu().flatten()
    return {"nrmse": float((a - b).norm() / b.norm().clamp_min(1e-20)), "max_abs": float((a - b).abs().max())}


def run(args):
    # Disable reduced-precision CUDA accumulation to keep the oracle independent
    # of the Ascend kernels while making its FP16 rounding convention explicit.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_gib * (1 << 30) / torch.cuda.get_device_properties(args.device).total_memory, args.device
        )
    source = SourceWeights(args.model, args.device)
    config = json.loads((args.model / "config.json").read_text())["text_config"]
    token_ids = json.loads(args.tokens.read_text())["tokens"]
    args.output.mkdir(parents=True, exist_ok=True)
    observations = {}
    if args.observed:
        for rank in range(6):
            observations[rank] = torch.load(args.observed / f"rank{rank}.pt", map_location="cpu", weights_only=True)

    def quantize_input(x, name):
        if args.activation_format == "fp16":
            return x
        base = name.removesuffix(".weight")
        if base + ".weight_packed" in source.index:
            if not torch.equal(
                source.tensor(base + ".input_global_scale"),
                torch.ones_like(source.tensor(base + ".input_global_scale")),
            ):
                raise ValueError("NVFP4 activation oracle currently requires unit input global scales")
            return nvfp4_roundtrip(x)
        if base + ".weight_scale_inv" in source.index:
            return mxfp8_roundtrip(x)
        return x

    def linear(x, name):
        x = quantize_input(x, name)
        weight = source.weight(name)
        return (x.float() @ weight.float().T).half()

    def mlp(x, prefix):
        gate, up = linear(x, prefix + "gate_proj.weight"), linear(x, prefix + "up_proj.weight")
        activated = activate(gate, up, config)
        activated = quantize_input(activated, prefix + "down_proj.weight")
        weight = source.weight(prefix + "down_proj.weight")
        return [
            (a.float() @ w.float().T).half() for a, w in zip(activated.chunk(args.tp, -1), weight.chunk(args.tp, -1))
        ]

    prefix_cache = getattr(args, "prefix_cache", None)
    prefix = torch.load(prefix_cache / "checkpoint.pt", map_location="cpu", weights_only=True) if prefix_cache else None
    prefix_length = len(prefix["tokens"]) if prefix is not None else 0
    input_ids = token_ids[prefix_length:]
    if not input_ids:
        raise ValueError("Reference input must extend the prefix by at least one token")
    # Slice on CPU before transferring; never retain the full embedding matrix.
    embed = "language_model.model.embed_tokens.weight"
    with safe_open(args.model / source.index[embed], framework="pt", device="cpu") as handle:
        hidden = handle.get_tensor(embed)[input_ids].to(args.device).half()
    cosine, sine = (
        value.to(args.device) for value in rotary_cache(len(token_ids), config["rotary_dim"], config["rope_theta"])
    )
    identity = {
        "model": str(args.model.resolve()),
        "config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256((args.model / "model.safetensors.index.json").read_bytes()).hexdigest(),
        "tp": args.tp,
        "activation_format": args.activation_format,
        "rotary": "cpu_reciprocal_power",
    }
    if prefix_cache:
        if (
            prefix["tokens"] != token_ids[:prefix_length]
            or prefix.get("identity") != identity
            or prefix["next_layer"] != config["num_hidden_layers"]
        ):
            raise ValueError("Prefix cache history or reference identity mismatch, or incomplete layers")
        del prefix
        cosine, sine = cosine[prefix_length:], sine[prefix_length:]
    residual = None
    reports, last_rows = [], {}
    first_layer = 0
    checkpoint = args.output / "checkpoint.pt"
    if args.resume:
        saved = torch.load(checkpoint, map_location=args.device, weights_only=True)
        if (
            saved["tokens"] != token_ids
            or saved.get("identity") != identity
            or saved.get("prefix_length", 0) != prefix_length
        ):
            raise ValueError("Checkpoint reference identity or token history mismatch")
        first_layer, hidden, residual = saved["next_layer"], saved["hidden"], saved["residual"]
        reports = json.loads((args.output / "layers.json").read_text())[:first_layer]
        last_rows = {
            layer: row
            for layer, row in torch.load(args.output / "last_rows.pt", map_location="cpu", weights_only=True).items()
            if layer < first_layer
        }
    started = time.monotonic()
    for layer in range(first_layer, min(args.layers, config["num_hidden_layers"])):
        prefix = f"language_model.model.layers.{layer}."
        residual = hidden if residual is None else (hidden + residual).half()
        x = normalize(residual, source.weight(prefix + "input_layernorm.weight"), config["rms_norm_eps"])
        projected = {}
        for name in ("q", "k", "v", "index_q", "index_k"):
            y = linear(x, prefix + f"self_attn.{name}_proj.weight").view(len(x), -1, config["head_dim"])
            if name != "v":
                y = normalize(y, source.weight(prefix + f"self_attn.{name}_norm.weight"), config["rms_norm_eps"])
                y = rotary(y, config["rotary_dim"], cosine, sine)
            projected[name] = y
        if prefix_cache:
            cached = torch.load(prefix_cache / f"kv-{layer:02d}.pt", map_location=args.device, weights_only=True)
            for name in ("k", "v", "index_k"):
                projected[name] = torch.cat((cached[name], projected[name]), 0)
            del cached
        attention, selected = sparse_attention(
            projected["q"],
            projected["k"],
            projected["v"],
            projected["index_q"],
            projected["index_k"].squeeze(1),
            config["sparse_attention_config"],
        )
        if args.save_kv:
            torch.save({k: projected[k].cpu() for k in ("k", "v", "index_k")}, args.output / f"kv-{layer:02d}.pt")
        attention = quantize_input(attention, prefix + "self_attn.o_proj.weight")
        weight = source.weight(prefix + "self_attn.o_proj.weight")
        hidden = sum(
            (a.float() @ w.float().T).half() for a, w in zip(attention.chunk(args.tp, -1), weight.chunk(args.tp, -1))
        )
        residual = (residual + hidden).half()
        x = normalize(residual, source.weight(prefix + "post_attention_layernorm.weight"), config["rms_norm_eps"])
        report = {"layer": layer}
        if not config["moe_layer_freq"][layer]:
            hidden = sum(mlp(x, prefix + "mlp.")).half()
        else:
            prefix += "block_sparse_moe."
            scores = (x.float() @ source.weight(prefix + "gate.weight").float().T).sigmoid()
            ids = (
                (scores + source.weight(prefix + "e_score_correction_bias"))
                .topk(config["num_experts_per_tok"], -1)
                .indices
            )
            probabilities = scores.gather(-1, ids)
            probabilities /= probabilities.sum(-1, keepdim=True)
            contributions = torch.zeros(len(x), ids.shape[1], x.shape[1], device=args.device, dtype=torch.float32)
            for expert in ids.unique().tolist():
                tokens, routes = torch.where(ids == expert)
                ep = prefix + f"experts.{expert}."
                gate, up = linear(x[tokens], ep + "w1.weight"), linear(x[tokens], ep + "w3.weight")
                value = linear(activate(gate, up, config), ep + "w2.weight")
                contributions[tokens, routes] = value.float() * probabilities[tokens, routes, None]
            owners = ids // (config["num_local_experts"] // args.tp)
            local = [torch.where((owners == rank)[..., None], contributions, 0).sum(1) for rank in range(args.tp)]
            hidden = sum(
                a * config["routed_scaling_factor"] + b.float()
                for a, b in zip(local, mlp(x, prefix + "shared_experts."))
            ).half()
            report["last_routing_ids"] = ids[-1].tolist()
            del contributions, local
        if not torch.isfinite(hidden).all():
            raise ValueError(f"Nonfinite independently propagated output at layer {layer}")
        last_rows[layer] = {
            "hidden": hidden[-1:].cpu(),
            "residual": residual[-1:].cpu(),
            "selected": selected[-1:].cpu(),
        }
        if observations:
            observed = [v[(layer, "prefill")] for v in observations.values() if (layer, "prefill") in v]
            report["hidden_vs_npu"] = difference(hidden[-1:], observed[0]["hidden_out"])
            report["residual_vs_npu"] = difference(residual[-1:], observed[0]["residual_out"])
            reference_selection = torch.cat(
                [r["selection"][head]["selected"] for r in observed for head in sorted(r["selection"])], 0
            )
            report["selected_blocks_equal"] = torch.equal(
                selected[-1].cpu().sort().values, reference_selection.sort().values
            )
            if "last_routing_ids" in report:
                report["routing_ids_equal"] = report["last_routing_ids"] == observed[0]["routing_ids"][0].tolist()
        report["seconds_this_invocation"] = time.monotonic() - started
        reports.append(report)
        (args.output / "layers.json").write_text(json.dumps(reports, indent=2) + "\n")
        torch.save(last_rows, args.output / "last_rows.pt")
        torch.save(
            {
                "identity": identity,
                "prefix_length": prefix_length,
                "tokens": token_ids,
                "next_layer": layer + 1,
                "hidden": hidden.cpu(),
                "residual": residual.cpu(),
            },
            checkpoint.with_suffix(".tmp"),
        )
        checkpoint.with_suffix(".tmp").replace(checkpoint)
        print(json.dumps(report), flush=True)
    if args.layers < config["num_hidden_layers"]:
        return
    tail_logits = getattr(args, "tail_logits", 1)
    if tail_logits < 1:
        raise ValueError("tail_logits must be positive")
    final = normalize(
        (hidden[-tail_logits:] + residual[-tail_logits:]).half(),
        source.weight("language_model.model.norm.weight"),
        config["rms_norm_eps"],
    )
    logits = torch.cat(
        [
            (final.float() @ source.tensor("language_model.lm_head.weight", slice(start, start + 2048)).float().T)
            .half()
            .float()
            .cpu()
            for start in range(0, config["vocab_size"], 2048)
        ],
        -1,
    )
    torch.save(logits, args.output / "logits.pt")
    values, indices = logits.log_softmax(-1).topk(10)
    result = {
        "identity": identity,
        "tokens": len(token_ids),
        "top_ids": indices[-1].tolist(),
        "top_logprobs": values[-1].tolist(),
        "logit_rows": len(logits),
        "scope": (
            "Independently propagated entire history, source weights, FP16 activations; "
            "no Ascend intermediate inputs; activation format recorded in identity; excludes original KV quantization"
        ),
    }
    if observations:
        target = observations[4][(59, "prefill")]["logits"]
        result.update(
            logits_vs_npu=difference(logits[-1:], target),
            top10_equal=torch.equal(indices[-1:], target.topk(10).indices),
        )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observed", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cuda-memory-gib", type=float, default=8)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--layers", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--activation-format", choices=("fp16", "nvfp4-mxfp8"), default="fp16")
    parser.add_argument("--save-kv", action="store_true")
    parser.add_argument("--prefix-cache", type=Path, help="Complete independent prefix cache; append a token suffix")
    parser.add_argument("--tail-logits", type=int, default=1, help="Number of trailing input rows to save logits for")
    args = parser.parse_args()
    torch.set_num_threads(4)
    with torch.inference_mode():
        run(args)


if __name__ == "__main__":
    main()
