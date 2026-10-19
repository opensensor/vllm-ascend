# SPDX-License-Identifier: Apache-2.0
"""Replay saved QFN DSpark traces against explicit full-context attention.

This diagnostic consumes target features from a real run. It does not claim
target-weight parity or exact arithmetic equivalence across accelerator types.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file


def difference(actual, expected):
    actual, expected = actual.float(), expected.to(actual.device).float()
    error = actual - expected
    return {
        "relative_rms": float(error.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-12)),
        "max_abs": float(error.abs().max()),
        "cosine": float(F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)),
    }


def prefix_length(proposed, expected):
    length = 0
    for actual, wanted in zip(proposed, expected):
        if actual != wanted:
            break
        length += 1
    return length


def summarize(trace, records):
    """Compare proposals on the recorded trajectory, not a new generation.

    A changed proposal can change future target execution, so reference prefix
    agreement is only an offline diagnostic. The final round is unverified.
    """
    truth = {}
    for step in trace:
        anchor = int(step["query_inputs"]["positions"][0])
        for position, token in zip(step["target_positions"].tolist(), step["target_token_ids"].tolist()):
            if position < anchor:
                truth[position] = token
        truth[anchor] = int(step["next_token_ids"].item())
    summary = {
        "rounds": len(records),
        "max_final_relative_rms": max(record["final"]["relative_rms"] for record in records),
        "min_final_cosine": min(record["final"]["cosine"] for record in records),
        "proposal_comparison": {},
    }
    for mode in ("cuda_head_on_ascend_hidden", "cuda_reference"):
        matching_tokens = matching_blocks = total_tokens = 0
        observed_accepted = reference_prefix = compared_rounds = 0
        disagreements = []
        for record in records:
            if "proposals" not in record:
                continue
            index = record["step"]
            expected = record["proposals"]["ascend"]
            actual = record["proposals"][mode]["tokens"]
            matching_tokens += sum(a == b for a, b in zip(actual, expected))
            total_tokens += len(expected)
            matching_blocks += actual == expected
            if actual != expected:
                disagreements.append(index)
            if index + 1 == len(trace):
                continue
            anchor = int(trace[index]["query_inputs"]["positions"][0])
            next_anchor = int(trace[index + 1]["query_inputs"]["positions"][0])
            accepted = next_anchor - anchor - 1
            if not 0 <= accepted <= len(expected):
                raise ValueError("trace must contain consecutive rounds from one request")
            target = [truth[position] for position in range(anchor + 1, min(anchor + len(expected), max(truth)) + 1)]
            if prefix_length(expected, target) != accepted:
                raise ValueError("recorded proposals disagree with the committed target trajectory")
            observed_accepted += accepted
            reference_prefix += prefix_length(actual, target)
            compared_rounds += 1
        summary["proposal_comparison"][mode] = {
            "matching_tokens": matching_tokens,
            "total_tokens": total_tokens,
            "matching_blocks": matching_blocks,
            "disagreeing_rounds": disagreements,
            "verified_rounds": compared_rounds,
            "observed_accepted": observed_accepted,
            "reference_prefix_matches_on_recorded_continuation": reference_prefix,
        }
    return summary


class Reference:
    def __init__(self, checkpoint, embeddings, device):
        self.config = json.loads((checkpoint / "config.json").read_text())
        self.weights = {
            k: v.to(device=device, dtype=torch.float16)
            for k, v in load_file(str(checkpoint / "model.safetensors")).items()
        }
        self.device = device
        self.rows = {int(i): row.to(device) for i, row in zip(embeddings["ids"], embeddings["rows"])}
        self.head_dim = self.config["head_dim"]
        self.heads = self.config["num_attention_heads"]
        self.kv_heads = self.config["num_key_value_heads"]

    def propose(self, hidden, anchor, lm_head):
        # vLLM's logits processor promotes each projection to FP32 before
        # adding the Markov bias. The next bias uses the previous draft token.
        base = F.linear(hidden, lm_head).float()
        previous = int(anchor)
        tokens, margins = [], []
        for logits in base:
            embedded = self.weights["markov_head.markov_w1.weight"][previous]
            bias = self.linear(embedded, "markov_head.markov_w2.weight").float()
            values, indices = (logits + bias).topk(2)
            previous = int(indices[0])
            tokens.append(previous)
            margins.append(float(values[0] - values[1]))
        return {"tokens": tokens, "margins": margins}

    def confidence(self, hidden, anchor, proposals):
        previous = torch.tensor([int(anchor), *proposals[:-1]], device=self.device)
        embedded = self.weights["markov_head.markov_w1.weight"][previous]
        features = torch.cat((hidden, embedded), dim=-1)
        logits = F.linear(
            features,
            self.weights["confidence_head.proj.weight"],
            self.weights["confidence_head.proj.bias"],
        )
        return logits.float().sigmoid().flatten().tolist()

    def norm(self, inputs, key):
        x = inputs.float()
        normed = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.config["rms_norm_eps"])
        return (normed * self.weights[key].float()).half()

    def linear(self, inputs, key):
        return F.linear(inputs, self.weights[key])

    def rope(self, inputs, positions):
        inv = self.config["rope_theta"] ** (
            -torch.arange(0, self.head_dim, 2, device=self.device).float() / self.head_dim
        )
        angles = positions.float()[:, None] * inv[None, :]
        cosine, sine = angles.cos().repeat(1, 2)[:, None], angles.sin().repeat(1, 2)[:, None]
        left, right = inputs.float().chunk(2, dim=-1)
        return (inputs.float() * cosine + torch.cat((-right, left), dim=-1) * sine).half()

    def layer(self, index, hidden, residual, context, context_positions, query_positions):
        prefix = f"layers.{index}."
        residual = hidden if residual is None else hidden + residual
        x = self.norm(residual, prefix + "input_layernorm.weight")
        q = self.linear(x, prefix + "self_attn.q_proj.weight").view(-1, self.heads, self.head_dim)
        k = torch.cat(
            (
                self.linear(context, prefix + "self_attn.k_proj.weight"),
                self.linear(x, prefix + "self_attn.k_proj.weight"),
            )
        )
        v = torch.cat(
            (
                self.linear(context, prefix + "self_attn.v_proj.weight"),
                self.linear(x, prefix + "self_attn.v_proj.weight"),
            )
        )
        k = k.view(-1, self.kv_heads, self.head_dim)
        v = v.view(-1, self.kv_heads, self.head_dim)
        q = self.rope(self.norm(q, prefix + "self_attn.q_norm.weight"), query_positions)
        k = self.rope(self.norm(k, prefix + "self_attn.k_norm.weight"), torch.cat((context_positions, query_positions)))
        repeats = self.heads // self.kv_heads
        q = q.transpose(0, 1).float()
        k = k.repeat_interleave(repeats, dim=1).transpose(0, 1).float()
        v = v.repeat_interleave(repeats, dim=1).transpose(0, 1).float()
        scores = q @ k.transpose(-1, -2) * self.head_dim**-0.5
        attention = (scores.softmax(dim=-1) @ v).transpose(0, 1).reshape(x.shape[0], -1).half()
        residual = residual + self.linear(attention, prefix + "self_attn.o_proj.weight")
        x = self.norm(residual, prefix + "post_attention_layernorm.weight")
        gate = self.linear(x, prefix + "mlp.gate_proj.weight")
        up = self.linear(x, prefix + "mlp.up_proj.weight")
        activated = (F.silu(gate.float()) * up.float()).half()
        return self.linear(activated, prefix + "mlp.down_proj.weight"), residual


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", default="0,1,2,31,63")
    parser.add_argument("--lm-head", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    reference = Reference(args.checkpoint, torch.load(args.embeddings, weights_only=True), args.device)
    trace = torch.load(args.trace, weights_only=True)["steps"]
    selected = set(range(len(trace))) if args.steps == "all" else {int(i) for i in args.steps.split(",")}
    lm_head = None
    if args.lm_head:
        with safe_open(str(args.lm_head), framework="pt") as weights:
            lm_head = weights.get_tensor("lm_head.weight").to(device=args.device, dtype=torch.float16)
    committed = {}
    report = {"device": args.device, "checkpoint": str(args.checkpoint), "steps": []}
    for index, step in enumerate(trace):
        for position, feature in zip(step["context"][1].tolist(), step["context"][0]):
            committed[position] = feature
        if index not in selected:
            continue
        query_positions = step["query_inputs"]["positions"].to(args.device)
        context_length = int(query_positions[0])
        context_positions = torch.arange(context_length, device=args.device)
        context = torch.stack([committed[i] for i in range(context_length)]).to(args.device)
        context = reference.norm(context, "hidden_norm.weight")
        embeddings = torch.stack([reference.rows[i] for i in step["query_inputs"]["input_ids"].tolist()])
        projected = reference.linear(step["aux"].to(args.device), "fc.weight")
        record = {
            "step": index,
            "context_length": context_length,
            "projected": difference(projected, step["projected"]),
            "layers": [],
        }
        hidden, residual = embeddings, None
        for layer in range(reference.config["num_hidden_layers"]):
            expected = step[f"layer_{layer}"]
            teacher_hidden, teacher_residual = (
                (embeddings, None) if layer == 0 else tuple(v.to(args.device) for v in step[f"layer_{layer - 1}"])
            )
            local = reference.layer(
                layer, teacher_hidden, teacher_residual, context, context_positions, query_positions
            )
            hidden, residual = reference.layer(layer, hidden, residual, context, context_positions, query_positions)
            record["layers"].append(
                {
                    "layer": layer,
                    "local_mlp": difference(local[0], expected[0]),
                    "local_attention_residual": difference(local[1], expected[1]),
                    "propagated": difference(hidden + residual, expected[0] + expected[1]),
                }
            )
        final = reference.norm(hidden + residual, "norm.weight")
        record["final"] = difference(final, step["draft_hidden"])
        if lm_head is not None:
            anchor = step["next_token_ids"].item()
            expected = step["proposed"].flatten().tolist()
            record["proposals"] = {
                "ascend": expected,
                "cuda_head_on_ascend_hidden": reference.propose(step["draft_hidden"].to(args.device), anchor, lm_head),
                "cuda_reference": reference.propose(final, anchor, lm_head),
            }
            record["confidence_on_ascend_proposals"] = reference.confidence(
                step["draft_hidden"].to(args.device), anchor, expected
            )
        report["steps"].append(record)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    report["summary"] = summarize(trace, report["steps"])
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
