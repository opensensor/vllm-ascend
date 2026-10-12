# SPDX-License-Identifier: Apache-2.0
"""Evaluate exact cold-expert dependencies from observed real-model routes.

Prefill chooses the resident set before decode. Hindsight is reported separately
as an optimistic bound. This study never drops, reroutes or renormalizes routes,
and its provisional weight ledger does not qualify a deployment or token rate.
"""

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

GIB = 1024**3
EXPERTS = 288
TOP_K = 8
FIRST_LAYER = 18
END_LAYER = 33
COMPONENT_GROUPS = {
    "mhc": ("deferred_mhc_post", "attention_mhc_pre", "attention_mhc_post", "ffn_mhc_pre"),
    "attention": ("attention",),
    "collectives": ("attention_sum", "mlp_sum"),
    "routing": ("router",),
    "routed_experts": ("routed_experts",),
    "shared_mlp": ("shared_mlp",),
}


def validate_packets(packets):
    sequences = set()
    for packet in packets:
        sequence, rows = packet["sequence"], packet["rows"]
        if (
            type(sequence) is not int
            or sequence < 0
            or sequence in sequences
            or type(rows) is not int
            or not 1 <= rows <= 32
            or type(packet["position"]) is not int
            or packet["position"] < 0
        ):
            raise ValueError("invalid or duplicate packet identity")
        sequences.add(sequence)
        if [row["layer"] for row in packet["layers"]] != list(range(FIRST_LAYER, END_LAYER)):
            raise ValueError("incomplete middle layer observation")
        for layer in packet["layers"]:
            if len(layer["expert_ids"]) != rows or len(layer["route_weights"]) != rows:
                raise ValueError("route rows disagree with packet")
            for ids, weights in zip(layer["expert_ids"], layer["route_weights"], strict=True):
                if (
                    len(ids) != TOP_K
                    or len(set(ids)) != TOP_K
                    or any(type(expert) is not int or not 0 <= expert < EXPERTS for expert in ids)
                    or len(weights) != TOP_K
                    or any(not math.isfinite(weight) or weight < 0 for weight in weights)
                ):
                    raise ValueError("invalid exact top-eight route observation")
            if any(not math.isfinite(value) or value < 0 for value in layer["components_ms"].values()):
                raise ValueError("invalid component timing")


def select_hot(packets, count):
    if type(count) is not int or not 0 <= count <= EXPERTS:
        raise ValueError("resident expert count exceeds layer geometry")
    hits = {index: Counter() for index in range(FIRST_LAYER, END_LAYER)}
    for packet in packets:
        for layer in packet["layers"]:
            for ids, weights in zip(layer["expert_ids"], layer["route_weights"], strict=True):
                hits[layer["layer"]].update(expert for expert, weight in zip(ids, weights, strict=True) if weight > 0)
    return {
        index: set(sorted(range(EXPERTS), key=lambda expert: (-counts[expert], expert))[:count])
        for index, counts in hits.items()
    }


def dependencies(packets, hot):
    waits, routes, owners = [], [], []
    for packet in packets:
        for row in range(packet["rows"]):
            row_waits = row_routes = row_owners = 0
            for layer in packet["layers"]:
                cold = [
                    expert
                    for expert, weight in zip(layer["expert_ids"][row], layer["route_weights"][row], strict=True)
                    if weight > 0 and expert not in hot[layer["layer"]]
                ]
                row_waits += bool(cold)
                row_routes += len(cold)
                row_owners += len({expert // (EXPERTS // 2) for expert in cold})
            waits.append(row_waits)
            routes.append(row_routes)
            owners.append(row_owners)
    if not waits:
        raise ValueError("decode evaluation needs observed rows")
    return dict(
        evaluated_tokens=len(waits),
        cold_route_fraction=sum(routes) / (len(routes) * (END_LAYER - FIRST_LAYER) * TOP_K),
        mean_waiting_layers=statistics.mean(waits),
        max_waiting_layers=max(waits),
        tokens_with_any_cold_expert=sum(value > 0 for value in waits) / len(waits),
        mean_chip_requests=statistics.mean(owners),
        waiting_layer_histogram=dict(sorted(Counter(waits).items())),
    )


def component_summary(packets):
    return {
        group: statistics.median(
            sum(layer["components_ms"].get(name, 0) for layer in packet["layers"] for name in names)
            for packet in packets
        )
        for group, names in COMPONENT_GROUPS.items()
    }


def study(packets, run, inventory, weight_budget_bytes=85 * GIB):
    if type(weight_budget_bytes) is not int or weight_budget_bytes <= 0:
        raise ValueError("weight budget must be a positive integer byte count")
    validate_packets(packets)
    layers = inventory["layers"]
    middle = layers[FIRST_LAYER:END_LAYER]
    expert_sizes = {row["expert_bytes"] // EXPERTS for row in middle}
    if len(expert_sizes) != 1 or any(row["bits"] != 4 for row in middle):
        raise ValueError("this study requires the uniform W4 middle interval")
    expert_bytes = expert_sizes.pop()
    existing_gpu = inventory["main_weights_bytes"] - sum(row["expert_bytes"] + row["dense_bytes"] for row in middle)
    backbone_gpu = existing_gpu + sum(row["dense_bytes"] for row in middle)
    admitted = max(0, min(EXPERTS, (weight_budget_bytes - backbone_gpu) // (expert_bytes * len(middle))))
    report = dict(
        production_speed_validated=False,
        scope="15 observed middle MoE layers; retain all other current GPU expert banks",
        ledger=dict(
            provisional=True,
            weight_budget_bytes=weight_budget_bytes,
            existing_gpu_weight_bytes=existing_gpu,
            backbone_gpu_weight_bytes=backbone_gpu,
            additional_expert_bytes=expert_bytes,
            uniform_hot_experts_per_layer=admitted,
            complete_gpu_weight_bytes=backbone_gpu + admitted * expert_bytes * len(middle),
            runtime_cache_workspace_reserve_hardware_qualified=False,
        ),
        workloads=[],
    )
    for request in run["workloads"]:
        observed = [p for p in packets if request["first_sequence"] <= p["sequence"] < request["end_sequence"]]
        prefill = [p for p in observed if p["position"] < request["prompt_tokens"]]
        decode = [p for p in observed if p["position"] >= request["prompt_tokens"]]
        if not prefill or not decode:
            raise ValueError("request has no causal prefill or decode observations")
        policies = []
        for count in sorted({0, admitted, 32, 64, 96, 144, 192, 256, EXPERTS}):
            bytes_required = backbone_gpu + count * expert_bytes * len(middle)
            policies.append(
                dict(
                    hot_experts_per_layer=count,
                    gpu_weight_bytes=bytes_required,
                    fits_provisional_weight_budget=bytes_required <= weight_budget_bytes,
                    prefill_trained=dependencies(decode, select_hot(prefill, count)),
                    decode_hindsight_bound=dependencies(decode, select_hot(decode, count)),
                )
            )
        report["workloads"].append(
            dict(
                name=request["name"],
                prompt_tokens=request["prompt_tokens"],
                profiled_component_median_ms=component_summary(decode),
                policies=policies,
            )
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packets = [json.loads(line) for line in args.trace.read_text().splitlines()]
    result = study(packets, json.loads(args.run.read_bytes()), json.loads(args.design.read_bytes())["inventory"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
