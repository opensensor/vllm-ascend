# SPDX-License-Identifier: Apache-2.0
"""Explicit two-chip/RTX placement and checkpoint metadata staging."""

import argparse
import json
import os
import shutil
from pathlib import Path

from tools.glm_perf.deployment_profiles import one_card_sidecard_profile, validate_profile

from .native_expert_probe import digest_file
from .placement import validate_glm_config

NPU_EXPERTS = 72
CUDA_EXPERTS = 144
NPU_DEVICES = (0, 1)
NATIVE_PARTITIONS = 4


def validate_device_pair(devices):
    if (
        type(devices) is not tuple
        or any(type(device) is not int for device in devices)
        or devices not in ((0, 1), (2, 3))
    ):
        raise ValueError("one-card sidecard requires exactly one physical NPU pair: 0,1 or 2,3")
    return devices


def validate_packed_gate(path, library_sha, source_manifest_sha):
    report = json.loads(Path(path).read_bytes())
    expected = {
        (layer, tokens, tokens * routes) for layer in (3, 8, 33) for tokens in (128, 512, 640) for routes in (1, 2, 8)
    }
    cases = report.get("cases", [])
    actual = {(case["layer"], case["tokens"], case["routes"]) for case in cases}
    if (
        report.get("passed") is not True
        or report.get("candidate_sha256") != library_sha
        or report.get("checkpoint_sha256") != source_manifest_sha
        or report.get("candidate_weight_layout") != "k32_nibbles_n_v1"
        or report.get("cuda_experts_per_layer") != CUDA_EXPERTS
        or report.get("candidate_partial_reduce") is not True
        or actual != expected
        or len(cases) != len(expected)
        or any(case.get("bitwise_equal") is not True for case in cases)
    ):
        raise ValueError("one-card service requires the complete pinned packed CUDA gate")
    return {
        "passed": True,
        "sha256": digest_file(path),
        "cases": len(cases),
        "cuda_experts_per_layer": CUDA_EXPERTS,
        "layers": [3, 8, 33],
        "library_sha256": library_sha,
    }


def npu_range(rank):
    if type(rank) is not int or rank not in NPU_DEVICES:
        raise ValueError("one-card sidecard requires NPU rank zero or one")
    start = CUDA_EXPERTS + rank * NPU_EXPERTS
    return start, start + NPU_EXPERTS


def stage_checkpoint(source, output):
    """Reuse authenticated tensor files; replace only copied metadata, never weights."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError("one-card checkpoint requires a fresh separate directory")
    manifest = json.loads((source / "native-layout.json").read_bytes())
    config = json.loads((source / "config.json").read_bytes())
    validate_glm_config(config)
    validate_profile(
        "two-card-selective-w2",
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        mtp_tokens=1,
        layer_bits=manifest["layer_bits"],
    )
    if (
        manifest.get("deployment_profile") != "two-card-selective-w2"
        or manifest.get("world_size") != NATIVE_PARTITIONS
        or manifest.get("num_experts") != CUDA_EXPERTS + len(NPU_DEVICES) * NPU_EXPERTS
        or manifest.get("complete") is not True
    ):
        raise ValueError("one-card staging requires the qualified four-partition checkpoint")
    source_sha = digest_file(source / "native-layout.json")
    shutil.copytree(source, output, copy_function=os.link)
    profile = one_card_sidecard_profile()
    config["ascend_glm_deployment_profile"] = profile.name
    if "text_config" in config:
        config["text_config"]["ascend_glm_deployment_profile"] = profile.name
    manifest.update(
        deployment_profile=profile.name,
        sidecard_source_manifest_sha256=source_sha,
        sidecard_cuda_experts=CUDA_EXPERTS,
        hardware_validation="not_run",
    )
    for name, value in (("config.json", config), ("native-layout.json", manifest)):
        temporary = output / (name + ".one-card.tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(output / name)
    return {
        "source_manifest_sha256": source_sha,
        "checkpoint_sha256": digest_file(output / "native-layout.json"),
        "checkpoint_root": str(output),
        "npu_workers": len(NPU_DEVICES),
        "hardware_validation": "not_run",
    }


def prepare_banks(model, rank):
    """Change lightweight holders before the native loader allocates any banks."""
    parallel = model.vllm_config.parallel_config
    if (parallel.tensor_parallel_size, parallel.pipeline_parallel_size) != (2, 1):
        raise ValueError("one-card native placement requires TP2/PP1")
    if model.vllm_config.speculative_config is not None or not model.vllm_config.model_config.enforce_eager:
        raise ValueError("one-card qualification requires eager target-only execution")
    start, stop = npu_range(rank)
    owners = [
        (name, module) for name, module in model.named_modules() if getattr(module, "w2_experts", None) is not None
    ]
    layers = [int(name.split(".")[name.split(".").index("layers") + 1]) for name, _ in owners]
    if sorted(layers) != list(range(3, 45)):
        raise ValueError("one-card native placement requires all 42 main MoE layers")
    for _, module in owners:
        bank = module.w2_experts
        if bank.offload_to_cpu or bank.grouped_ready or getattr(bank, "gate_packed_bank", None) is not None:
            raise ValueError("one-card bank placement must precede resident allocations")
    for _, module in owners:
        bank = module.w2_experts
        bank.local_expert_offset = start
        bank.num_local_experts = stop - start
    return owners


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage_checkpoint(args.source, args.output)), flush=True)


if __name__ == "__main__":
    main()
