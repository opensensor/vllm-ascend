# SPDX-License-Identifier: Apache-2.0
"""Generate or launch an isolated, automatically profiled TP2 W4 service.

Source the CANN environment and install the matched Ascend plugin/operators
first. --show performs no NPU initialization. A generated command is a launch
recipe, not proof that its context, concurrency or operators fit on a card.
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

MODEL_CONTEXT_LIMIT = 262144
NATIVE_ROUTE_LIMIT = 25600
TOP_K = 10
MAX_BATCH_TOKENS = NATIVE_ROUTE_LIMIT // TOP_K
SINGLE_CARD_PREFIX_SLOTS = 16
PREVIEW_ENVIRONMENT_KEYS = (
    "ASCEND_RT_VISIBLE_DEVICES",
    "SOC_VERSION",
    "VLLM_ASCEND_ENABLE_310P",
    "VLLM_USE_V2_MODEL_RUNNER",
    "VLLM_ASCEND_KV_CACHE_FRACTION",
    "VLLM_SERVER_DEV_MODE",
    "VLLM_USE_BREAKABLE_CUDAGRAPH",
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_CUSTOM_OPP_PATH",
    "ATB_HOME_PATH",
    "HCCL_SOCKET_IFNAME",
    "HCCL_IF_IP",
    "HCCL_IF_BASE_PORT",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONPATH",
    "TE_PARALLEL_COMPILER",
)


def build_environment(profile: "SingleCardProfile", inherited):
    environment = dict(inherited, ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, profile.devices)))
    # Select explicitly instead of inheriting shell state. The frozen Qwen MTP
    # compatibility patch can still choose v1; verify the actual startup log.
    environment["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    if profile.resident_experiments:
        environment.update(VLLM_SERVER_DEV_MODE="1", VLLM_USE_BREAKABLE_CUDAGRAPH="1")
    return environment


@dataclass(frozen=True)
class SingleCardProfile:
    model: Path
    devices: tuple[int, int] = (0, 1)
    port: int = 8002
    context: int = 24576
    batch_tokens: int = 512
    sequences: int = 1
    speculative_tokens: int = 2
    memory_utilization: float = 0.965
    updated: bool = False
    quantize_dense_projections: bool = False
    quantize_lm_head: bool = False
    images: bool = False
    resident_experiments: bool = False
    prefix_slots: int = SINGLE_CARD_PREFIX_SLOTS
    parallel_gather: bool = False

    def validate(self):
        if (
            len(self.devices) != 2
            or any(type(x) is not int or x < 0 for x in self.devices)
            or self.devices[0] % 2
            or self.devices[1] != self.devices[0] + 1
        ):
            raise ValueError("select one consecutive dual-chip card pair, such as 0,1 or 2,3")
        for name, value, minimum, maximum in (
            ("port", self.port, 1024, 65535),
            ("context", self.context, 1, MODEL_CONTEXT_LIMIT),
            ("batch_tokens", self.batch_tokens, 1, MAX_BATCH_TOKENS),
            ("sequences", self.sequences, 1, 2),
            ("speculative_tokens", self.speculative_tokens, 0, 2),
            ("prefix_slots", self.prefix_slots, 1, 64),
        ):
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        if self.batch_tokens < self.sequences * (self.speculative_tokens + 1):
            raise ValueError("batch token budget must accommodate speculative verification")
        if not math.isfinite(self.memory_utilization) or not 0.8 <= self.memory_utilization <= 0.965:
            raise ValueError("memory utilization must be finite and between 0.8 and 0.965")
        if (self.quantize_dense_projections or self.quantize_lm_head) and not self.updated:
            raise ValueError("dense projection quantization requires the updated profile")
        if self.resident_experiments and not self.updated:
            raise ValueError("resident experiments require the updated profile")
        if self.prefix_slots != SINGLE_CARD_PREFIX_SLOTS and not self.updated:
            raise ValueError("custom prefix residency requires the updated profile")
        if type(self.parallel_gather) is not bool or (self.parallel_gather and not self.updated):
            raise ValueError("parallel QSA gather requires a boolean and the updated profile")
        if not self.updated and self.batch_tokens > 512:
            raise ValueError("historical baseline operator capacity is 512 tokens")
        return self

    @property
    def graph_sizes(self):
        self.validate()
        rows = self.speculative_tokens + 1
        return [rows] if self.sequences == 1 else [rows, rows * self.sequences]


def build_command(profile: SingleCardProfile, python: str = sys.executable):
    profile.validate()
    config = json.loads((profile.model / "config.json").read_text())
    metadata = dict(config.get("text_config", {}).get("ascend_expert_quantization", {}))
    if metadata.get("format") != "qwen4exp_w4a16_group_v1" or metadata.get("bits") != 4:
        raise ValueError("requires the original Qwen W4 G128 checkpoint contract")
    if metadata.get("group_size") != 128 or metadata.get("scale_dtype") != "float16":
        raise ValueError("requires G128 weights with FP16 scales")
    metadata.update(backend="cube_310_int4_a8", activation_quantization="int8_per_group")
    if profile.updated:
        metadata.update(
            grouped_activation="cann_builtin_fp16",
            grouped_finalize="cann_v2",
            grouped_prefill_chunk_tokens=profile.batch_tokens,
            shared_expert_execution="tp_sharded",
        )
    if profile.quantize_dense_projections:
        metadata.update(lm_head_execution="w8a8_dynamic", ple_projection_execution="w8a8_dynamic")
    elif profile.quantize_lm_head:
        metadata.update(lm_head_execution="w8a8_dynamic")
    text_overrides = {"ascend_expert_quantization": metadata}
    if profile.updated:
        text_overrides["ascend_qsa_prefill"] = {"parallel_gather": profile.parallel_gather}
    command = [python, "-m", "vllm.entrypoints.cli.main", "serve", str(profile.model)]
    options = {
        "--served-model-name": "qwen38-single-card",
        "--host": "127.0.0.1",
        "--port": str(profile.port),
        "--dtype": "float16",
        "--tensor-parallel-size": "2",
        "--max-model-len": str(profile.context),
        "--max-num-batched-tokens": str(profile.batch_tokens),
        "--max-num-seqs": str(profile.sequences),
        "--gpu-memory-utilization": str(profile.memory_utilization),
        "--mamba-cache-mode": "align",
        "--tool-call-parser": "qwen3_xml",
        "--reasoning-parser": "qwen3",
        "--compilation-config": json.dumps(
            {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": profile.graph_sizes}
        ),
        "--hf-overrides": json.dumps({"text_config": text_overrides}),
        "--limit-mm-per-prompt": json.dumps({"image": int(profile.images), "video": 0}),
    }
    for flag, value in options.items():
        command.extend((flag, value))
    command.extend(
        (
            "--no-async-scheduling",
            "--disable-custom-all-reduce",
            "--enable-expert-parallel",
            "--enable-ep-weight-filter",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--enable-prompt-tokens-details",
            "--enable-auto-tool-choice",
            "--cudagraph-metrics",
            "--enable-logging-iteration-details",
        )
    )
    if not profile.images:
        command.append("--language-model-only")
    if profile.speculative_tokens:
        command.extend(
            (
                "--speculative-config",
                json.dumps({"method": "mtp", "num_speculative_tokens": profile.speculative_tokens}),
            )
        )
    if profile.updated:
        command.extend(("--additional-config", json.dumps({"qwen_prefix_mamba_min_slots": profile.prefix_slots})))
        command.extend(("--scheduler-cls", "vllm_ascend.core.qwen_prefix_replay_scheduler.QwenPrefixReplayScheduler"))
    if profile.resident_experiments:
        command.extend(
            (
                "--worker-extension-cls",
                "tools.qwen4exp.prefix_npu_validation.QwenPrefixValidationExtension",
                "--middleware",
                "tools.glm_perf.resident_middleware.ResidentControlMiddleware",
            )
        )
    # Deliberately no TP4 fixed-cache value or profiling bypass. Both ranks
    # must pass the real automatic memory profile for this exact geometry.
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--port", default=8002, type=int)
    parser.add_argument("--context", default=24576, type=int)
    parser.add_argument("--batch-tokens", default=512, type=int)
    parser.add_argument("--sequences", default=1, type=int)
    parser.add_argument("--speculative-tokens", default=2, type=int)
    parser.add_argument("--memory-utilization", default=0.965, type=float)
    parser.add_argument("--updated", action="store_true")
    parser.add_argument("--quantize-dense-projections", action="store_true")
    parser.add_argument("--quantize-lm-head", action="store_true", help="W8 output head with unchanged PLE projection")
    parser.add_argument("--images", action="store_true")
    parser.add_argument("--resident-experiments", action="store_true")
    parser.add_argument("--prefix-slots", default=SINGLE_CARD_PREFIX_SLOTS, type=int)
    parser.add_argument("--parallel-gather", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    try:
        devices = tuple(int(x) for x in args.devices.split(","))
        profile = SingleCardProfile(
            args.model,
            devices=devices,
            port=args.port,
            context=args.context,
            batch_tokens=args.batch_tokens,
            sequences=args.sequences,
            speculative_tokens=args.speculative_tokens,
            memory_utilization=args.memory_utilization,
            updated=args.updated,
            quantize_dense_projections=args.quantize_dense_projections,
            quantize_lm_head=args.quantize_lm_head,
            images=args.images,
            resident_experiments=args.resident_experiments,
            prefix_slots=args.prefix_slots,
            parallel_gather=args.parallel_gather,
        )
        command = build_command(profile)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    environment = build_environment(profile, os.environ)
    if args.show:
        preview = {key: environment.get(key) for key in PREVIEW_ENVIRONMENT_KEYS}
        print(
            json.dumps(
                {"devices": environment["ASCEND_RT_VISIBLE_DEVICES"], "command": command, "environment": preview},
                indent=2,
            )
        )
    else:
        os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
