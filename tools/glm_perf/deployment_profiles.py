# SPDX-License-Identifier: Apache-2.0
"""Explicit two/three dual-chip-card recipes; no device discovery or launches."""

import argparse
import json
from dataclasses import asdict, dataclass

W4_LAYERS = tuple(range(3, 8)) + (11, 15) + tuple(range(18, 33))
SELECTIVE_W3_LAYERS = (8, 9, 10, 12, 13, 14, 16, 17)
PREFILL_TOKENS = 640
QUALIFICATION_CONTEXT = 8192


@dataclass(frozen=True)
class GlmDeploymentProfile:
    name: str
    cards: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    default_expert_bits: int
    w3_layers: tuple[int, ...]
    mtp_tokens: int

    @property
    def chips(self):
        return self.cards * 2

    @property
    def expert_partition_size(self):
        # Experts use the stage's TP group; total workers is NOT this field.
        return self.tensor_parallel_size

    def launch_args(self, model_path, *, max_model_len=QUALIFICATION_CONTEXT, port=8001):
        if type(max_model_len) is not int or max_model_len < 1:
            raise ValueError("context limit must be a positive integer")
        args = [
            "vllm",
            "serve",
            str(model_path),
            "--served-model-name",
            self.name,
            "--load-format",
            "glm_native_int4",
            "--dtype",
            "float16",
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--pipeline-parallel-size",
            str(self.pipeline_parallel_size),
            "--max-model-len",
            str(max_model_len),
            "--max-num-batched-tokens",
            str(PREFILL_TOKENS),
            "--max-num-seqs",
            "4",
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
            "--enable-prompt-tokens-details",
            "--no-async-scheduling",
            "--mamba-cache-mode",
            "align",
            "--disable-custom-all-reduce",
            "--reasoning-parser",
            "glm47",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "poolside_v1",
            "--port",
            str(port),
        ]
        overrides = dict(
            ascend_glm_nz_packed_codes=True,
            ascend_glm_mhc_batched_round=True,
            ascend_glm_mhc_fp16_state=True,
            ascend_glm_mhc_ai_core_round=True,
            ascend_glm_kda_nz_grouped=True,
            ascend_glm_prefill_route_histogram=True,
            ascend_glm_native_mhc_post=True,
            ascend_glm_live_kpool_score=True,
            ascend_glm_decode_swiglu=True,
            ascend_glm_decode_combine=True,
            ascend_glm_kpool_state_block_size=32,
        )
        if self.mtp_tokens:
            overrides["ascend_glm_mtp_full_graph"] = True
            args += [
                "--speculative-config",
                json.dumps(
                    {
                        "method": "mtp",
                        "num_speculative_tokens": self.mtp_tokens,
                        "draft_load_config": {"load_format": "glm_native_int4"},
                    }
                ),
                "--compilation-config",
                json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [2, 8]}),
            ]
        else:
            # First PP qualification must exercise correctness before graph replay.
            args += ["--enforce-eager"]
        args += ["--hf-overrides", json.dumps(overrides)]
        return args

    def conversion_args(self, source, output):
        return [
            "python",
            "-m",
            "tools.glm_w2.convert_full",
            "--source-dir",
            str(source),
            "--out-dir",
            str(output),
            "--expert-bits",
            str(self.default_expert_bits),
            "--w4-layers",
            ",".join(map(str, W4_LAYERS)),
            "--w3-layers",
            ",".join(map(str, self.w3_layers)),
        ]


def profile(cards):
    if type(cards) is not int:
        raise ValueError("GLM profile requires 2 or 3 dual-chip cards")
    if cards == 2:
        return GlmDeploymentProfile("two-card-selective-w2", 2, 4, 1, 2, SELECTIVE_W3_LAYERS, 1)
    if cards == 3:
        return GlmDeploymentProfile("three-card-w3", 3, 2, 3, 3, (), 0)
    raise ValueError("GLM profile requires 2 or 3 dual-chip cards")


def validate_profile(name, *, tensor_parallel_size, pipeline_parallel_size, mtp_tokens, layer_bits=None):
    selected = next((profile(cards) for cards in (2, 3) if profile(cards).name == name), None)
    if selected is None:
        raise ValueError("unknown GLM deployment profile")
    if (tensor_parallel_size, pipeline_parallel_size, mtp_tokens) != (
        selected.tensor_parallel_size,
        selected.pipeline_parallel_size,
        selected.mtp_tokens,
    ):
        raise ValueError("runtime topology or MTP differs from checkpoint deployment profile")
    if layer_bits is not None:
        for layer, bits in layer_bits.items():
            expected = (
                3
                if int(layer) in selected.w3_layers
                else 4
                if int(layer) in W4_LAYERS
                else selected.default_expert_bits
            )
            if bits != expected:
                raise ValueError("checkpoint precision differs from deployment profile")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=int, choices=(2, 3), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", help="Higher-precision FP8 source, never a W2 checkpoint")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=QUALIFICATION_CONTEXT,
        help="Start short for qualification; raise only after memory admission",
    )
    args = parser.parse_args()
    selected = profile(args.cards)
    result = dict(
        asdict(selected),
        expert_partition_size=selected.expert_partition_size,
        launch_args=selected.launch_args(args.model, max_model_len=args.max_model_len),
        hardware_validation="pending",
    )
    if args.source:
        result["conversion_args"] = selected.conversion_args(args.source, args.model + "-canonical")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
