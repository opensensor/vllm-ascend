# SPDX-License-Identifier: Apache-2.0
import json
from dataclasses import replace

import pytest

from tools.qwen4exp.single_card_profile import SingleCardProfile, build_command, build_environment


@pytest.fixture
def profile(tmp_path):
    model = tmp_path / "checkpoint with spaces"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "ascend_expert_quantization": {
                        "format": "qwen4exp_w4a16_group_v1",
                        "bits": 4,
                        "group_size": 128,
                        "scale_dtype": "float16",
                        "backend": "eager_dequant",
                        "symmetric": False,
                    }
                }
            }
        )
    )
    return SingleCardProfile(model)


def option(command, name):
    return command[command.index(name) + 1]


def test_graph_runner_is_explicit_and_library_paths_are_preserved(profile):
    inherited = {"VLLM_USE_V2_MODEL_RUNNER": "0", "LD_LIBRARY_PATH": "/matched/libraries"}
    environment = build_environment(profile, inherited)
    assert environment["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert environment["LD_LIBRARY_PATH"] == "/matched/libraries"
    assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "0,1"
    assert inherited["VLLM_USE_V2_MODEL_RUNNER"] == "0"


def test_single_card_command_keeps_real_weights_and_profiles_memory(profile):
    command = build_command(profile, python="python with spaces")
    assert command[:5] == ["python with spaces", "-m", "vllm.entrypoints.cli.main", "serve", str(profile.model)]
    assert option(command, "--tensor-parallel-size") == "2"
    assert option(command, "--max-model-len") == "24576"
    assert "--enable-ep-weight-filter" in command
    assert "--kv-cache-memory" not in command
    assert "--load-format" not in command
    assert json.loads(option(command, "--speculative-config")) == {"method": "mtp", "num_speculative_tokens": 2}
    config = json.loads(option(command, "--compilation-config"))
    assert config["cudagraph_capture_sizes"] == [3]
    assert config["cudagraph_mode"] == "FULL_DECODE_ONLY"


@pytest.mark.parametrize("mtp, expected", [(0, [1, 2]), (1, [2, 4]), (2, [3, 6])])
def test_two_sequences_have_exact_decode_and_verify_graphs(profile, mtp, expected):
    candidate = replace(profile, sequences=2, speculative_tokens=mtp)
    command = build_command(candidate)
    assert candidate.graph_sizes == expected
    assert ("--speculative-config" in command) == (mtp != 0)


def test_updated_fusions_preserve_w4_contract_and_dense_precision(profile):
    command = build_command(replace(profile, updated=True, batch_tokens=1024))
    metadata = json.loads(option(command, "--hf-overrides"))["text_config"]["ascend_expert_quantization"]
    assert metadata["bits"] == 4 and metadata["group_size"] == 128
    assert metadata["grouped_activation"] == "cann_builtin_fp16"
    assert metadata["grouped_finalize"] == "cann_v2"
    assert metadata["grouped_prefill_chunk_tokens"] == 1024
    assert "lm_head_execution" not in metadata
    assert "ple_projection_execution" not in metadata
    assert json.loads(option(command, "--additional-config")) == {"qwen_prefix_mamba_min_slots": 16}
    assert (
        option(command, "--scheduler-cls") == "vllm_ascend.core.qwen_prefix_replay_scheduler.QwenPrefixReplayScheduler"
    )


def test_head_only_quantization_leaves_ple_and_expert_precision_unchanged(profile):
    command = build_command(replace(profile, updated=True, quantize_lm_head=True))
    metadata = json.loads(option(command, "--hf-overrides"))["text_config"]["ascend_expert_quantization"]
    assert metadata["lm_head_execution"] == "w8a8_dynamic"
    assert "ple_projection_execution" not in metadata
    assert metadata["bits"] == 4 and metadata["group_size"] == 128 and metadata["scale_dtype"] == "float16"
    with pytest.raises(ValueError, match="requires the updated"):
        build_command(replace(profile, quantize_lm_head=True))


def test_dense_quantization_is_explicit_and_images_are_optional(profile):
    candidate = replace(profile, updated=True, quantize_dense_projections=True, images=True)
    command = build_command(candidate)
    metadata = json.loads(option(command, "--hf-overrides"))["text_config"]["ascend_expert_quantization"]
    assert metadata["lm_head_execution"] == metadata["ple_projection_execution"] == "w8a8_dynamic"
    assert "--language-model-only" not in command
    assert json.loads(option(command, "--limit-mm-per-prompt")) == {"image": 1, "video": 0}


@pytest.mark.parametrize(
    "change",
    [
        {"devices": (0, 2)},
        {"devices": (1, 2)},
        {"devices": (0, 1, 2, 3)},
        {"devices": (-2, -1)},
        {"context": 0},
        {"context": 1048576},
        {"context": True},
        {"batch_tokens": 2561},
        {"batch_tokens": 1024},
        {"batch_tokens": 2},
        {"port": 65536},
        {"sequences": 4},
        {"speculative_tokens": 3},
        {"memory_utilization": float("nan")},
        {"memory_utilization": 0.99},
        {"quantize_dense_projections": True},
        {"resident_experiments": True},
        {"prefix_slots": 8},
        {"prefix_slots": 0},
        {"prefix_slots": True},
        {"prefix_slots": 65},
        {"parallel_gather": True},
        {"parallel_gather": "false"},
    ],
)
def test_unsafe_or_unqualified_single_card_profiles_are_rejected(profile, change):
    with pytest.raises(ValueError):
        build_command(replace(profile, **change))


def test_new_quant_checkpoint_cannot_silently_use_w4_profile(profile):
    (profile.model / "config.json").write_text(json.dumps({"text_config": {"ascend_expert_quantization": {"bits": 8}}}))
    with pytest.raises(ValueError, match="original Qwen W4"):
        build_command(profile)


def test_resident_controls_are_opt_in_and_loopback_only(profile):
    assert "--worker-extension-cls" not in build_command(replace(profile, updated=True))
    command = build_command(replace(profile, updated=True, resident_experiments=True))
    assert option(command, "--host") == "127.0.0.1"
    assert (
        option(command, "--worker-extension-cls")
        == "tools.qwen4exp.prefix_npu_validation.QwenPrefixValidationExtension"
    )
    assert option(command, "--middleware") == "tools.glm_perf.resident_middleware.ResidentControlMiddleware"


def test_context_candidate_can_trade_checkpoint_residency_for_attention_cache(profile):
    command = build_command(replace(profile, updated=True, prefix_slots=7, context=131072))
    assert option(command, "--max-model-len") == "131072"
    assert json.loads(option(command, "--additional-config")) == {"qwen_prefix_mamba_min_slots": 7}
    assert "--language-model-only" in command
    assert json.loads(option(command, "--limit-mm-per-prompt")) == {"image": 0, "video": 0}


def test_updated_qsa_gathers_are_sequential_unless_explicitly_selected(profile):
    baseline = json.loads(option(build_command(profile), "--hf-overrides"))["text_config"]
    assert "ascend_qsa_prefill" not in baseline
    updated = replace(profile, updated=True)
    config = json.loads(option(build_command(updated), "--hf-overrides"))["text_config"]
    assert config["ascend_qsa_prefill"] == {"parallel_gather": False}
    config = json.loads(option(build_command(replace(updated, parallel_gather=True)), "--hf-overrides"))["text_config"]
    assert config["ascend_qsa_prefill"] == {"parallel_gather": True}
