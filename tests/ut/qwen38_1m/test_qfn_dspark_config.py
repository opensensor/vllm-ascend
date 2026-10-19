# SPDX-License-Identifier: Apache-2.0

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from vllm_ascend._310p.qwen4exp_dspark_config import QFN_LEGACY_SPEC_ERROR, verify_qfn_config
from vllm_ascend._310p.qwen4exp_mtp import is_qwen4exp_speculative_target


def config():
    return NS(
        model_config=NS(
            architectures=["Qwen4ExpForConditionalGeneration"],
            enforce_eager=True,
            max_model_len=2048,
            multimodal_config=NS(language_model_only=True),
            hf_text_config=NS(hc_count=4, hidden_size=2560, vocab_size=248320),
        ),
        speculative_config=NS(
            method="dspark",
            enforce_eager=True,
            draft_tensor_parallel_size=1,
            num_speculative_tokens=3,
            draft_model_config=NS(
                hf_config=NS(
                    dflash_config={"target_layer_ids": [1, 12, 23, 34, 45]},
                    hidden_size=2560,
                    vocab_size=248320,
                    num_attention_heads=32,
                    num_key_value_heads=8,
                )
            ),
        ),
        parallel_config=NS(
            tensor_parallel_size=6,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=NS(async_scheduling=False, max_num_seqs=1),
        cache_config=NS(enable_prefix_caching=False, mamba_cache_mode="align"),
        additional_config={"qfn_dspark_eager_probe": True},
    )


def legacy(config):
    raise NotImplementedError(QFN_LEGACY_SPEC_ERROR)


def test_terminal_allowlist_only_and_no_temporary_config_mutation():
    cfg = config()
    before = deepcopy(cfg)
    called = []

    def original(received):
        assert received is cfg and received == before
        called.append(True)
        legacy(received)

    verify_qfn_config(original, cfg, is_310p=True)
    assert called == [True] and cfg == before


def test_upstream_no_prefix_cache_normalization_is_preserved():
    cfg = config()

    def original(received):
        received.cache_config.mamba_cache_mode = "none"
        legacy(received)

    verify_qfn_config(original, cfg, is_310p=True)
    assert cfg.cache_config.mamba_cache_mode == "none"


def test_upstream_cannot_expand_the_qualified_configuration():
    def original(received):
        received.scheduler_config.async_scheduling = True
        legacy(received)

    with pytest.raises(ValueError, match="synchronous scheduling"):
        verify_qfn_config(original, config(), is_310p=True)


@pytest.mark.parametrize("message", ["unsupported RoPE", "PLE requires PP1", "unknown upstream change"])
def test_preserves_other_upstream_failures(message):
    def original(config):
        raise NotImplementedError(message)

    with pytest.raises(NotImplementedError, match=message):
        verify_qfn_config(original, config(), is_310p=True)


@pytest.mark.parametrize("method,chip", [("mtp", True), ("dspark", False), (None, True)])
def test_other_paths_remain_upstream(method, chip):
    cfg = config()
    cfg.speculative_config.method = method
    with pytest.raises(NotImplementedError, match="native MTP"):
        verify_qfn_config(legacy, cfg, is_310p=chip)


@pytest.mark.parametrize(
    "path,value",
    [
        ("model_config.enforce_eager", False),
        ("speculative_config.enforce_eager", False),
        ("scheduler_config.async_scheduling", True),
        ("parallel_config.tensor_parallel_size", 4),
        ("speculative_config.draft_tensor_parallel_size", 6),
        ("parallel_config.pipeline_parallel_size", 2),
        ("parallel_config.decode_context_parallel_size", 2),
        ("parallel_config.prefill_context_parallel_size", 2),
        ("cache_config.enable_prefix_caching", True),
        ("cache_config.mamba_cache_mode", "all"),
        ("scheduler_config.max_num_seqs", 2),
        ("model_config.max_model_len", 8192),
        ("speculative_config.num_speculative_tokens", 8),
        ("model_config.multimodal_config.language_model_only", False),
        ("model_config.hf_text_config.hc_count", 2),
        ("speculative_config.draft_model_config.hf_config.vocab_size", 1024),
        ("speculative_config.draft_model_config.hf_config.dflash_config", {"target_layer_ids": [5, 15, 25, 35, 45]}),
        ("additional_config", {}),
    ],
)
def test_rejects_unqualified_pair_before_loading(path, value):
    cfg = config()
    obj = cfg
    for part in path.split(".")[:-1]:
        obj = getattr(obj, part)
    setattr(obj, path.split(".")[-1], value)
    with pytest.raises(ValueError):
        verify_qfn_config(legacy, cfg, is_310p=True)


def test_k7_is_admitted_only_under_existing_eager_probe_guards():
    cfg = config()
    cfg.speculative_config.num_speculative_tokens = 7
    verify_qfn_config(legacy, cfg, is_310p=True)
    cfg.model_config.enforce_eager = False
    with pytest.raises(ValueError):
        verify_qfn_config(legacy, cfg, is_310p=True)


def graph_config():
    cfg = config()
    cfg.additional_config = {"qfn_dspark_graph_probe": True}
    cfg.model_config.enforce_eager = False
    cfg.speculative_config.num_speculative_tokens = 2
    cfg.scheduler_config.max_num_seqs = 4
    cfg.scheduler_config.scheduler_cls = "vllm_ascend.core.qwen_prefill_scheduler.QwenPrefillPacedScheduler"
    cfg.cache_config.enable_prefix_caching = True
    cfg.compilation_config = NS(cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[3, 6, 12])
    return cfg


@pytest.mark.parametrize("requests", [1, 4])
def test_target_graph_probe_keeps_draft_eager(requests):
    cfg = graph_config()
    cfg.scheduler_config.max_num_seqs = requests
    verify_qfn_config(legacy, cfg, is_310p=True)


@pytest.mark.parametrize("requests", [1, 4])
def test_k7_graph_probe_requires_its_own_capture_geometry(requests):
    cfg = graph_config()
    cfg.speculative_config.num_speculative_tokens = 7
    cfg.scheduler_config.max_num_seqs = requests
    with pytest.raises(ValueError):
        verify_qfn_config(legacy, cfg, is_310p=True)
    cfg.compilation_config.cudagraph_capture_sizes = [8, 16, 32]
    verify_qfn_config(legacy, cfg, is_310p=True)
    cfg.speculative_config.num_speculative_tokens = 2
    with pytest.raises(ValueError):
        verify_qfn_config(legacy, cfg, is_310p=True)


@pytest.mark.parametrize(
    "path,value",
    [
        ("additional_config", {"qfn_dspark_graph_probe": True, "qfn_dspark_eager_probe": True}),
        ("model_config.enforce_eager", True),
        ("speculative_config.enforce_eager", False),
        ("speculative_config.num_speculative_tokens", 3),
        ("scheduler_config.max_num_seqs", 6),
        ("scheduler_config.async_scheduling", True),
        ("cache_config.enable_prefix_caching", False),
        ("cache_config.mamba_cache_mode", "none"),
        ("scheduler_config.scheduler_cls", "vllm.v1.core.sched.scheduler.Scheduler"),
        ("model_config.max_model_len", 16384),
        ("compilation_config.cudagraph_mode", "FULL"),
        ("compilation_config.cudagraph_capture_sizes", [3, 6, 12, 18]),
    ],
)
def test_target_graph_probe_rejects_unqualified_layouts(path, value):
    cfg = graph_config()
    obj, field = path.split(".") if "." in path else (None, path)
    setattr(getattr(cfg, obj) if obj else cfg, field, value)
    with pytest.raises(ValueError):
        verify_qfn_config(legacy, cfg, is_310p=True)


@pytest.mark.parametrize("method", ["mtp", "dspark"])
def test_v2_environment_cannot_override_required_target_state_runner(method):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/platform/patch_use_v2_model_runner.py"
    tree = ast.parse(path.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_needs_310p_qwen4exp_speculative_v1", "_patched_use_v2_model_runner"}
    ]
    scope = {
        "is_310p": lambda: True,
        "is_qwen4exp_speculative_target": is_qwen4exp_speculative_target,
        "envs": NS(VLLM_USE_V2_MODEL_RUNNER=True),
        "logger": NS(warning_once=lambda *a: None),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    cfg = config()
    cfg.speculative_config.method = method
    assert scope["_patched_use_v2_model_runner"](cfg) is False
