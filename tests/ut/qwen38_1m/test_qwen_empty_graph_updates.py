# SPDX-License-Identifier: Apache-2.0
"""Only fixed Qwen targets with an empty task registry skip update events."""

import ast
import importlib.util
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("qwen_empty_task_helpers", ROOT / "vllm_ascend/_310p/qwen4exp_mtp.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def config(architecture="Qwen4ExpForConditionalGeneration", asynchronous=False, dcp=1):
    return SimpleNamespace(
        model_config=SimpleNamespace(architectures=[architecture]),
        speculative_config=SimpleNamespace(method="mtp"),
        scheduler_config=SimpleNamespace(async_scheduling=asynchronous),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
        additional_config={},
    )


def registry():
    return SimpleNamespace(handles={3: [], 18: []}, attn_params={3: [], 18: []})


@pytest.mark.parametrize(
    "reason", ["eligible", "draft", "model", "async", "dcp", "method", "handles", "params", "keys", "empty", "missing"]
)
def test_no_task_update_gate_requires_all_contracts(reason):
    cfg, params = config(), registry()
    if reason == "model":
        cfg.model_config.architectures = ["OtherModel"]
    elif reason == "async":
        cfg.scheduler_config.async_scheduling = True
    elif reason == "dcp":
        cfg.parallel_config.decode_context_parallel_size = 2
    elif reason == "method":
        cfg.speculative_config.method = "other"
    elif reason == "handles":
        params.handles[18].append(object())
    elif reason == "params":
        params.attn_params[3].append(object())
    elif reason == "keys":
        del params.handles[18]
    elif reason == "empty":
        params.handles.clear()
        params.attn_params.clear()
    elif reason == "missing":
        params = None
    assert helpers.qwen4exp_target_has_no_task_updates(cfg, reason == "draft", params) == (reason == "eligible")


def load_method(path, cls, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
    method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "mode,capturing,enpu,mutable",
    [
        ("full", False, False, False),
        ("full", False, False, True),
        ("full", True, False, False),
        ("none", False, False, False),
        ("full", False, True, False),
    ],
)
def test_runner_skips_only_empty_update_stream_round_trip(mode, capturing, enpu, mutable):
    calls = []
    cfg, params = config(), registry()
    if mutable:
        params.handles[3].append(object())
    context = SimpleNamespace(cudagraph_runtime_mode=mode, capturing=capturing)
    main = SimpleNamespace(wait_stream=lambda stream: calls.append("main-wait"))
    update = SimpleNamespace(wait_stream=lambda stream: calls.append("update-wait"))
    namespace = {
        "partial": partial,
        "torch": SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: main)),
        "get_forward_context": lambda: context,
        "CUDAGraphMode": SimpleNamespace(FULL="full"),
        "get_graph_params": lambda: params,
        "qwen4exp_target_has_no_task_updates": helpers.qwen4exp_target_has_no_task_updates,
    }
    method = load_method("vllm_ascend/_310p/model_runner_310p.py", "NPUModelRunner310", "_model_forward", namespace)
    output = object()

    def model(**kwargs):
        calls.append("model")
        assert kwargs["input_ids_cpu"] == [0, 1, 2, 3]
        return output

    runner = SimpleNamespace(
        uses_mrope=False,
        _qwen4exp_mtp_ple=True,
        model=model,
        input_ids=SimpleNamespace(cpu=list(range(8))),
        _ple_query_start_loc_cpu=None,
        _ple_context_cpu=None,
        enable_enpu=enpu,
        speculative_config=cfg.speculative_config,
        vllm_config=cfg,
        update_stream=update,
        _stage_qwen4exp_ple_inputs=lambda rows: (None, None),
        _update_full_graph_params_if_needed=lambda context, rows: calls.append("update"),
    )
    assert method(runner, 4) is output
    if enpu:
        expected = ["update", "model"]
    elif mode == "full" and not capturing:
        expected = ["update-wait", "update", "main-wait", "model"] if mutable else ["model"]
    else:
        expected = ["model", "update"]
    assert calls == expected


@pytest.mark.parametrize("hardware_310p", [False, True])
@pytest.mark.parametrize("event_present", [False, True])
@pytest.mark.parametrize("mutable", [False, True])
def test_target_replay_keeps_host_boundary_when_task_registry_is_populated(event_present, mutable, hardware_310p):
    calls = []
    params = registry()
    if mutable:
        params.attn_params[18].append(object())
    context = SimpleNamespace(cudagraph_runtime_mode="full")
    if event_present:
        context._ascend_replay_ready_event = SimpleNamespace(synchronize=lambda: calls.append("host-event"))
    stream = SimpleNamespace(
        synchronize=lambda: calls.append("host-stream"), wait_event=lambda event: calls.append("stream-event")
    )

    class Parent:
        def _replay(self, entry, args, kwargs):
            calls.append("replay")

    source = "vllm_ascend/compilation/breakable_aclgraph.py"
    tree = ast.parse((ROOT / source).read_text())
    klass = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BreakableACLGraphWrapper"
    )
    replay = next(node for node in klass.body if getattr(node, "name", None) == "_replay")
    namespace = {
        "Parent": Parent,
        "torch": SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: stream)),
        "get_forward_context": lambda: context,
        "CUDAGraphMode": SimpleNamespace(FULL="full"),
        "_EXTRA_CTX": SimpleNamespace(is_draft_model=False),
        "get_graph_params": lambda: params,
        "is_310p": lambda: hardware_310p,
        "qwen4exp_draft_supports_stream_ordering": helpers.qwen4exp_draft_supports_stream_ordering,
        "qwen4exp_target_has_no_task_updates": helpers.qwen4exp_target_has_no_task_updates,
    }
    wrapper = ast.ClassDef(
        name="Wrapper", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=[replay], decorator_list=[]
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(wrapper)
    exec(compile(ast.fix_missing_locations(module), source, "exec"), namespace)
    instance = namespace["Wrapper"]()
    instance.vllm_config = config()
    instance.use_eagle = instance.enable_enpu = False
    output = object()
    assert instance._replay(SimpleNamespace(output=output), (), {}) is output
    expected = []
    if mutable or not hardware_310p:
        expected.append("host-event" if event_present else "host-stream")
    elif event_present:
        expected.append("stream-event")
    assert calls == expected + ["replay"]


@pytest.mark.parametrize("field", ["handles", "attn_params"])
def test_unknown_registry_values_are_rejected_without_tensor_truth_testing(field):
    class UnknownTaskCollection:
        def __bool__(self):
            pytest.fail("unknown task collections must not be inspected through truth conversion")

    params = registry()
    getattr(params, field)[18] = UnknownTaskCollection()
    assert not helpers.qwen4exp_target_has_no_task_updates(config(), False, params)
