# SPDX-License-Identifier: Apache-2.0
"""Fixed Qwen draft replay needs stream dependencies, not host barriers."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
helper_spec = importlib.util.spec_from_file_location(
    "qwen_mtp_ordering_helpers", ROOT / "vllm_ascend/_310p/qwen4exp_mtp.py"
)
helpers = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helpers)
SOURCE = ROOT / "vllm_ascend/compilation/breakable_aclgraph.py"
tree = ast.parse(SOURCE.read_text())
klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BreakableACLGraphWrapper")
method = next(node for node in klass.body if getattr(node, "name", None) == "_replay")


@pytest.mark.parametrize("hardware_310p", [False, True])
@pytest.mark.parametrize("mode", ["full", "none"])
@pytest.mark.parametrize("event_present", [False, True])
@pytest.mark.parametrize(
    "draft,architecture,async_scheduling,dcp,method_name",
    [
        (True, "Qwen4ExpMTP", False, 1, "mtp"),
        (False, "Qwen4ExpMTP", False, 1, "mtp"),
        (True, "OtherMTP", False, 1, "mtp"),
        (True, "Qwen4ExpMTP", True, 1, "mtp"),
        (True, "Qwen4ExpMTP", False, 2, "mtp"),
        (True, "Qwen4ExpMTP", False, 1, "draft_model"),
    ],
)
def test_production_replay_preserves_boundary_for_other_execution_paths(
    mode, event_present, draft, architecture, async_scheduling, dcp, method_name, hardware_310p
):
    calls = []
    stream = SimpleNamespace(
        synchronize=lambda: calls.append("host-stream-sync"), wait_event=lambda event: calls.append("stream-wait")
    )
    ready = SimpleNamespace(synchronize=lambda: calls.append("host-event-sync"))
    context = SimpleNamespace(cudagraph_runtime_mode=mode)
    if event_present:
        context._ascend_replay_ready_event = ready
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method_name,
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture])),
        ),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
    )

    class Parent:
        def _replay(self, entry, args, kwargs):
            assert args == ("input",) and kwargs == {"positions": "positions"}
            calls.append("replay")

    namespace = {
        "torch": SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: stream)),
        "get_forward_context": lambda: context,
        "CUDAGraphMode": SimpleNamespace(FULL="full"),
        "_EXTRA_CTX": SimpleNamespace(is_draft_model=draft),
        "qwen4exp_draft_supports_stream_ordering": helpers.qwen4exp_draft_supports_stream_ordering,
        "qwen4exp_target_has_no_task_updates": helpers.qwen4exp_target_has_no_task_updates,
        "get_graph_params": lambda: None,
        "is_310p": lambda: hardware_310p,
        "Parent": Parent,
    }
    wrapper_ast = ast.ClassDef(
        name="Wrapper", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(wrapper_ast)
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    wrapper = namespace["Wrapper"]()
    wrapper.vllm_config = config
    wrapper.use_eagle = False
    wrapper.enable_enpu = False
    output = object()
    assert wrapper._replay(SimpleNamespace(output=output), ("input",), {"positions": "positions"}) is output
    expected = []
    if mode == "full":
        fixed_draft = (
            hardware_310p
            and draft
            and architecture == "Qwen4ExpMTP"
            and not async_scheduling
            and dcp == 1
            and method_name == "mtp"
        )
        if fixed_draft:
            if event_present:
                expected.append("stream-wait")
        else:
            expected.append("host-event-sync" if event_present else "host-stream-sync")
    assert calls == expected + ["replay"]


def test_missing_configuration_keeps_established_completion_boundary():
    assert not helpers.qwen4exp_draft_supports_stream_ordering(SimpleNamespace(), True)
