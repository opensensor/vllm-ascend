# SPDX-License-Identifier: Apache-2.0
"""Execute the real config factory with isolated, strict CPU-only dependencies."""

import ast
import copy
import dataclasses
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pydantic import ConfigDict, TypeAdapter, ValidationError
from pydantic.dataclasses import dataclass as config_dataclass


@pytest.fixture
def factory(monkeypatch):
    events = []

    @config_dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
    class StrictAscendConfig:
        scheduler_config: Any
        sparse_kv_offload_config: Any
        kvpp_config: Any
        dump_config_path: Any
        enable_cpu_binding: bool = False
        ascend_compilation_config: Any = None
        eplb_config: Any = None
        rl_config: Any = dataclasses.field(
            default_factory=lambda: SimpleNamespace(apply=lambda config: events.append("rl-apply"))
        )
        finegrained_tp_config: Any = dataclasses.field(
            default_factory=lambda: SimpleNamespace(_validate_preconditions=lambda config: events.append("tp-validate"))
        )
        xlite_graph_config: Any = dataclasses.field(
            default_factory=lambda: SimpleNamespace(
                _validate_preconditions=lambda config: events.append("graph-validate")
            )
        )

        @staticmethod
        def _resolve_dump_config_path(additional):
            return additional.get("dump_config_path")

        def derive_and_validate(self, config):
            events.append("derive")

    utils = ModuleType("vllm_ascend.utils")
    utils.clear_enable_sp = lambda: events.append("clear-sp")
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", utils)
    source = Path(__file__).resolve().parents[3] / "vllm_ascend" / "ascend_config.py"
    parsed = ast.parse(source.read_text(), filename=str(source))
    names = {"validate_additional_config_bool", "_is_ascend_config_initialized", "init_ascend_config"}
    # Compile the current product functions unchanged, including the real
    # pass-through list, cache path and kwargs filtering. Inject only external
    # config/SDK dependencies: unknown kwargs still reach a real strict
    # pydantic dataclass rather than a permissive mock constructor.
    nodes = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
        or isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
    ]
    assert {node.name for node in nodes if isinstance(node, ast.FunctionDef)} == names
    scope = dict(
        TypeAdapter=TypeAdapter,
        AscendConfig=StrictAscendConfig,
        SchedulerConfig=SimpleNamespace(from_additional_config=lambda config: {"scheduler": "isolated"}),
        SparseKVOffloadConfig=SimpleNamespace(from_additional_config=lambda config, value: value),
        KVPPConfig=SimpleNamespace(from_vllm_config=lambda config: {"kvpp": "isolated"}),
        logger=SimpleNamespace(warning=lambda *args: events.append("warning"), warning_once=lambda *args: None),
        importlib=SimpleNamespace(util=SimpleNamespace(find_spec=lambda name: None)),
        _ASCEND_CONFIG=None,
        _INIT_VLLM_CONFIG=None,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return SimpleNamespace(init=scope["init_ascend_config"], scope=scope, events=events)


@pytest.mark.parametrize("payload", (None, {}, {"schema_version": 1, "native_library": "/private/pinned/library.so"}))
def test_recognized_communicator_config_is_preserved_without_forwarding_constructor_kw(factory, payload):
    original = {"npu_collective": payload, "enable_cpu_binding": True}
    before = copy.deepcopy(original)
    config = SimpleNamespace(additional_config=original)
    result = factory.init(config)
    assert result.enable_cpu_binding is True
    assert "npu_collective" not in result.__dict__
    assert config.additional_config is original and original == before
    assert config.additional_config["npu_collective"] is payload
    assert factory.scope["_ASCEND_CONFIG"] is result
    assert factory.scope["_INIT_VLLM_CONFIG"] is config
    assert factory.events == ["derive", "rl-apply", "tp-validate", "graph-validate", "clear-sp"]
    assert factory.init(config) is result


@pytest.mark.parametrize("additional", (None, {}))
def test_existing_default_config_initialization_remains_accepted(factory, additional):
    config = SimpleNamespace(additional_config=additional)
    result = factory.init(config)
    assert result.enable_cpu_binding is False
    assert config.additional_config is additional


@pytest.mark.parametrize("payload", (True, False, 1, 0, "enabled", [], ["mapping"], ()))
def test_non_mapping_communicator_value_is_rejected_before_constructor_or_caching(factory, payload):
    config = SimpleNamespace(additional_config={"npu_collective": payload})
    with pytest.raises(ValueError, match=r"additional_config\.npu_collective must be a dict or None"):
        factory.init(config)
    assert factory.scope["_ASCEND_CONFIG"] is None
    assert not factory.events
    assert config.additional_config["npu_collective"] is payload


def test_invalid_communicator_type_cannot_bypass_validation_through_cached_config(factory):
    config = SimpleNamespace(additional_config={"npu_collective": {}})
    result = factory.init(config)
    config.additional_config["npu_collective"] = False
    with pytest.raises(ValueError, match="npu_collective"):
        factory.init(config)
    assert factory.scope["_ASCEND_CONFIG"] is result


@pytest.mark.parametrize("unknown", ("unknown_option", "npu_collectiv", "npu_collective_enabled"))
def test_only_exact_pass_through_is_allowed_unknown_keys_still_reach_strict_constructor(factory, unknown):
    config = SimpleNamespace(additional_config={"npu_collective": {}, unknown: True})
    with pytest.raises(ValidationError) as failure:
        factory.init(config)
    assert any(
        error["type"] == "unexpected_keyword_argument" and error["loc"] == (unknown,)
        for error in failure.value.errors()
    )
    assert not factory.events
    assert factory.scope["_ASCEND_CONFIG"] is None
    assert config.additional_config == {"npu_collective": {}, unknown: True}


def test_nested_schema_and_eager_policy_remain_deferred_to_communicator(factory):
    payload = {"intentionally_invalid_nested_schema": "validated only at TP setup"}
    config = SimpleNamespace(
        additional_config={"npu_collective": payload}, model_config=SimpleNamespace(enforce_eager=False)
    )
    factory.init(config)
    assert config.additional_config["npu_collective"] is payload
    assert config.model_config.enforce_eager is False
