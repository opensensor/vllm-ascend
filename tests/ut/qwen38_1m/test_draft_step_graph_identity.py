# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU replay regression for the per-model graphs used by Qwen MTP."""

import ast
import importlib.util
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

WRAPPER_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/compilation/breakable_aclgraph.py"


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False
    num_active_loras: int = 0


def _load_wrapper(context_holder, extra_context):
    """Execute the production wrapper with a CPU graph-address oracle."""

    class GraphWrapper:
        def __init__(self, runnable, vllm_config):
            self.entries = {}
            self.runnable = runnable
            self.runtime_mode = None

        def _capture(self, entry, args, kwargs):
            # A device graph retains operand addresses captured from metadata,
            # even though QSA's eager indexer reads the current Python context.
            slots = context_holder[0].attn_metadata["qsa"].slot_mapping
            entry.capture = True
            entry.replay = lambda: self.runnable(slots, *args, **kwargs)
            entry.output = entry.replay()
            return entry.output

        def _replay(self, entry, args, kwargs):
            entry.output = entry.replay()
            return entry.output

        def clear_graphs(self):
            self.entries.clear()

    tree = ast.parse(WRAPPER_PATH.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if getattr(node, "name", None) in ("_DraftStepBatchDescriptor", "BreakableACLGraphWrapper")
    ]
    namespace = {
        "__name__": __name__,
        "Any": Any,
        "Callable": object,
        "VllmConfig": object,
        "dataclass": dataclass,
        "fields": fields,
        "BatchDescriptor": Descriptor,
        "BreakableCUDAGraphWrapper": GraphWrapper,
        "get_forward_context": lambda: context_holder[0],
        "is_forward_context_available": lambda: context_holder[0] is not None,
        "_EXTRA_CTX": extra_context,
        "CUDAGraphMode": SimpleNamespace(FULL="full", NONE="none"),
        "_BreakableEntry": lambda **kwargs: SimpleNamespace(capture=None, output=None, **kwargs),
        "weak_ref_workspaces": lambda params: None,
        "get_graph_params": lambda: None,
        "get_draft_graph_params": lambda: None,
        "get_draft_graph_prefill_params": lambda: None,
    }
    # Use upstream's actual entry dispatch, rather than a reconstruction of
    # its key/capture/replay rules. Device capture alone is replaced above.
    upstream_path = Path(importlib.util.find_spec("vllm").origin).parent / "compilation/breakable_cudagraph.py"
    upstream_tree = ast.parse(upstream_path.read_text(encoding="utf-8"))
    upstream_class = next(
        node for node in upstream_tree.body if getattr(node, "name", None) == "BreakableCUDAGraphWrapper"
    )
    dispatch = next(node for node in upstream_class.body if getattr(node, "name", None) == "__call__")
    exec(compile(ast.Module(body=[dispatch], type_ignores=[]), str(upstream_path), "exec"), namespace)
    GraphWrapper.__call__ = namespace["__call__"]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(WRAPPER_PATH), "exec"), namespace)
    return namespace["BreakableACLGraphWrapper"]


@pytest.mark.parametrize("steps_count", [2, 3])
@pytest.mark.parametrize("runtime_mode", ["full", "piecewise"])
def test_draft_graphs_retain_each_steps_slot_addresses_across_proposals(steps_count, runtime_mode):
    descriptor = Descriptor(18, 6, True, True, 2)
    slots = [torch.tensor([5 + step]) for step in range(steps_count)]
    maps = [{"qsa": SimpleNamespace(slot_mapping=slot)} for slot in slots]
    context_holder = [
        SimpleNamespace(
            attn_metadata=maps[0],
            draft_attn_metadatas=maps,
            batch_descriptor=descriptor,
            cudagraph_runtime_mode=runtime_mode,
        )
    ]
    wrapper_type = _load_wrapper(context_holder, SimpleNamespace(is_draft_model=True))
    cache = torch.zeros(32, dtype=torch.int64)
    value = torch.zeros(1, dtype=torch.int64)

    def store(captured_slots, values):
        cache.index_copy_(0, captured_slots, values)
        return cache

    wrapper = wrapper_type(store, None, enable_enpu=True)
    for step, metadata in enumerate(maps):
        context_holder[0].attn_metadata = metadata
        value.fill_(step + 1)
        wrapper(value)
        assert context_holder[0].batch_descriptor is descriptor
    assert cache[5 : 5 + steps_count].tolist() == list(range(1, steps_count + 1))
    assert len(wrapper.entries) == steps_count
    for entry_descriptor in wrapper.entries:
        for field in fields(Descriptor):
            assert getattr(entry_descriptor, field.name) == getattr(descriptor, field.name)

    # New Python metadata, same per-step persistent operands, updated content.
    new_maps = [{"qsa": SimpleNamespace(slot_mapping=slot)} for slot in slots]
    context_holder[0] = SimpleNamespace(
        attn_metadata=new_maps[0],
        draft_attn_metadatas=new_maps,
        batch_descriptor=descriptor,
        cudagraph_runtime_mode=runtime_mode,
    )
    for step, metadata in enumerate(new_maps):
        slots[step].fill_(15 + step)
        context_holder[0].attn_metadata = metadata
        value.fill_(11 + step)
        wrapper(value)
    assert cache[15 : 15 + steps_count].tolist() == list(range(11, 11 + steps_count))
    assert len(wrapper.entries) == steps_count
    wrapper.clear_graphs()
    assert wrapper.entries == {}


def test_legacy_shared_bucket_reproduces_second_step_cache_overwrite():
    descriptor = Descriptor(3, 1, True)
    maps = [{"qsa": SimpleNamespace(slot_mapping=torch.tensor([5 + step]))} for step in range(2)]
    context_holder = [
        SimpleNamespace(
            attn_metadata=maps[0],
            draft_attn_metadatas=maps,
            batch_descriptor=descriptor,
            cudagraph_runtime_mode="full",
        )
    ]
    # Disable the step-specific branch to reproduce the original dispatch.
    wrapper_type = _load_wrapper(context_holder, SimpleNamespace(is_draft_model=False))
    cache = torch.zeros(8, dtype=torch.int64)
    value = torch.ones(1, dtype=torch.int64)
    wrapper = wrapper_type(lambda slots, values: cache.index_copy_(0, slots, values), None, enable_enpu=True)
    wrapper(value)
    context_holder[0].attn_metadata = maps[1]
    value.fill_(2)
    wrapper(value)
    assert cache[5:7].tolist() == [2, 0]
    assert len(wrapper.entries) == 1


@pytest.mark.parametrize("case", ["target", "missing_steps", "missing_descriptor", "unmatched_metadata"])
def test_unrelated_graph_dispatch_and_descriptor_survive_failures(case):
    descriptor = None if case == "missing_descriptor" else Descriptor(3)
    metadata = {"qsa": SimpleNamespace(slot_mapping=torch.tensor([0]))}
    context = SimpleNamespace(
        attn_metadata=metadata,
        draft_attn_metadatas=None if case == "missing_steps" else [metadata],
        batch_descriptor=descriptor,
        cudagraph_runtime_mode="full",
    )
    if case == "unmatched_metadata":
        context.attn_metadata = dict(metadata)
    wrapper_type = _load_wrapper([context], SimpleNamespace(is_draft_model=case != "target"))

    def fail(*args):
        raise RuntimeError("capture failure")

    wrapper = wrapper_type(fail, None, enable_enpu=True)
    expected_error = AssertionError if case == "missing_descriptor" else RuntimeError
    with pytest.raises(expected_error):
        wrapper(torch.ones(1))
    assert context.batch_descriptor is descriptor
    if descriptor is not None:
        assert list(wrapper.entries) == [descriptor]


def test_draft_descriptor_is_restored_when_capture_raises():
    descriptor = Descriptor(3)
    metadata = {"qsa": SimpleNamespace(slot_mapping=torch.tensor([0]))}
    context = SimpleNamespace(
        attn_metadata=metadata,
        draft_attn_metadatas=[metadata],
        batch_descriptor=descriptor,
        cudagraph_runtime_mode="full",
    )
    wrapper_type = _load_wrapper([context], SimpleNamespace(is_draft_model=True))

    def fail(*args):
        raise RuntimeError("capture failure")

    with pytest.raises(RuntimeError, match="capture failure"):
        wrapper_type(fail, None, enable_enpu=True)(torch.ones(1))
    assert context.batch_descriptor is descriptor


@pytest.mark.parametrize("has_context", [False, True])
def test_eager_draft_calls_retain_original_dispatch(has_context):
    descriptor = Descriptor(3)
    metadata = {"qsa": SimpleNamespace(slot_mapping=torch.tensor([0]))}
    context = (
        SimpleNamespace(
            attn_metadata=metadata,
            draft_attn_metadatas=[metadata],
            batch_descriptor=descriptor,
            cudagraph_runtime_mode="none",
        )
        if has_context
        else None
    )
    wrapper_type = _load_wrapper([context], SimpleNamespace(is_draft_model=True))
    tensor = torch.ones(1)
    wrapper = wrapper_type(lambda value: value, None, enable_enpu=True)
    assert wrapper(tensor) is tensor
    assert wrapper.entries == {}
    if has_context:
        assert context.batch_descriptor is descriptor
