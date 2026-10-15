# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold variant admission, no live capture, and complete native ownership."""

import ast
import json
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.qwen38_1m.test_qsa_graph_dispatch_candidate import Graph, metadata, observer, register
from tools.qwen4exp.resident_candidates import qsa_cold_variants as candidate
from tools.qwen4exp.resident_candidates.qsa_graph_dispatch import PrefixGeometry
from tools.sidecard.npu_collective_graph import CollectiveGraphObserver
from tools.sidecard.qfn_resident_lifecycle import renew_collective_graphs


def setup(size=3, *, capture_sizes=candidate.CAPTURE_SIZES, prefix_sizes=candidate.PREFIX_SIZES):
    descriptor = namedtuple("Descriptor", "num_tokens")(size)
    boundaries = metadata(512, (3,) * (size // 3))
    context = SimpleNamespace(batch_descriptor=descriptor, attn_metadata={"qsa": boundaries}, capturing=False)
    wrapper = SimpleNamespace(entries={})
    controller = candidate.ColdVariants(
        {"qsa": PrefixGeometry()}, observer(), capture_sizes=capture_sizes, prefix_sizes=prefix_sizes
    )
    calls = []

    def capture(key, variant):
        assert context._qfn_qsa_force_sparse_capture
        assert bool(context._qfn_qsa_prefix_capture) == (variant == "prefix")
        graph = Graph()
        graph.num_graphs = 2 if variant == "prefix" else 14
        graph.num_eager_breaks = graph.num_graphs - 1
        entry = SimpleNamespace(batch_descriptor=descriptor, capture=graph)
        wrapper.entries[key] = entry
        register(controller.observer, graph)
        context.capturing = True
        calls.append(("capture", variant))
        return entry, variant

    def replay(entry):
        calls.append(("replay", entry.capture.num_graphs))
        return entry.capture.num_graphs

    def eager():
        calls.append(("eager",))
        return "eager"

    def run(cold=False):
        return controller.call(
            wrapper, context, (), {}, capture_allowed=cold, capture=capture, replay=replay, eager=eager
        )

    return boundaries, context, wrapper, controller, calls, run


def test_c1_captures_once_then_switches_both_directions_without_retirement():
    boundaries, context, wrapper, controller, calls, run = setup()
    assert run(True) == "prefix"
    assert not context.capturing and not hasattr(context, "_qfn_qsa_prefix_capture")
    assert calls == [("capture", "general"), ("capture", "prefix")]
    entries = dict(wrapper.entries)
    leases = controller.observer.snapshot()["native_leases_acquired"]
    for length, expected in ((512, 2), (2051, 2), (2052, 14), (262144, 14), (22, 2), (2052, 14)):
        boundaries.seq_lens_cpu.fill_(length)
        assert run() == expected
    assert wrapper.entries == entries
    assert all(entry.capture.resets == 0 for entry in entries.values())
    assert controller.observer.snapshot()["native_leases_acquired"] == leases
    assert len(calls) == 8 and all(call[0] == "replay" for call in calls[2:])


def test_c2_keeps_only_original_general_graph():
    boundaries, _, wrapper, _, calls, run = setup(6)
    assert run(True) == "general"
    for length in (22, 2052, 262144, 22):
        boundaries.seq_lens_cpu.fill_(length)
        assert run() == 14
    assert len(wrapper.entries) == 1
    assert calls == [("capture", "general")] + [("replay", 14)] * 4


def test_live_missing_variant_cannot_allocate_or_capture():
    _, _, wrapper, _, calls, run = setup()
    assert run() == "eager" and wrapper.entries == {}
    assert calls == [("eager",)]


def test_changed_binding_fails_before_any_graph_or_eager_submission():
    boundaries, _, wrapper, controller, calls, run = setup()
    run(True)
    boundaries.block_tables = boundaries.block_tables.clone()
    with pytest.raises(RuntimeError, match="coordinated recovery required before submission"):
        run()
    assert calls == [("capture", "general"), ("capture", "prefix")]
    boundaries.seq_lens_cpu = None
    with pytest.raises(RuntimeError, match="metadata changed; coordinated recovery"):
        run()
    assert len(wrapper.entries) == len(controller.contracts) == 2
    assert len(calls) == 2


def test_unsupported_descriptor_never_captures_even_during_cold_window():
    _, _, wrapper, _, calls, run = setup(9)
    assert run(True) == "eager" and wrapper.entries == {}
    assert calls == [("eager",)]


def test_global_renewal_discovers_and_retires_both_variants(monkeypatch):
    _, _, wrapper, controller, _, run = setup()
    run(True)
    old = controller.observer
    wrapper._qfn_qsa_cold_variants = controller
    wrapper._collective_graph_observer = old
    wrapper.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=False),
        model_config=SimpleNamespace(max_model_len=262144),
    )
    wrapper.runnable = object()
    old.collective.graph_observer = old
    pool = object()
    controller.prefix_pool = pool
    captures = [entry.capture for entry in wrapper.entries.values()]
    receipt = renew_collective_graphs(
        old.collective,
        [wrapper],
        synchronize=lambda: None,
        clear_state=lambda wrappers: [item.entries.clear() for item in wrappers],
        observer_factory=CollectiveGraphObserver,
    )
    assert receipt["retired"] and all(graph.resets == 1 for graph in captures)
    assert receipt["native_leases_acquired"] == receipt["native_leases_released"]
    assert controller.prefix_pool is pool  # Retained until verified generation replacement.
    monkeypatch.setattr(candidate, "discover_geometries", lambda *args: {"qsa": PrefixGeometry()})
    fresh = candidate.controller_for_observer(wrapper, old.collective.graph_observer)
    assert fresh is not controller and fresh.prefix_pool is None and not fresh.contracts


def test_context_scope_restores_capturing_and_missing_attributes_after_failure():
    context = SimpleNamespace(capturing=False)
    with (
        pytest.raises(ValueError, match="cold failed"),
        candidate.context_attributes(context, capturing=False, prefix=True),
    ):
        context.capturing = True
        raise ValueError("cold failed")
    assert context.capturing is False and not hasattr(context, "prefix")


def test_cold_failure_does_not_admit_automatic_eager_retry():
    _, _, _, controller, _, run = setup()
    controller.failed = True
    with pytest.raises(RuntimeError, match="paused native recovery"):
        run()


def test_default_factory_blocked_and_explicit_factory_does_not_modify_live_code(monkeypatch):
    from vllm_ascend.models.qwen4_exp import model

    with pytest.raises(RuntimeError, match="blocked"):
        candidate.replacements()
    monkeypatch.setattr(candidate, "validate_qualified_wrapper", lambda: None)
    original = model._QSAAttention.forward
    replacements = candidate.make_replacements(qualification_gate=True)
    assert len(replacements) == 3 and model._QSAAttention.forward is original
    assert replacements["vllm_ascend.models.qwen4_exp.model:_QSAAttention.forward"] is not original


@pytest.mark.parametrize("allocator_selection", [None, "different allocator pool"])
def test_capture_restores_actual_allocator_selection_on_failure_and_retains_prefix_pool(allocator_selection):
    wrapper = SimpleNamespace(graph_pool="wrapper pool")
    controller = candidate.ColdVariants({}, None)
    allocator = SimpleNamespace(_graph_pool_id=allocator_selection)
    allocator.set_graph_pool_id = lambda value: setattr(allocator, "_graph_pool_id", value)
    pool = object()
    with (
        pytest.raises(RuntimeError, match="capture failed"),
        candidate.capture_pool(wrapper, controller, "prefix", allocator, lambda: pool),
    ):
        assert wrapper.graph_pool is pool
        allocator.set_graph_pool_id(pool)
        raise RuntimeError("capture failed")
    assert wrapper.graph_pool == "wrapper pool" and allocator._graph_pool_id == allocator_selection
    assert controller.prefix_pool is pool
    with candidate.capture_pool(wrapper, controller, "prefix", allocator, lambda: pytest.fail("new pool")):
        assert wrapper.graph_pool is pool


def test_wrapper_factory_guard_rejects_newer_draft_step_override():
    def upstream():
        pass

    def newer_draft_step_override():
        pass

    assert candidate.validate_wrapper_method(upstream, upstream) is upstream
    with pytest.raises(RuntimeError, match="qualified inherited"):
        candidate.validate_wrapper_method(newer_draft_step_override, upstream)
    own_prior = lambda: None
    own_prior._qfn_qsa_cold_base_call = upstream
    assert candidate.validate_wrapper_method(own_prior, upstream) is upstream


def real_gdn_metadata_helper():
    # Execute the actual pure-Torch helpers without importing hardware-only
    # torch_npu/upstream adapter dependencies. Qualified source has the same
    # recurrent memoization and uniform-state branch (audited read-only).
    source = Path("vllm_ascend/_310p/ops/fla/gdn_310.py").read_text()
    names = {"_flatten_state_indices", "_cached_recurrent_step_meta"}
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == 2
    namespace = {"torch": torch, "_EXTRA_CTX": SimpleNamespace(capturing=True)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<real-gdn-metadata-helpers>", "exec"), namespace)
    return namespace


def test_each_variant_records_real_gdn_derivation_and_retains_independent_scratch():
    _, context, wrapper, controller, _, _ = setup()
    original = context.attn_metadata
    gdn = SimpleNamespace(
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        state_indices=torch.tensor([[7, 8, 9]], dtype=torch.int32),
    )
    stale = (torch.tensor([-111, -111, -111]), torch.tensor([-222]))
    old_caches = {
        "_gdn_recurrent_step_meta": {("qwen4exp_spec", 3, True): stale},
        "_gdn_chunk_plan": object(),
        "_qwen4exp_mixed_views": object(),
        "_qwen4exp_query_ranges": object(),
    }
    vars(gdn).update(old_caches)
    original.update(gdn0=gdn, gdn1=gdn)
    helpers = real_gdn_metadata_helper()
    actual_flatten = helpers["_flatten_state_indices"]
    derivations = []

    def flatten(*args, **kwargs):
        derivations.append(args[0].data_ptr())
        return actual_flatten(*args, **kwargs)

    helpers["_flatten_state_indices"] = flatten
    helper = helpers["_cached_recurrent_step_meta"]
    recorded = {}

    def capture(key, variant):
        captured_metadata = context.attn_metadata
        assert captured_metadata["gdn0"] is captured_metadata["gdn1"]
        step = captured_metadata["gdn0"]
        assert step is not gdn
        assert step.query_start_loc is gdn.query_start_loc and step.state_indices is gdn.state_indices
        derived = helper(step, "qwen4exp_spec", step.query_start_loc, step.state_indices, 3, True)
        # All GDN layers within this one capture share its derivation.
        assert helper(step, "qwen4exp_spec", step.query_start_loc, step.state_indices, 3, True) is derived
        recorded[variant] = derived
        graph = Graph()
        graph.num_graphs, graph.num_eager_breaks = 2, 1
        entry = SimpleNamespace(capture=graph)
        wrapper.entries[key] = entry
        return entry, variant

    controller.call(
        wrapper,
        context,
        (),
        {},
        capture_allowed=True,
        capture=capture,
        replay=lambda entry: pytest.fail("startup replay"),
        eager=lambda: pytest.fail("fallback"),
    )
    assert len(derivations) == 2  # Before isolation the prefix omitted its derivation entirely.
    assert context.attn_metadata is original
    assert all(getattr(gdn, name) is value for name, value in old_caches.items())
    for variant, tensors in recorded.items():
        contract = controller.contracts[(context.batch_descriptor, variant)]
        assert contract.metadata["gdn0"]._gdn_recurrent_step_meta[("qwen4exp_spec", 3, True)] is tensors
        assert tensors[0].tolist() == [7, 8, 9] and tensors[1].tolist() == [3]
    for general, prefix in zip(recorded["general"], recorded["prefix"]):
        assert general.data_ptr() != prefix.data_ptr()
    recorded["general"][0].fill_(99)  # General/C2 scratch clobber must not corrupt prefix.
    assert recorded["prefix"][0].tolist() == [7, 8, 9]
    snapshot = controller.snapshot()
    assert snapshot["variant_metadata_separation"] == [
        {
            "descriptor": str(context.batch_descriptor),
            "shared_metadata_objects": 0,
            "shared_gdn_tensor_storages": 0,
            "prefix_gdn_tensor_storages": 2,
            "general_gdn_tensor_storages": 2,
        }
    ]
    assert all(value["gdn_cache_objects"] == 1 for value in snapshot["variants"])
    encoded = json.dumps(snapshot)
    assert all(str(value.data_ptr()) not in encoded for tensors in recorded.values() for value in tensors)


def test_original_metadata_restored_when_cold_capture_fails_after_derivation():
    _, context, wrapper, controller, _, _ = setup()
    original = context.attn_metadata
    original["qsa"]._gdn_recurrent_step_meta = {"original": object()}
    original_cache = original["qsa"]._gdn_recurrent_step_meta

    def capture(key, variant):
        assert context.attn_metadata is not original
        context.attn_metadata["qsa"]._gdn_recurrent_step_meta = {"new": object()}
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        controller.call(
            wrapper,
            context,
            (),
            {},
            capture_allowed=True,
            capture=capture,
            replay=lambda entry: None,
            eager=lambda: None,
        )
    assert context.attn_metadata is original and original["qsa"]._gdn_recurrent_step_meta is original_cache
    assert controller.failed


def test_real_gdn_helper_exposes_stale_cross_capture_memo_without_isolation():
    helper = real_gdn_metadata_helper()["_cached_recurrent_step_meta"]
    metadata = SimpleNamespace()
    boundaries = torch.tensor([0, 3], dtype=torch.int32)
    indices = torch.tensor([[7, 8, 9]], dtype=torch.int32)
    general = helper(metadata, "qwen4exp_spec", boundaries, indices, 3, True)
    boundaries[1] = 1
    indices.add_(20)
    broken_prefix = helper(metadata, "qwen4exp_spec", boundaries, indices, 3, True)
    assert broken_prefix is general
    assert broken_prefix[0].tolist() == [7, 8, 9] and broken_prefix[1].tolist() == [3]
    isolated = candidate.isolate_capture_metadata({"gdn": metadata})["gdn"]
    repaired_prefix = helper(isolated, "qwen4exp_spec", boundaries, indices, 3, True)
    assert repaired_prefix[0].tolist() == [27, 28, 29] and repaired_prefix[1].tolist() == [1]


@pytest.mark.parametrize("size", (9, 12))
def test_extended_coverage_captures_once_and_replays_general_across_lengths(size):
    boundaries, _, wrapper, controller, calls, run = setup(size, capture_sizes=candidate.EXTENDED_CAPTURE_SIZES)
    assert run(True) == "general"
    for length in (22, 2052, 262144, 22):
        boundaries.seq_lens_cpu.fill_(length)
        assert run() == 14
    assert len(wrapper.entries) == 1
    assert controller.snapshot()["capture_sizes"] == [3, 6, 9, 12]
    assert calls == [("capture", "general")] + [("replay", 14)] * 4


@pytest.mark.parametrize("sizes", ((12,), (3, 6, 12), (3, 6, 9, 12, 15)))
def test_unknown_capture_coverage_is_rejected(sizes):
    with pytest.raises(ValueError, match="bounded"):
        setup(capture_sizes=sizes)


@pytest.mark.parametrize("size", (3, 6, 9, 12))
def test_extended_prefix_selects_general_when_any_packed_request_crosses_boundary(size):
    sizes = candidate.EXTENDED_CAPTURE_SIZES
    metadata, _, wrapper, controller, calls, run = setup(size, capture_sizes=sizes, prefix_sizes=sizes)
    assert run(True) == "prefix"
    assert run() == 2
    metadata.seq_lens_cpu[-1] = 2052
    assert run() == 14
    metadata.seq_lens_cpu[-1] = 2051
    assert run() == 2
    metadata.seq_lens_cpu.fill_(262144)
    assert run() == 14
    assert len(wrapper.entries) == 2
    assert controller.snapshot()["prefix_sizes"] == [3, 6, 9, 12]
    assert calls[:2] == [("capture", "general"), ("capture", "prefix")]
    assert all(call[0] == "replay" for call in calls[2:])
