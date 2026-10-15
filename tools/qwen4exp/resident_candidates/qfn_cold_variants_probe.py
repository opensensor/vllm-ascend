# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiny cold-variant gate; private tensors, pools, and zero-owner observer only.

No serving wrapper, live collective, model parameter, or cache is inspected or
modified. Passing this probe cannot qualify QSA math or model-state behavior.
"""

from collections import namedtuple
from types import SimpleNamespace

from tools.qwen4exp.resident_candidates.qfn_capture_semantics_probe import TARGET, _error, make_rpc

SWITCHES = (
    (512, (1.0, 2.0, 3.0), "prefix"),
    (2052, (3.0, 5.0, 7.0), "general"),
    (22, (-2.0, 4.0, 6.0), "prefix"),
    (262144, (8.0, 9.0, 10.0), "general"),
    (2051, (11.0, 12.0, 13.0), "prefix"),
)
VARIANT_OFFSETS = (11.0, 29.0)
DERIVED_CACHE_KEY = ("probe", 3, False)


class ProbeDescriptor(namedtuple("ProbeDescriptorBase", "num_tokens")):
    __slots__ = ()


def private_metadata(torch_module, device):
    """Real metadata tensors for the actual admission inspector; no KV storage."""
    return SimpleNamespace(
        seq_lens_cpu=torch_module.tensor([512], dtype=torch_module.int32),
        query_lens_cpu=torch_module.tensor([3], dtype=torch_module.int32),
        num_actual_tokens=3,
        num_prefills=0,
        num_decodes=1,
        attn_state=SimpleNamespace(name="SpecDecoding"),
        block_tables=torch_module.zeros((1, 1), dtype=torch_module.int32, device=device),
        query_start_loc=torch_module.tensor([0, 3], dtype=torch_module.int32, device=device),
        slot_mapping=torch_module.arange(3, dtype=torch_module.int64, device=device),
        seq_lens=torch_module.tensor([512], dtype=torch_module.int32, device=device),
    )


def run_switches(torch_module, metadata, inputs, parameter, counter, invoke):
    """Changing inputs/lengths must execute exactly one selected native replay."""
    records = []
    original_parameter = parameter.cpu().clone()
    counter.zero_()
    for step, (length, values, variant) in enumerate(SWITCHES, start=1):
        metadata.seq_lens_cpu.fill_(length)
        metadata.seq_lens.copy_(metadata.seq_lens_cpu)
        current = torch_module.tensor(values, dtype=inputs.dtype)
        inputs.copy_(current)
        output = invoke()
        observed = output.cpu()
        offset = VARIANT_OFFSETS[0 if variant == "prefix" else 1]
        expected = current * original_parameter + offset
        torch_module.testing.assert_close(observed, expected, rtol=0, atol=0)
        observed_counter = float(counter.cpu().item())
        if observed_counter != step:
            raise RuntimeError(f"variant dispatch executed {observed_counter} increments after {step} requests")
        records.append(
            dict(length=length, variant=variant, values=observed.tolist(), replay_increments=observed_counter)
        )
    if not torch_module.equal(parameter.cpu(), original_parameter):
        raise RuntimeError("private read-only parameter changed")
    return records


def cached_product(metadata, inputs, parameter):
    """Reproduce GDN's per-metadata memoized device producer on private tensors.

    A second capture on the same metadata takes the cache hit and never records
    the multiplication. Separate graph pools alone cannot fix that dependency.
    """
    cache = getattr(metadata, "_gdn_recurrent_step_meta", None)
    if cache is None:
        cache = {}
        metadata._gdn_recurrent_step_meta = cache
    if DERIVED_CACHE_KEY not in cache:
        cache[DERIVED_CACHE_KEY] = inputs * parameter
    return cache[DERIVED_CACHE_KEY]


def run_probe():
    import torch
    from vllm.compilation import monitor
    from vllm.compilation.breakable_cudagraph import _BreakableEntry
    from vllm.distributed.device_communicators import pynccl_allocator

    from tools.qwen4exp.resident_candidates import qsa_cold_variants as variants
    from tools.qwen4exp.resident_candidates.qsa_graph_dispatch import (
        PrefixDispatcher,
        PrefixGeometry,
        serialized_recapture_window,
    )
    from tools.sidecard.npu_collective_breakable import _CollectiveBreakableCapture
    from tools.sidecard.npu_collective_graph import CollectiveGraphObserver
    from vllm_ascend.utils import weak_ref_tensor

    result = dict(scope="private_cold_variants_native_pools_not_qsa_or_transport_qualification", passed=False)
    facade = SimpleNamespace(state="active", owners={}, fallbacks={}, _context=lambda: None)
    observer = CollectiveGraphObserver(facade, track_replays=False)
    window_owner = PrefixDispatcher({})
    caller_stream = torch.npu.current_stream()
    monitor_before = monitor.cudagraph_capturing_enabled
    allocator_before = pynccl_allocator._graph_pool_id
    wrapper = None
    captures = []
    derived_addresses = []
    try:
        if caller_stream != torch.npu.default_stream():
            raise RuntimeError("private cold-variant gate requires the normal default caller stream")
        device = f"npu:{torch.npu.current_device()}"
        with torch.inference_mode():
            metadata = private_metadata(torch, device)
            context = SimpleNamespace(
                batch_descriptor=ProbeDescriptor(3), attn_metadata={"probe.qsa": metadata}, capturing=False
            )
            inputs = torch.zeros(3, dtype=torch.float32, device=device)
            parameter = torch.tensor([2.0, 4.0, 8.0], dtype=torch.float32, device=device)
            counter = torch.zeros(1, dtype=torch.float32, device=device)
            wrapper = SimpleNamespace(entries={}, graph_pool=torch.npu.graph_pool_handle())
            controller = variants.ColdVariants({"probe.qsa": PrefixGeometry()}, observer)
            # Warm operations outside either private graph pool.
            for offset in VARIANT_OFFSETS:
                warm = inputs * parameter + offset
                counter.add_(1.0)
            del warm
            counter.zero_()
            caller_stream.synchronize()

            def capture(key, variant):
                if torch.npu.current_stream() == torch.npu.default_stream():
                    raise RuntimeError("private capture attempted on default stream")
                entry = _BreakableEntry(batch_descriptor=context.batch_descriptor)
                wrapper.entries[key] = entry
                with variants.capture_pool(wrapper, controller, variant, pynccl_allocator, torch.npu.graph_pool_handle):
                    pynccl_allocator.set_graph_pool_id(wrapper.graph_pool)
                    graph = _CollectiveBreakableCapture(pool=wrapper.graph_pool)
                    captures.append(graph)
                    with observer.capture_scope(graph), graph:
                        counter.add_(1.0)
                        derived = cached_product(context.attn_metadata["probe.qsa"], inputs, parameter)
                        derived_addresses.append(derived.data_ptr())
                        output = derived + VARIANT_OFFSETS[0 if variant == "prefix" else 1]
                        output = weak_ref_tensor(output)
                    entry.capture, entry.output = graph, output
                return entry, entry.output

            def replay(entry):
                entry.capture.replay()
                return entry.output

            def eager():
                raise RuntimeError("unexpected private gate eager fallback")

            def invoke(cold=False):
                return controller.call(
                    wrapper, context, (inputs,), {}, capture_allowed=cold, capture=capture, replay=replay, eager=eager
                )

            with serialized_recapture_window(window_owner):
                invoke(cold=True)
            caller_stream.synchronize()
            before = observer.snapshot()
            output_addresses = [entry.output.data_ptr() for entry in wrapper.entries.values()]
            result["counter_after_recording"] = float(counter.cpu().item())
            result["records"] = run_switches(torch, metadata, inputs, parameter, counter, invoke)
            after = observer.snapshot()
            result.update(
                private_parameter_unchanged=True,
                captures_before_replays=before,
                captures_after_replays=after,
                capture_count=len(captures),
                entries=len(wrapper.entries),
                distinct_pools=controller.prefix_pool is not None and controller.prefix_pool != wrapper.graph_pool,
                distinct_output_storage=len(set(output_addresses)) == 2,
                distinct_derived_storage=len(set(derived_addresses)) == 2,
                original_metadata_clean=not hasattr(metadata, "_gdn_recurrent_step_meta"),
                context_flags_restored=(
                    not context.capturing
                    and not hasattr(context, "_qfn_qsa_prefix_capture")
                    and not hasattr(context, "_qfn_qsa_force_sparse_capture")
                ),
                no_live_observer_used=True,
                private_native_owners=len(facade.owners),
            )
            result["passed"] = (
                len(captures) == len(wrapper.entries) == 2
                and before == after
                and before["registered_graphs"] == 2
                and before["native_leases_acquired"] == 0
                and result["distinct_pools"]
                and result["distinct_output_storage"]
                and result["distinct_derived_storage"]
                and result["original_metadata_clean"]
                and result["context_flags_restored"]
            )
    except BaseException as error:
        result["error"] = _error(error)
    finally:
        try:

            def drain():
                caller_stream.synchronize()
                if window_owner.recapture_resources is not None:
                    window_owner.recapture_resources[0].synchronize()

            result["retirement"] = observer.retire(drain)
            if result["retirement"]["reset_graphs"] != len(captures):
                raise RuntimeError("not all private captures were reset")
        except BaseException as error:
            result["passed"] = False
            result["cleanup_error"] = _error(error)
        result["caller_stream_restored"] = torch.npu.current_stream() == caller_stream
        result["monitor_restored"] = monitor.cudagraph_capturing_enabled == monitor_before
        result["allocator_pool_restored"] = pynccl_allocator._graph_pool_id == allocator_before
        if not all(result[name] for name in ("caller_stream_restored", "monitor_restored", "allocator_pool_restored")):
            result["passed"] = False
    return result


def replacements():
    from vllm_ascend._310p.worker_310p import NPUWorker310

    return {TARGET: make_rpc(NPUWorker310.qfn_cpu_profile_stop, probe=run_probe)}
