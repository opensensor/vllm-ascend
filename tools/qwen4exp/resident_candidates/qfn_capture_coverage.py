# SPDX-License-Identifier: Apache-2.0
"""Cold resident C1-C4 graph coverage; no request-time capture or replay observer."""

import hashlib
import importlib
from pathlib import Path

from tools.qwen4exp.resident_candidates.qsa_cold_variants import CAPTURE_SIZES, EXTENDED_CAPTURE_SIZES
from tools.qwen4exp.resident_candidates.qsa_graph_dispatch import validate_empty_graph_parameters

SOURCE_DIGESTS = (
    ("vllm_ascend._310p.model_runner_310p", "60b6fa07020df773a379eeb2895e06fba01ccaadfa80a196b4a47890c25565a8"),
    ("vllm_ascend.worker.model_runner_v1", "2623ecb3ae4b765a0f81d582c1e0cb8485c9164e993660416fb9edab5e892eda"),
    ("vllm.v1.attention.backends.gdn_attn", "3831019404a92200a38b43571c040cffeb6638bbdc71e4cb5e680dbc6c93d054"),
    ("vllm_ascend.ops.gdn_attn_builder", "499b30235bcef4b4b4a24ee2b2eaff91c21e306b7a129bc9f0a5fd91731f899e"),
    ("vllm_ascend._310p.ops.gdn_attn_builder_310", "5c72bac8c3a5b8e88a514d5b98cd476de867bf50472f9ee75e0f9652fd3cfaaf"),
    ("vllm.v1.cudagraph_dispatcher", "51f632753af729aeee6ce4a802d9d74b64f22eef02a524f36e2a24b07ef0cf63"),
)
MIN_FREE_BYTES = 256 << 20


def validate_runner(runner):
    config = runner.compilation_config
    sizes = tuple(config.cudagraph_capture_sizes)
    if (
        type(runner).__name__ != "NPUModelRunner310"
        or sizes not in (CAPTURE_SIZES, EXTENDED_CAPTURE_SIZES)
        or config.max_cudagraph_capture_size != max(sizes)
        or tuple(runner.cudagraph_batch_sizes) != sizes
        or config.cudagraph_mode.name != "FULL_DECODE_ONLY"
        or runner.uniform_decode_query_len != 3
        or runner.max_num_reqs != 4
        or runner.parallel_config.tensor_parallel_size != 4
        or runner.vllm_config.scheduler_config.async_scheduling
        or runner.vllm_config.scheduler_config.max_num_seqs != 4
        or runner.cudagraph_dispatcher.compilation_config is not config
        or runner.vllm_config.compilation_config is not config
    ):
        raise RuntimeError("capture coverage requires qualified TP4 MTP2 synchronous C4 runner")
    return sizes


def configure_retired_runner(runner, wrappers, parameters, sizes):
    """All inspection precedes mutation; caller must retain pause on failure.

    Standard resident_apply has already drained, retired all target/draft
    graphs and released native leases. Only empty graph-parameter dictionaries
    may be resized. Existing parameter and dispatcher objects remain owned.
    """
    sizes = tuple(sizes)
    if sizes not in (CAPTURE_SIZES, EXTENDED_CAPTURE_SIZES):
        raise ValueError("unsupported capture coverage")
    validate_runner(runner)
    if not wrappers or any(wrapper.entries for wrapper in wrappers):
        raise RuntimeError("retire every graph before changing coverage")
    if getattr(runner, "execute_model_state", None) is not None:
        raise RuntimeError("pending model execution prevents changing coverage")
    for params in parameters:
        validate_empty_graph_parameters(params)
    config = runner.compilation_config
    config.cudagraph_capture_sizes = list(sizes)
    config.max_cudagraph_capture_size = max(sizes)
    runner.cudagraph_batch_sizes = list(sizes)
    dispatcher = runner.cudagraph_dispatcher
    for keys in dispatcher.cudagraph_keys.values():
        keys.clear()
    dispatcher.keys_initialized = False
    dispatcher.initialize_cudagraph_keys(config.cudagraph_mode, runner.uniform_decode_query_len)
    for params in parameters:
        if params is None:
            continue
        for name in ("events", "handles", "attn_params", "workspaces"):
            values = getattr(params, name)
            values.clear()
            values.update({size: None if name == "workspaces" else [] for size in sizes})
    # The qualified Ascend proposer dispatches through runner.cudagraph_dispatcher
    # and records its draft graphs inside runner._dummy_run for each target key.
    return list(sizes)


def gdn_builders(runner):
    from vllm_ascend._310p.ops.gdn_attn_builder_310 import GDNAttentionMetadataBuilder310

    groups = list(runner.attn_groups)
    groups.extend(getattr(runner.drafter, "attn_groups", ()))
    found = {}
    while groups:
        group = groups.pop()
        if isinstance(group, (list, tuple)):
            groups.extend(group)
            continue
        for builder in group.metadata_builders:
            if hasattr(builder, "decode_cudagraph_max_bs"):
                if type(builder) is not GDNAttentionMetadataBuilder310:
                    raise RuntimeError("unreviewed recurrent metadata builder")
                found[id(builder)] = builder
    if not found:
        raise RuntimeError("expected resident GDN metadata builders")
    return list(found.values())


def resize_gdn_metadata(builder, capacity, owner, *, torch_module):
    """Cold allocation with old/new tensors retained before publishing pointers."""
    if builder.num_spec != 2 or capacity not in (6, 12):
        raise RuntimeError("metadata capacity requires the qualified MTP2 builder")
    old_capacity = builder.decode_cudagraph_max_bs
    if old_capacity not in (6, 12):
        raise RuntimeError("unqualified existing GDN metadata capacity")
    shapes = {
        "spec_state_indices_tensor": (capacity, 3),
        "non_spec_state_indices_tensor": (capacity,),
        "spec_sequence_masks": (capacity,),
        "spec_token_indx": (capacity * 3,),
        "non_spec_token_indx": (capacity * 3,),
        "spec_query_start_loc": (capacity + 1,),
        "non_spec_query_start_loc": (capacity + 1,),
        "num_accepted_tokens": (capacity,),
        "spec_sequence_masks_cpu": (capacity,),
        "spec_sequence_indices_cpu": (capacity,),
        "non_spec_sequence_indices_cpu": (capacity,),
        "spec_sequence_indices": (capacity,),
        "non_spec_sequence_indices": (capacity,),
    }
    if builder.spec_actual_seq_lengths is not None or builder.non_spec_actual_seq_lengths is not None:
        raise RuntimeError("common-kernel GDN metadata is outside the 310P contract")
    owner.update(builder=builder, old_capacity=old_capacity, capacity=capacity, previous={}, buffers={})
    for name, shape in shapes.items():
        old = getattr(builder, name)
        expected = tuple(
            old_capacity * 3
            if name in ("spec_token_indx", "non_spec_token_indx")
            else old_capacity + 1
            if name in ("spec_query_start_loc", "non_spec_query_start_loc")
            else old_capacity
            if axis == 0
            else value
            for axis, value in enumerate(shape)
        )
        if tuple(old.shape) != expected:
            raise RuntimeError(f"unexpected GDN buffer layout: {name}")
        owner["previous"][name] = old
    # Every original layout is admitted before allocating; same-capacity reload
    # preserves pointers and does not accumulate duplicate buffers.
    if old_capacity == capacity:
        return
    for name, shape in shapes.items():
        old = owner["previous"][name]
        kwargs = dict(dtype=old.dtype, device=old.device)
        if old.device.type == "cpu":
            kwargs["pin_memory"] = old.is_pinned()
        owner["buffers"][name] = torch_module.empty(shape, **kwargs)
    for name, value in owner["buffers"].items():
        setattr(builder, name, value)
    builder.decode_cudagraph_max_bs = capacity


def make_capture(sizes, prefix_sizes=(3,)):
    sizes, prefix_sizes = tuple(sizes), tuple(prefix_sizes)
    if prefix_sizes not in ((3,), sizes):
        raise ValueError("unsupported prefix coverage")

    def resident_capture(self):
        from tools.sidecard.qfn_resident_worker import QfnResidentWorkerExtension
        from vllm_ascend.compilation.acl_graph import (
            get_draft_graph_params,
            get_draft_graph_prefill_params,
            get_graph_params,
        )

        if self._resident_session().graphs_dirty:
            try:
                import torch

                configure_retired_runner(
                    self.model_runner,
                    self._resident_wrappers(),
                    [get_graph_params(), get_draft_graph_params(), get_draft_graph_prefill_params()],
                    sizes,
                )
                for wrapper in self._resident_wrappers():
                    wrapper._qfn_qsa_prefix_sizes = prefix_sizes
                # Graphs are retired before any builder pointer is replaced.
                owners = getattr(self, "_qfn_capture_metadata_owners", None)
                if owners is None:
                    owners = []
                    self._qfn_capture_metadata_owners = owners
                for builder in gdn_builders(self.model_runner):
                    owner = {}
                    owners.append(owner)
                    resize_gdn_metadata(builder, max(sizes), owner, torch_module=torch)
            except Exception as error:
                return self._resident_error(error)
        return QfnResidentWorkerExtension.resident_capture(self)

    return resident_capture


def inventory(worker):
    from tools.qwen4exp.resident_candidates.qsa_cold_variants import cold_inventory

    result = cold_inventory(worker)
    runner = worker.model_runner
    result["capture_coverage"] = dict(
        rank=worker.rank,
        runner_sizes=list(runner.cudagraph_batch_sizes),
        configured_sizes=list(runner.compilation_config.cudagraph_capture_sizes),
        maximum=runner.compilation_config.max_cudagraph_capture_size,
        gdn_capacities=[builder.decode_cudagraph_max_bs for builder in gdn_builders(runner)],
        descriptors=[str(desc) for _, descs in runner.cudagraph_dispatcher.get_capture_descs() for desc in descs],
    )
    return result


def replacements(worker, *, sizes=EXTENDED_CAPTURE_SIZES, prefix_sizes=(3,)):
    import torch

    for name, digest in SOURCE_DIGESTS:
        module = importlib.import_module(name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"capture coverage requires pinned source: {name}")
    validate_runner(worker.model_runner)
    free, _ = torch.npu.mem_get_info()
    if free < MIN_FREE_BYTES:
        raise RuntimeError("insufficient graph capture headroom")
    return {
        "vllm_ascend._310p.worker_310p:NPUWorker310.resident_capture": make_capture(sizes, prefix_sizes),
        "vllm_ascend._310p.worker_310p:NPUWorker310.qfn_graph_inventory": inventory,
    }
