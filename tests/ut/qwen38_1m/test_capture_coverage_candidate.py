# SPDX-License-Identifier: Apache-2.0
"""Cold coverage changes must retain storage and reject live graph ownership."""

from types import SimpleNamespace

import pytest

from tools.qwen4exp.resident_candidates import qfn_capture_coverage as candidate


class Dispatcher:
    def __init__(self, config):
        self.compilation_config = config
        self.cudagraph_keys = {"full": {3, 6}}
        self.keys_initialized = True
        self.mapping = [0, 3, 3, 3, 6, 6, 6]

    def initialize_cudagraph_keys(self, mode, query_len):
        assert mode.name == "FULL_DECODE_ONLY" and query_len == 3
        sizes = self.compilation_config.cudagraph_capture_sizes
        self.cudagraph_keys["full"].update(sizes)
        self.mapping = [0] + [next(size for size in sizes if size >= n) for n in range(1, max(sizes) + 1)]
        self.keys_initialized = True


class NPUModelRunner310:
    def __init__(self):
        self.compilation_config = SimpleNamespace(
            cudagraph_capture_sizes=[3, 6],
            max_cudagraph_capture_size=6,
            cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY"),
        )
        self.cudagraph_batch_sizes = [3, 6]
        self.uniform_decode_query_len = 3
        self.max_num_reqs = 4
        self.parallel_config = SimpleNamespace(tensor_parallel_size=4)
        self.vllm_config = SimpleNamespace(
            compilation_config=self.compilation_config,
            scheduler_config=SimpleNamespace(async_scheduling=False, max_num_seqs=4),
        )
        self.cudagraph_dispatcher = Dispatcher(self.compilation_config)
        self.execute_model_state = None


def setup():
    params = SimpleNamespace(
        events={3: [], 6: []}, handles={3: [], 6: []}, attn_params={3: [], 6: []}, workspaces={3: None, 6: None}
    )
    return NPUModelRunner310(), [SimpleNamespace(entries={})], [params, None]


def test_extended_dispatch_and_baseline_rollback_preserve_parameter_owners():
    runner, wrappers, params = setup()
    owners = [id(getattr(params[0], field)) for field in ("events", "handles", "attn_params", "workspaces")]
    for sizes in ((3, 6, 9, 12), (3, 6)):
        assert candidate.configure_retired_runner(runner, wrappers, params, sizes) == list(sizes)
        assert runner.cudagraph_dispatcher.cudagraph_keys == {"full": set(sizes)}
        assert runner.cudagraph_dispatcher.mapping[-1] == max(sizes)
        assert runner.cudagraph_batch_sizes == list(sizes)
        assert runner.compilation_config.max_cudagraph_capture_size == max(sizes)
        assert list(params[0].events) == list(sizes)
        assert owners == [id(getattr(params[0], field)) for field in ("events", "handles", "attn_params", "workspaces")]
    assert len(runner.cudagraph_dispatcher.mapping) == 7


@pytest.mark.parametrize("field", ("events", "handles", "attn_params", "workspaces"))
def test_live_parameter_owner_rejected_before_configuration_mutation(field):
    runner, wrappers, params = setup()
    getattr(params[0], field)[3] = object() if field == "workspaces" else [object()]
    with pytest.raises(RuntimeError, match="global attention"):
        candidate.configure_retired_runner(runner, wrappers, params, (3, 6, 9, 12))
    assert runner.compilation_config.cudagraph_capture_sizes == [3, 6]
    assert runner.cudagraph_dispatcher.cudagraph_keys == {"full": {3, 6}}


@pytest.mark.parametrize("busy", ("graph", "execution"))
def test_live_graph_or_execution_rejected_before_mutation(busy):
    runner, wrappers, params = setup()
    if busy == "graph":
        wrappers[0].entries[3] = object()
    else:
        runner.execute_model_state = object()
    with pytest.raises(RuntimeError):
        candidate.configure_retired_runner(runner, wrappers, params, (3, 6, 9, 12))
    assert runner.compilation_config.max_cudagraph_capture_size == 6


@pytest.mark.parametrize("field,value", (("uniform_decode_query_len", 2), ("max_num_reqs", 8)))
def test_unqualified_geometry_rejected(field, value):
    runner, wrappers, params = setup()
    setattr(runner, field, value)
    with pytest.raises(RuntimeError, match="qualified"):
        candidate.configure_retired_runner(runner, wrappers, params, (3, 6, 9, 12))


def test_nonstandard_capture_sizes_fail_before_touching_runner():
    with pytest.raises(ValueError, match="unsupported"):
        candidate.configure_retired_runner(None, [], [], (12,))


def builder_fixture():
    import torch

    builder = SimpleNamespace(
        num_spec=2, decode_cudagraph_max_bs=6, spec_actual_seq_lengths=None, non_spec_actual_seq_lengths=None
    )
    shapes = dict(
        spec_state_indices_tensor=(6, 3),
        non_spec_state_indices_tensor=(6,),
        spec_sequence_masks=(6,),
        spec_token_indx=(18,),
        non_spec_token_indx=(18,),
        spec_query_start_loc=(7,),
        non_spec_query_start_loc=(7,),
        num_accepted_tokens=(6,),
        spec_sequence_masks_cpu=(6,),
        spec_sequence_indices_cpu=(6,),
        non_spec_sequence_indices_cpu=(6,),
        spec_sequence_indices=(6,),
        non_spec_sequence_indices=(6,),
    )
    for name, shape in shapes.items():
        dtype = torch.bool if "masks" in name else torch.int64 if "sequence_indices" in name else torch.int32
        setattr(builder, name, torch.ones(shape, dtype=dtype))
    return builder


def test_gdn_expansion_and_rollback_replace_all_buffers_with_matching_types():
    import torch

    builder = builder_fixture()
    for capacity in (12, 6):
        owner = {}
        candidate.resize_gdn_metadata(builder, capacity, owner, torch_module=torch)
        assert builder.decode_cudagraph_max_bs == capacity
        assert len(owner["buffers"]) == 13
        assert builder.spec_state_indices_tensor.shape == (capacity, 3)
        assert builder.spec_token_indx.numel() == capacity * 3
        assert builder.spec_query_start_loc.numel() == capacity + 1
        for name, previous in owner["previous"].items():
            assert getattr(builder, name).dtype == previous.dtype
            assert getattr(builder, name).data_ptr() != previous.data_ptr()


def test_same_capacity_reload_keeps_gdn_storage():
    import torch

    builder = builder_fixture()
    before = builder.spec_state_indices_tensor
    owner = {}
    candidate.resize_gdn_metadata(builder, 6, owner, torch_module=torch)
    assert builder.spec_state_indices_tensor is before
    assert not owner["buffers"]


def test_gdn_failed_allocation_retains_partial_tensors_without_publishing():
    import torch

    builder = builder_fixture()
    before = builder.spec_state_indices_tensor
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("allocation failed")
        return torch.empty(*args, **kwargs)

    owner = {}
    with pytest.raises(RuntimeError, match="allocation failed"):
        candidate.resize_gdn_metadata(builder, 12, owner, torch_module=SimpleNamespace(empty=fail_second))
    assert len(owner["buffers"]) == 1
    assert builder.decode_cudagraph_max_bs == 6
    assert builder.spec_state_indices_tensor is before


def test_gdn_unexpected_shape_rejected_before_allocation():
    import torch

    builder = builder_fixture()
    builder.spec_token_indx = torch.empty(6, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="layout"):
        candidate.resize_gdn_metadata(builder, 12, {}, torch_module=None)
    assert builder.decode_cudagraph_max_bs == 6
