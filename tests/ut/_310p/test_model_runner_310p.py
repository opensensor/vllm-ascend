#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec, MLAAttentionSpec
from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

from tests.ut.base import TestBase
from vllm_ascend._310p.model_runner_310p import (
    NPUModelRunner310,
    _allocate_attention_cache_tensor,
    _get_attention_cache_tensor_shape,
    _get_layer_attention_backends,
    _iter_kv_cache_tensors,
)
from vllm_ascend._310p.prefix_mamba_state import PrefixMambaStateTier
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def _prepare_inputs_source() -> str:
    source_path = Path(__file__).resolve().parents[3] / "vllm_ascend" / "_310p" / "model_runner_310p.py"
    source = source_path.read_text(encoding="utf-8")
    start = source.index("    def _prepare_inputs(")
    end = source.index("    @torch.inference_mode()", start)
    return source[start:end]


def test_prepare_inputs_keeps_aclgraph_metadata_on_cpu() -> None:
    source = _prepare_inputs_source()

    assert "block_table.compute_slot_mapping(" in source
    assert "req_indices," in source
    assert "positions_np[:total_num_scheduled_tokens]" in source

    assert "self.input_batch.block_table.compute_slot_mapping(" not in source
    assert "query_start_loc.gpu[: num_reqs + 1]" not in source
    assert "req_indices_gpu" not in source
    assert "self.num_computed_tokens[req_indices_gpu]" not in source

    assert "self.positions[:total_num_scheduled_tokens].copy_(" in source
    assert "self._positions_cpu_buf[:total_num_scheduled_tokens]" in source
    assert "self.seq_lens[:num_reqs].copy_(" in source
    assert "self.optimistic_seq_lens_cpu[:num_reqs]" in source
    assert "self._sync_num_accepted_tokens(" in source
    assert "self.input_batch.num_accepted_tokens_cpu[" not in source


def test_model_forward_updates_mtp_full_graph_params_before_replay() -> None:
    runner = object.__new__(NPUModelRunner310)
    runner.uses_mrope = False
    runner.enable_enpu = False
    runner.speculative_config = SimpleNamespace(method="mtp")
    runner._qwen4exp_mtp_ple = False
    runner.update_stream = MagicMock()
    runner._all_gather_hidden_states_and_aux = MagicMock()

    calls = []

    def fake_update(*args):
        calls.append("update")

    def fake_model(**kwargs):
        calls.append("model")
        return torch.ones(1)

    runner.model = fake_model
    runner._update_full_graph_params_if_needed = fake_update
    forward_context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        capturing=False,
    )

    with patch(
        "vllm_ascend._310p.model_runner_310p.get_forward_context",
        return_value=forward_context,
    ):
        hidden_states = runner._model_forward(
            8,
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0]),
        )

    assert calls == ["update", "model"]
    torch.testing.assert_close(hidden_states, torch.ones(1))


def test_310p_runner_does_not_advertise_standardized_shared_kv_backing() -> None:
    assert NPUModelRunner310.supports_standardized_shared_kv_backing is False
    assert NPUModelRunner310.supports_glm5_next_shared_kv_slots is True
    assert NPUModelRunner310.supports_compact_mamba_state is False


def test_input_embedding_staging_buffer_is_not_pinned() -> None:
    runner = object.__new__(NPUModelRunner310)
    runner.device = torch.device("cpu")
    runner.max_num_tokens = 2048
    runner.inputs_embeds_size = 2560

    with (
        patch("vllm_ascend._310p.model_runner_310p.CpuGpuBuffer") as buffer_factory,
        patch("vllm_ascend._310p.model_runner_310p.NPUModelRunner._make_buffer") as parent_make_buffer,
    ):
        buffer = runner._make_buffer(2048, 2560, dtype=torch.float16, numpy=False)
        assert buffer is buffer_factory.return_value
        buffer_factory.assert_called_once_with(
            2048,
            2560,
            dtype=torch.float16,
            device=torch.device("cpu"),
            pin_memory=False,
            with_numpy=False,
        )

        runner._make_buffer(1, dtype=torch.int32)
        parent_make_buffer.assert_called_once_with(1, dtype=torch.int32, numpy=True)


def test_glm5_next_cache_initialization_uses_shared_slot_allocator() -> None:
    runner = object.__new__(NPUModelRunner310)
    runner.model_config = SimpleNamespace(use_mla=False)
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None)
    runner.use_sparse = False
    runner.runner_only_attn_layers = set()
    runner.shared_kv_cache_layers = {}
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.kv_caches = []
    runner._qsa_index_caches = {}

    layer_name = "model.layers.3.self_attn"
    spec = SimpleNamespace(model_version="glm5_next")
    cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=spec,
                layer_names=[layer_name],
            )
        ]
    )
    raw_caches = {layer_name: torch.empty(1)}
    reshaped_caches = {layer_name: [torch.empty(1)]}

    with (
        patch.object(
            NPUModelRunner,
            "_allocate_kv_cache_tensors",
            return_value=raw_caches,
        ) as allocate,
        patch.object(
            NPUModelRunner,
            "_reshape_kv_cache_tensors",
            return_value=reshaped_caches,
        ) as reshape,
        patch("vllm.v1.worker.utils.bind_kv_cache") as bind_kv_cache,
    ):
        result = runner.initialize_kv_cache_tensors(cache_config)

    assert result is reshaped_caches
    allocate.assert_called_once_with(runner, cache_config)
    reshape.assert_called_once_with(runner, cache_config, raw_caches)
    bind_kv_cache.assert_called_once_with(
        reshaped_caches,
        runner.compilation_config.static_forward_context,
        runner.kv_caches,
    )


def test_iter_kv_cache_tensors_flattens_hybrid_layout() -> None:
    attention_k = torch.empty(2)
    attention_v = torch.empty(2)
    mamba_conv = torch.empty(3)
    mamba_ssm = torch.empty(4)

    flattened = list(_iter_kv_cache_tensors([(attention_k, attention_v), [mamba_conv, mamba_ssm]]))

    assert flattened == [attention_k, attention_v, mamba_conv, mamba_ssm]


def test_mla_cache_shape_uses_divisible_kernel_page_and_keeps_blocks() -> None:
    backend = SimpleNamespace(
        get_supported_kernel_block_sizes=lambda: [32, 16],
        get_kv_cache_shape=lambda blocks, block_size, heads, head_size: (
            blocks,
            block_size,
            heads,
            head_size,
        ),
    )
    cache_spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float16,
    )

    shape = _get_attention_cache_tensor_shape(backend, 64, cache_spec)

    assert shape == (64, 16, 1, 512)


def test_dense_cache_shape_splits_leading_key_value_axis() -> None:
    backend = SimpleNamespace(
        get_supported_kernel_block_sizes=lambda: [128, 64],
        get_kv_cache_shape=lambda blocks, block_size, heads, head_size: (
            2,
            blocks,
            heads * head_size // 16,
            block_size,
            16,
        ),
    )
    cache_spec = AttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
    )

    shape = _get_attention_cache_tensor_shape(backend, 64, cache_spec)

    assert shape == (64, 8, 128, 16)


def test_layer_attention_backends_preserve_hybrid_group_selection() -> None:
    linear_backend = object()
    mla_backend = object()
    attn_groups = [
        [
            SimpleNamespace(backend=linear_backend, layer_names=["layers.0.attn"]),
            SimpleNamespace(backend=mla_backend, layer_names=["layers.3.attn"]),
        ]
    ]

    layer_backends = _get_layer_attention_backends(attn_groups)

    assert layer_backends == {
        "layers.0.attn": linear_backend,
        "layers.3.attn": mla_backend,
    }


def test_mla_cache_allocation_keeps_nd_layout() -> None:
    cache_spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float16,
    )

    with patch(
        "vllm_ascend._310p.model_runner_310p.torch_npu.empty_with_format",
        create=True,
    ) as empty_nz:
        cache = _allocate_attention_cache_tensor(
            (4, 16, 1, 8),
            torch.float16,
            torch.device("cpu"),
            cache_spec,
        )

    assert cache.shape == (4, 16, 1, 8)
    assert cache.is_contiguous()
    empty_nz.assert_not_called()


def test_nested_hybrid_cache_copy_uses_scheduler_block_geometry() -> None:
    attention = (torch.arange(128).reshape(64, 2), torch.arange(128, 256).reshape(64, 2))
    mamba = [torch.arange(192).reshape(64, 3)]

    with patch(
        "vllm.v1.worker.utils.async_tensor_h2d",
        return_value=torch.tensor([[1, 5]], dtype=torch.int64),
    ):
        copy_kv_cache_blocks_inplace(
            _iter_kv_cache_tensors([attention, mamba]),
            num_blocks=64,
            kv_cache_block_copies=[(1, 5)],
        )

    for cache in (*attention, *mamba):
        torch.testing.assert_close(cache[5], cache[1])


def test_update_states_copies_nested_hybrid_cache_once() -> None:
    runner = object.__new__(NPUModelRunner310)
    caches = [(torch.empty(2), torch.empty(2)), [torch.empty(3)]]
    runner.kv_caches = caches
    runner.kv_cache_config = SimpleNamespace(num_blocks=8)
    block_copies = [(1, 2)]
    scheduler_output = SimpleNamespace(
        kv_cache_block_copies=block_copies,
        finished_req_ids=set(),
    )
    deferred = object()
    base_observed_copies = []

    def fake_base_update_states(_runner, output):
        base_observed_copies.append(output.kv_cache_block_copies)
        return deferred

    with (
        patch(
            "vllm_ascend._310p.model_runner_310p.NPUModelRunner._update_states",
            new=fake_base_update_states,
        ),
        patch("vllm.v1.worker.utils.copy_kv_cache_blocks_inplace") as copy_blocks,
    ):
        result = runner._update_states(scheduler_output)

    assert result is deferred
    assert base_observed_copies == [None]
    assert scheduler_output.kv_cache_block_copies == block_copies
    copied_caches = list(copy_blocks.call_args.args[0])
    assert copied_caches == [caches[0][0], caches[0][1], caches[1][0]]
    assert copy_blocks.call_args.args[1:] == (8, block_copies)


def test_single_request_runner_compacts_mamba_allocation() -> None:
    runner = object.__new__(NPUModelRunner310)
    runner.device = torch.device("cpu")
    runner.runner_only_attn_layers = set()
    runner.max_num_reqs = 1
    runner.supports_compact_mamba_state = True
    runner.num_compact_mamba_blocks = 1
    spec = MambaSpec(
        block_size=128,
        shapes=((4, 8), (2, 4)),
        dtypes=(torch.float16, torch.float16),
    )
    layer_name = "model.layers.0.linear_attn"
    kv_cache_config = SimpleNamespace(
        num_blocks=128,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=[layer_name])],
        kv_cache_tensors=[
            SimpleNamespace(
                size=128 * spec.page_size_bytes,
                layers=[layer_name],
                shared_by=[layer_name],
            )
        ],
    )

    caches = runner._allocate_kv_cache_tensors(kv_cache_config)

    assert caches[layer_name][0].shape == (1, 4, 8)
    assert caches[layer_name][1].shape == (1, 2, 4)
    assert caches[layer_name][0].untyped_storage().nbytes() == spec.page_size_bytes


def test_310p_spec_dummy_capture_keeps_decode_graph_enabled() -> None:
    runner = object.__new__(NPUModelRunner310)
    runner.input_batch = SimpleNamespace(num_computed_tokens_cpu=np.array([1], dtype=np.int32))
    runner.attn_state = AscendAttentionState.ChunkedPrefill
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=1)
    runner.uniform_decode_query_len = 2
    kwargs = dict(
        num_tokens=2,
        num_reqs=1,
        num_scheduled_tokens_np=np.array([2], dtype=np.int32),
        max_num_scheduled_tokens=2,
        use_cascade_attn=False,
    )
    with patch.object(NPUModelRunner, "_determine_batch_execution_and_padding", return_value=None) as parent:
        runner._spec_dummy_capture = True
        runner._determine_batch_execution_and_padding(**kwargs)
        assert parent.call_args.kwargs["force_eager"] is False

        runner._spec_dummy_capture = False
        runner._determine_batch_execution_and_padding(**kwargs)
        assert parent.call_args.kwargs["force_eager"] is True


def test_single_request_runner_remaps_only_mamba_device_table() -> None:
    runner = object.__new__(NPUModelRunner310)
    runner.supports_compact_mamba_state = True
    runner.num_compact_mamba_blocks = 1
    attention_gpu = torch.tensor([[17, 18]], dtype=torch.int32)
    mamba_gpu = torch.tensor([[91, 92]], dtype=torch.int32)
    attention_table = SimpleNamespace(
        is_mamba_group=False,
        block_table=SimpleNamespace(gpu=attention_gpu),
    )
    mamba_table = SimpleNamespace(
        is_mamba_group=True,
        block_table=SimpleNamespace(gpu=mamba_gpu),
    )
    runner.input_batch = SimpleNamespace(block_table=SimpleNamespace(block_tables=[attention_table, mamba_table]))

    runner._remap_compact_mamba_block_tables(num_reqs=1)

    torch.testing.assert_close(attention_gpu, torch.tensor([[17, 18]], dtype=torch.int32))
    torch.testing.assert_close(mamba_gpu, torch.zeros_like(mamba_gpu))


def test_single_request_mtp_keeps_two_distinct_mamba_states() -> None:
    runner = object.__new__(NPUModelRunner310)
    runner.device = torch.device("cpu")
    runner.runner_only_attn_layers = set()
    runner.max_num_reqs = 1
    runner.supports_compact_mamba_state = True
    runner.num_compact_mamba_blocks = 2
    spec = MambaSpec(
        block_size=128,
        shapes=((4, 8), (2, 4)),
        dtypes=(torch.float16, torch.float16),
        num_speculative_blocks=1,
    )
    layer_name = "model.layers.0.linear_attn"
    kv_cache_config = SimpleNamespace(
        num_blocks=128,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=[layer_name])],
        kv_cache_tensors=[
            SimpleNamespace(
                size=128 * spec.page_size_bytes,
                layers=[layer_name],
                shared_by=[layer_name],
            )
        ],
    )

    caches = runner._allocate_kv_cache_tensors(kv_cache_config)
    assert caches[layer_name][0].shape == (2, 4, 8)
    assert caches[layer_name][1].shape == (2, 2, 4)

    attention_gpu = torch.tensor([[17, 18, 19]], dtype=torch.int32)
    mamba_gpu = torch.tensor([[91, 92, 93]], dtype=torch.int32)
    runner.input_batch = SimpleNamespace(
        block_table=SimpleNamespace(
            block_tables=[
                SimpleNamespace(is_mamba_group=False, block_table=SimpleNamespace(gpu=attention_gpu)),
                SimpleNamespace(is_mamba_group=True, block_table=SimpleNamespace(gpu=mamba_gpu)),
            ]
        )
    )
    runner._remap_compact_mamba_block_tables(num_reqs=1)
    torch.testing.assert_close(attention_gpu, torch.tensor([[17, 18, 19]], dtype=torch.int32))
    torch.testing.assert_close(mamba_gpu, torch.tensor([[0, 1, 0]], dtype=torch.int32))


def test_single_request_mtp_remaps_later_blocks_for_forward_and_state_copies() -> None:
    runner = object.__new__(NPUModelRunner310)
    runner.supports_compact_mamba_state = True
    runner.supports_prefix_mamba_state_tier = False
    runner.num_compact_mamba_blocks = 2
    attention_gpu = torch.tensor([[17, 18, 19, 20]], dtype=torch.int32)
    mamba_cpu = np.array([[100, 101, 102, 103]], dtype=np.int32)
    mamba_gpu = torch.from_numpy(mamba_cpu.copy())
    runner.input_batch = SimpleNamespace(
        req_ids=["request"],
        block_table=SimpleNamespace(
            block_tables=[
                SimpleNamespace(is_mamba_group=False, block_table=SimpleNamespace(gpu=attention_gpu)),
                SimpleNamespace(is_mamba_group=True, block_table=SimpleNamespace(np=mamba_cpu, gpu=mamba_gpu)),
            ]
        ),
    )
    original = ([17, 18, 19, 20], [100, 101, 102, 103])
    req_state = SimpleNamespace(block_ids=original)
    runner.requests = {"request": req_state}

    runner._remap_compact_mamba_block_tables(num_reqs=1)
    torch.testing.assert_close(mamba_gpu, torch.tensor([[0, 1, 0, 1]], dtype=torch.int32))
    np.testing.assert_array_equal(mamba_cpu, [[100, 101, 102, 103]])
    np.testing.assert_array_equal(runner.input_batch._prefix_mamba_postprocess_tables[1], [[0, 1, 0, 1]])

    runner._stage_prefix_mamba_request_ids()
    assert req_state.block_ids == ([17, 18, 19, 20], [0, 1, 0, 1])
    runner._restore_prefix_mamba_request_ids()
    assert req_state.block_ids is original


def test_prefix_mamba_remap_preserves_scheduler_ids() -> None:
    runner = object.__new__(NPUModelRunner310)
    runner.device = torch.device("cpu")
    runner.supports_compact_mamba_state = True
    runner.supports_prefix_mamba_state_tier = True
    states = torch.zeros((4, 2), dtype=torch.float16)
    runner._prefix_mamba_tiers = {1: PrefixMambaStateTier([(states,)], 4)}
    attention_gpu = torch.tensor([[17, 18, 0]], dtype=torch.int32)
    mamba_cpu = np.array([[0, 101, 102]], dtype=np.int32)
    mamba_gpu = torch.tensor([[0, 101, 102]], dtype=torch.int32)
    runner.input_batch = SimpleNamespace(
        block_table=SimpleNamespace(
            block_tables=[
                SimpleNamespace(is_mamba_group=False, block_table=SimpleNamespace(gpu=attention_gpu)),
                SimpleNamespace(
                    is_mamba_group=True,
                    num_blocks_per_row=np.array([3]),
                    block_table=SimpleNamespace(np=mamba_cpu, gpu=mamba_gpu),
                ),
            ]
        )
    )

    runner._remap_compact_mamba_block_tables(num_reqs=1)

    torch.testing.assert_close(attention_gpu, torch.tensor([[17, 18, 0]], dtype=torch.int32))
    torch.testing.assert_close(mamba_gpu, torch.tensor([[0, 1, 2]], dtype=torch.int32))
    np.testing.assert_array_equal(mamba_cpu, [[0, 101, 102]])


def test_prefix_mamba_fresh_ids_exclude_cached_prefix() -> None:
    runner = object.__new__(NPUModelRunner310)
    runner._prefix_mamba_tiers = {1: object()}
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=object()),
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=128)),
        ]
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(num_computed_tokens=256, block_ids=([7, 8, 9], [0, 101, 102]))],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["ongoing"],
            resumed_req_ids=set(),
            new_block_ids=[([10], [103])],
            num_computed_tokens=[384],
        ),
    )

    assert runner._new_prefix_mamba_block_ids(scheduler_output) == {1: {102, 103}}


def test_prefix_mamba_precopy_uses_slots_then_restores_scheduler_ids() -> None:
    runner = object.__new__(NPUModelRunner310)
    states = torch.zeros((4, 2), dtype=torch.float16)
    tier = PrefixMambaStateTier([(states,)], 4)
    tier.remap_table(np.array([[0, 101, 102]], dtype=np.int32), 3)
    runner._prefix_mamba_tiers = {1: tier}
    runner._prefix_mamba_active_columns = {1: (0, 1, 2)}
    original = ([17, 18, 19], [0, 101, 102])
    req_state = SimpleNamespace(block_ids=original)
    runner.requests = {"request": req_state}
    runner.input_batch = SimpleNamespace(
        req_ids=["request"],
        block_table=SimpleNamespace(
            block_tables=[SimpleNamespace(is_mamba_group=False), SimpleNamespace(is_mamba_group=True)]
        ),
    )

    runner._stage_prefix_mamba_request_ids()
    assert req_state.block_ids == ([17, 18, 19], [0, 1, 2])
    assert req_state.block_ids is not original

    runner._restore_prefix_mamba_request_ids()
    assert req_state.block_ids is original


def test_prefix_mamba_stages_only_live_tail_at_full_context() -> None:
    runner = object.__new__(NPUModelRunner310)
    runner.device = torch.device("cpu")
    runner.supports_compact_mamba_state = True
    runner.supports_prefix_mamba_state_tier = True
    states = torch.zeros((4, 2), dtype=torch.float16)
    runner._prefix_mamba_tiers = {1: PrefixMambaStateTier([(states,)], 4)}
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(),
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=128, num_speculative_blocks=0)),
        ]
    )
    mamba_cpu = np.arange(1, 1025, dtype=np.int32).reshape(1, -1)
    mamba_gpu = torch.from_numpy(mamba_cpu.copy())
    raw_ids = ([17], mamba_cpu[0].tolist())
    req_state = SimpleNamespace(block_ids=raw_ids)
    runner.requests = {"request": req_state}
    runner.input_batch = SimpleNamespace(
        req_ids=["request"],
        num_computed_tokens_cpu=np.array([130944], dtype=np.int32),
        block_table=SimpleNamespace(
            block_tables=[
                SimpleNamespace(is_mamba_group=False),
                SimpleNamespace(
                    is_mamba_group=True,
                    num_blocks_per_row=np.array([1024]),
                    block_table=SimpleNamespace(np=mamba_cpu, gpu=mamba_gpu),
                ),
            ]
        ),
    )

    runner._remap_compact_mamba_block_tables(num_reqs=1, num_scheduled_tokens=np.array([128]))
    assert torch.count_nonzero(mamba_gpu).item() == 2
    assert mamba_gpu[0, -2:].tolist() == [1, 2]
    np.testing.assert_array_equal(mamba_cpu, np.arange(1, 1025, dtype=np.int32).reshape(1, -1))

    runner._stage_prefix_mamba_request_ids()
    assert req_state.block_ids[1][-2:] == [1, 2]
    assert all(block_id == 0 for block_id in req_state.block_ids[1][:-2])
    runner._restore_prefix_mamba_request_ids()
    assert req_state.block_ids is raw_ids


def test_prefix_mamba_stages_cached_checkpoint_across_long_chunk() -> None:
    runner = object.__new__(NPUModelRunner310)
    runner.device = torch.device("cpu")
    runner.supports_compact_mamba_state = True
    runner.supports_prefix_mamba_state_tier = True
    states = torch.zeros((4, 2), dtype=torch.float16)
    runner._prefix_mamba_tiers = {1: PrefixMambaStateTier([(states,)], 4)}
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(),
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=512, num_speculative_blocks=0)),
        ]
    )
    raw_table = np.arange(1, 55, dtype=np.int32).reshape(1, -1)
    staged_table = torch.from_numpy(raw_table.copy())
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array([23424], dtype=np.int32),
        block_table=SimpleNamespace(
            block_tables=[
                SimpleNamespace(is_mamba_group=False),
                SimpleNamespace(
                    is_mamba_group=True,
                    num_blocks_per_row=np.array([54]),
                    block_table=SimpleNamespace(np=raw_table, gpu=staged_table),
                ),
            ]
        ),
    )

    runner._remap_compact_mamba_block_tables(num_reqs=1, num_scheduled_tokens=np.array([3783]))

    assert runner._prefix_mamba_active_columns == {1: (45, 53)}
    assert torch.count_nonzero(staged_table).item() == 2
    assert staged_table[0, 45] != staged_table[0, 53]
    np.testing.assert_array_equal(raw_table, np.arange(1, 55, dtype=np.int32).reshape(1, -1))


class TestNPUModelRunner310(TestBase):
    def test_may_reinitialize_input_batch_expands_prefix_mamba_block_table(self):
        runner = object.__new__(NPUModelRunner310)
        runner.max_num_reqs = 8
        runner.max_model_len = 512
        runner.max_encoder_len = 0
        runner.max_num_tokens = 1024
        runner.device = torch.device("cpu")
        runner.pin_memory = False
        runner.is_pooling_model = False
        runner.model_config = SimpleNamespace(max_model_len=512, get_vocab_size=lambda: 32000)
        runner.cache_config = SimpleNamespace(block_size=128, enable_prefix_caching=True)
        runner.parallel_config = SimpleNamespace(cp_kv_cache_interleave_size=4)
        runner.vllm_config = SimpleNamespace(speculative_config=None)
        runner.offload_config = SimpleNamespace(uva=SimpleNamespace(cpu_offload_gb=0))
        runner.input_batch = SimpleNamespace(logitsprocs=MagicMock())
        attention_backend = SimpleNamespace(get_supported_kernel_block_sizes=lambda: [128, 64])
        runner.attn_groups = [[SimpleNamespace(backend=attention_backend)]]

        attention_spec = AttentionSpec(
            block_size=128,
            num_kv_heads=2,
            head_size=64,
            dtype=torch.float16,
        )
        mamba_spec = MambaSpec(
            block_size=128,
            shapes=((16,),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
            num_speculative_blocks=2,
        )
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=attention_spec),
                SimpleNamespace(kv_cache_spec=mamba_spec),
            ]
        )

        with (
            patch("vllm_ascend._310p.model_runner_310p.NPUInputBatch") as mock_input_batch,
            patch(
                "vllm_ascend._310p.model_runner_310p.get_decode_context_model_parallel_world_size",
                return_value=1,
            ),
        ):
            runner.may_reinitialize_input_batch(kv_cache_config)

        kwargs = mock_input_batch.call_args.kwargs
        self.assertEqual(kwargs["block_sizes"], [128, 128])
        self.assertEqual(kwargs["kernel_block_sizes"], [[128, 64], [0]])
        self.assertEqual(kwargs["max_num_blocks_per_req"], [4, 6])
        self.assertIs(kwargs["kv_cache_groups"], kv_cache_config.kv_cache_groups)
        self.assertEqual(kwargs["cp_kv_cache_interleave_size"], 4)
