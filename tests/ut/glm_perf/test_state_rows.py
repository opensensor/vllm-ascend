# SPDX-License-Identifier: Apache-2.0
"""Gapped cache aliasing, selected writes, startup preparation and scoped rewrite."""

from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tools.glm_perf.build_state_rows import build
from tools.glm_perf.resident_candidates.kda_state_rows import extend_replacements, wrap_prefill
from tools.glm_perf.state_rows import NativeStateRows, cache_geometry


def _prefill_initial_state(cache, indices, flags):
    state = cache[indices].float().contiguous()
    state.masked_fill_(~flags.reshape(-1, 1, 1, 1), 0)
    return state


def _original_prefill(recurrent_state, state_indices, has_initial_state, adjustment):
    initial_state = _prefill_initial_state(recurrent_state, state_indices, has_initial_state)
    result = (initial_state, initial_state * 0.25 + adjustment)
    recurrent_state[state_indices] = result[1].to(recurrent_state.dtype)
    return result[0]


def _partial_prefill(recurrent_state, state_indices, has_initial_state):
    return _prefill_initial_state(recurrent_state, state_indices, has_initial_state)


@pytest.fixture
def native():
    copy = NativeStateRows.__new__(NativeStateRows)
    copy.device, copy.configs = torch.device("cpu"), {}
    copy.gather_kernel, copy.scatter_kernel = "gather", "scatter"
    copy.gathers = copy.scatters = 0

    def launch(kernel, args, blocks):
        flat, indices = args[:2]
        rows, stride, payload, selected, width = args[-1].tolist()
        assert blocks == 8 and flat.is_contiguous() and width == indices.element_size()
        for row in range(selected):
            slot = int(indices[row]) % rows
            if kernel == "gather":
                flags, output = args[2:4]
                output[row].copy_(flat[slot * stride : slot * stride + payload].reshape(output.shape[1:]))
                if not flags[row]:
                    output[row].zero_()
            else:
                flat[slot * stride : slot * stride + payload].copy_(args[2][row].flatten())

    copy.launch = launch
    return copy


def bank():
    backing = torch.randn(16 + 7 * 704 + 16).half()
    cache = backing.as_strided((7, 2, 16, 16), (704, 256, 16, 1), 16)
    return backing, cache


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_only_selected_rows_change_and_page_gaps_survive(native, dtype, monkeypatch):
    backing, cache = bank()
    before = backing.clone()
    geometry = native.prepare(cache)
    assert geometry.payload == 512 and geometry.span == 6 * 704 + 512
    assert cache.storage_offset() == 16
    ids, flags = torch.tensor([0, 2, -1, 3], dtype=dtype), torch.tensor([True, False, True, False])
    expected = _prefill_initial_state(cache, ids.long(), flags).half()
    values = torch.randn(4, 2, 16, 16).half()
    expected_backing = before.clone()
    expected_backing.as_strided(cache.shape, cache.stride(), cache.storage_offset())[ids.long()] = values
    # The hot path must consume startup descriptors, never create or transfer
    # a config tensor while processing each KDA layer.
    monkeypatch.setattr(torch, "tensor", lambda *args, **kwargs: pytest.fail("serving-time descriptor allocation"))
    assert torch.equal(native.gather(cache, ids, flags).view(torch.int16), expected.view(torch.int16))
    native.scatter(cache, ids, values)
    assert torch.equal(backing.view(torch.int16), expected_backing.view(torch.int16))
    assert native.gathers == native.scatters == 1


def test_scoped_rewrite_keeps_math_and_fp32_promotion(native):
    original_backing, original_cache = bank()
    changed_backing = original_backing.clone()
    changed_cache = changed_backing.as_strided(original_cache.shape, original_cache.stride(), 16)
    native.prepare(changed_cache)
    ids, flags = torch.tensor([0, 2, -1, 3]), torch.tensor([True, False, True, False])
    adjustment = torch.randn(4, 2, 16, 16)
    expected = _original_prefill(original_cache, ids, flags, adjustment)
    wrapped = wrap_prefill(_original_prefill, native)
    actual = wrapped(changed_cache, ids, flags, adjustment)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(changed_backing.view(torch.int16), original_backing.view(torch.int16))
    assert wrap_prefill(wrapped, native).__glm_state_rows_original__ is _original_prefill
    with pytest.raises(ValueError, match="partial replacement"):
        wrap_prefill(_partial_prefill, native)


def test_empty_selection_launches_nothing(native):
    _, cache = bank()
    native.prepare(cache)
    ids, flags = torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.bool)
    native.launch = lambda *args: pytest.fail("empty launch")
    output = native.gather(cache, ids, flags)
    native.scatter(cache, ids, output)
    assert output.shape == (0, 2, 16, 16)


def test_oversized_dummy_uses_original_path(native):
    _, cache = bank()
    ids, flags = torch.arange(5), torch.ones(5, dtype=torch.bool)
    adjustment = torch.zeros(5, 2, 16, 16)
    native.launch = lambda *args: pytest.fail("unprepared dummy must use original path")
    result = wrap_prefill(_original_prefill, native)(cache, ids, flags, adjustment)
    assert result.shape == (5, 2, 16, 16) and native.gathers == native.scatters == 0


def test_first_activation_prepares_before_capture(native, monkeypatch):
    import sys

    package = ModuleType("vllm_ascend.models.glm5next_w2")
    package.kda_310 = SimpleNamespace(_run_prefill=_original_prefill)
    monkeypatch.setitem(sys.modules, package.__name__, package)
    _, cache = bank()

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.A_log, self.kv_cache = object(), (torch.empty(1), cache)

    worker = SimpleNamespace(model_runner=SimpleNamespace(model=Layer()))
    worker._resident_error = lambda error: {"error": str(error)}

    def capture(self):
        assert (cache_geometry(cache), 4, torch.int64) in native.configs
        return {"captured": True}

    prefix = "vllm_ascend._310p.worker_310p:NPUWorker310."
    previous_apply = lambda self, generation: {"applied": True}
    changes = {
        prefix + "resident_apply": previous_apply,
        prefix + "resident_capture": capture,
        prefix + "resident_status": lambda self: {},
    }
    changed = extend_replacements(changes, native)
    assert not native.configs and changed[prefix + "resident_apply"] is previous_apply
    assert changed[prefix + "resident_capture"](worker) == {"captured": True}
    worker.model_runner.model = torch.nn.Module()
    assert "no bound KDA cache" in changed[prefix + "resident_capture"](worker)["error"]


def test_layout_and_unprepared_descriptor_fail_before_launch(native):
    _, cache = bank()
    ids, flags = torch.tensor([0]), torch.tensor([True])
    with pytest.raises(ValueError, match="not prepared"):
        native.gather(cache, ids, flags)
    with pytest.raises(ValueError, match="payload must be contiguous"):
        cache_geometry(cache.transpose(2, 3))
    with pytest.raises(ValueError, match="rank-four FP16"):
        cache_geometry(cache.float())
    with pytest.raises(ValueError, match="nonoverlapping"):
        cache_geometry(cache.as_strided(cache.shape, (256, 256, 16, 1)))
    native.prepare(cache)
    with pytest.raises(ValueError, match="INT32/INT64"):
        native.gather(cache, ids.float(), flags)
    with pytest.raises(ValueError, match="BOOL"):
        native.gather(cache, ids, flags.int())
    with pytest.raises(ValueError, match="writer shape"):
        native.scatter(cache, ids, torch.empty(1, 512).half())


def test_invalid_build_version_leaves_no_directory(tmp_path):
    with pytest.raises(ValueError, match="version must be positive"):
        build(tmp_path / "invalid", Path("/unused"), Path("/unused"), 0)
    assert not (tmp_path / "invalid").exists()
