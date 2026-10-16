# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp import route_finalize as candidate
from tools.qwen4exp.resident_candidates import qfn_route_finalize as binding


def make_resource(monkeypatch):
    monkeypatch.setattr(candidate, "_supports_device", lambda _: True)
    monkeypatch.setattr(candidate, "_is_capturing", lambda: False)
    calls = []

    def launch(kernel, tensors, blocks):
        routed, inverse, weights, output, config = tensors
        assert blocks == 8 and config.tolist() == [routed.shape[0]]
        calls.append(config)
        output.copy_(candidate.simulation(routed, inverse, weights))

    return candidate.RouteFinalizer(object(), launch), calls


@pytest.mark.parametrize("rows", [10, 650, 6400])
def test_exact_changed_routes_local_tail_and_reduction(monkeypatch, rows):
    resource, calls = make_resource(monkeypatch)
    result = candidate.validate(resource, "cpu", rows=(rows,))
    assert result["passed"]
    assert len(calls) == 3 and all(config is calls[0] for config in calls)


def test_source_inputs_unchanged_and_outputs_not_aliased(monkeypatch):
    resource, _ = make_resource(monkeypatch)
    routed = torch.randn(10, 2560).half()
    inverse = torch.arange(9, -1, -1)
    weights = torch.rand(10, 1)
    copies = [t.clone() for t in (routed, inverse, weights)]
    output = resource(routed, inverse, weights)
    assert all(torch.equal(t, copy) for t, copy in zip((routed, inverse, weights), copies))
    assert output.data_ptr() not in {t.data_ptr() for t in (routed, inverse, weights)}


@pytest.mark.parametrize("case", ["width", "rows", "dtype", "inverse", "weights", "contiguous", "device"])
def test_bad_contract_rejected_before_native_launch(monkeypatch, case):
    resource, calls = make_resource(monkeypatch)
    routed, inverse, weights = torch.zeros(10, 2560).half(), torch.arange(10), torch.ones(10, 1)
    if case == "width":
        routed = routed[:, :1280].contiguous()
    elif case == "rows":
        routed = routed[:9]
    elif case == "dtype":
        routed = routed.float()
    elif case == "inverse":
        inverse = inverse.int()
    elif case == "weights":
        weights = weights[:, 0]
    elif case == "contiguous":
        routed = torch.zeros(2560, 10).half().T
    else:
        monkeypatch.setattr(candidate, "_supports_device", lambda _: False)
    with pytest.raises(ValueError):
        resource(routed, inverse, weights)
    assert not calls


def test_capture_warmup_and_descriptor_bound(monkeypatch):
    resource, calls = make_resource(monkeypatch)
    args = (torch.zeros(10, 2560).half(), torch.arange(10), torch.ones(10, 1))
    monkeypatch.setattr(candidate, "_is_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="warm up"):
        resource(*args)
    monkeypatch.setattr(candidate, "_is_capturing", lambda: False)
    resource(*args)
    monkeypatch.setattr(candidate, "_is_capturing", lambda: True)
    resource(*args)
    assert calls[0] is calls[1]
    monkeypatch.setattr(candidate, "_is_capturing", lambda: False)
    monkeypatch.setattr(candidate, "MAX_CONFIGS", 1)
    with pytest.raises(RuntimeError, match="capacity"):
        resource(torch.zeros(20, 2560).half(), torch.arange(20), torch.ones(20, 1))


def test_exact_source_seam_and_dtype_fallback():
    source = (Path(__file__).parent / "fixtures/qfn_pinned_grouped_w4.txt").read_text()
    namespace = {"torch": torch}
    exec(source, namespace)
    original = namespace["_forward_grouped"]
    calls = []

    def old(self, *args):
        calls.append("original")
        return 77

    forward = binding.build_forward(old, lambda *args: None, source=source)
    module = SimpleNamespace(native_int4=False)
    assert forward(module, torch.zeros(1, 2560), None, None) == 77
    assert calls == ["original"]
    # Only the finalization block changes; the original sum remains unchanged.
    assert source.count(binding.ORIGINAL) == 1
    assert "reshape(tokens, self.top_k, hidden).sum(1)" in binding.REPLACEMENT
    assert "activation = (F.silu(gate) * up).to(self.params_dtype)" in source
    with pytest.raises(RuntimeError, match="exact qualified"):
        binding.build_forward(original, None, source=source.replace("sum(1)", "sum(0)"))


def test_build_rejects_newer_header_before_copy(tmp_path):
    from tools.qwen4exp.build_route_finalize import freeze_headers

    directory = tmp_path / "csrc/gmm/qwen_w4_a8_int4_matmul_v310/op_kernel"
    directory.mkdir(parents=True)
    (directory / "native_int4_schedule.h").write_text("newer ABI")
    destination = tmp_path / "frozen"
    destination.mkdir()
    with pytest.raises(ValueError, match="qualified prefill header"):
        freeze_headers(tmp_path, destination)
    assert not list(destination.iterdir())


@pytest.mark.parametrize(
    "config_state",
    [
        "room",
        "full_existing",
        "full_missing",
        "one_slot_two_missing",
        "capturing_missing",
        "eager_only_capturing",
        "eager_only",
    ],
)
@pytest.mark.parametrize("fuse_activation", [False, True, "packed"])
def test_qualified_loop_preserves_chunking_math_and_local_routes(monkeypatch, config_state, fuse_activation):
    resource, calls = make_resource(monkeypatch)
    source = (Path(__file__).parent / "fixtures/qfn_pinned_grouped_w4.txt").read_text()

    def dispatch(weights, ids, **kwargs):
        rows = weights.numel()
        order = torch.arange(rows - 1, -1, -1)
        return SimpleNamespace(
            order=order,
            inverse_order=torch.argsort(order),
            token_indices=torch.arange(weights.shape[0]).repeat_interleave(10),
            route_weights=weights.reshape(-1, 1).float(),
            group_list=torch.tensor([rows // 2]),
        )

    class Bank:
        def __init__(self, width):
            self.width = width

        def grouped_linear(self, inputs, group_ends):
            result = torch.arange(inputs.shape[0] * self.width).reshape(-1, self.width).half() / 100
            result[int(group_ends[-1]) :] = 0
            return result

    namespace = {"torch": torch, "F": torch.nn.functional, "build_grouped_expert_dispatch": dispatch}
    exec(source, namespace)
    original = namespace["_forward_grouped"]
    activation_calls = []

    def activation(projected, compute_dtype, params_dtype):
        activation_calls.append(projected.shape[0])
        gate, up = projected.to(compute_dtype).chunk(2, -1)
        return (torch.nn.functional.silu(gate) * up).to(params_dtype)

    def packed_down(projected, projection, ends, compute_dtype, params_dtype):
        return projection.grouped_linear(activation(projected, compute_dtype, params_dtype), ends)

    forward = binding.build_forward(
        original,
        resource,
        source=source,
        eager_only=config_state.startswith("eager_only"),
        activation=activation if fuse_activation is True else None,
        packed_down=packed_down if fuse_activation == "packed" else None,
    )
    module = SimpleNamespace(
        native_int4=True,
        compute_dtype=torch.float32,
        params_dtype=torch.float16,
        top_k=10,
        grouped_chunk_tokens=2,
        num_local_experts=1,
        expert_offset=0,
        max_routed_rows=128,
        projections={"gate_up_proj": Bank(1280), "down_proj": Bank(2560)},
    )
    inputs, weights, ids = torch.ones(5, 2560).half(), torch.rand(5, 10), torch.zeros(5, 10).long()
    if config_state in ("full_existing", "eager_only_capturing"):
        monkeypatch.setattr(candidate, "MAX_CONFIGS", 2)
        resource._configs = {(rows, inputs.device): torch.tensor([rows]) for rows in (20, 10)}
        monkeypatch.setattr(candidate, "_is_capturing", lambda: True)
    elif config_state in ("full_missing", "one_slot_two_missing"):
        monkeypatch.setattr(candidate, "MAX_CONFIGS", 2)
        rows_to_cache = (30, 40) if config_state == "full_missing" else (30,)
        resource._configs = {(rows, inputs.device): torch.tensor([rows]) for rows in rows_to_cache}
    elif config_state == "capturing_missing":
        monkeypatch.setattr(candidate, "_is_capturing", lambda: True)
    prior_configs = resource._configs.copy()
    expected = original(module, inputs, weights, ids)
    actual = forward(module, inputs, weights, ids)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    if config_state in ("room", "full_existing", "eager_only"):
        assert [int(config[0]) for config in calls] == [20, 20, 10]
        assert activation_calls == ([20, 20, 10] if fuse_activation else [])
    else:
        assert not calls
        assert not activation_calls
        assert resource._configs == prior_configs


def test_build_requires_cpu_only_backend_selection(monkeypatch):
    from tools.qwen4exp.build_route_finalize import build

    monkeypatch.delenv("TORCH_DEVICE_BACKEND_AUTOLOAD", raising=False)
    with pytest.raises(RuntimeError, match="AUTOLOAD=0"):
        build(None, None, None, None, None)


def test_packed_down_rejects_two_activation_owners():
    source = (Path(__file__).parent / "fixtures/qfn_pinned_grouped_w4.txt").read_text()
    with pytest.raises(ValueError, match="own its retained activation fallback"):
        binding.build_forward(lambda: None, None, source=source, activation=object(), packed_down=object())
