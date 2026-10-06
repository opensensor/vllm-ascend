# SPDX-License-Identifier: Apache-2.0
"""Lossless byte-layout conversion, preparation failures and exact rollback."""

from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf import fused_weight_layout as layout
from tools.glm_perf.glm_int4 import activation_limbs, pack_nz_codes, signed_nibbles


@pytest.mark.parametrize("bits", (2, 3, 4))
@pytest.mark.parametrize("shape", ((2, 128, 256), (1, 256, 512)))
def test_native_layout_roundtrip_and_cube_order(bits, shape):
    signed = torch.arange(torch.tensor(shape).prod().item()).remainder(1 << bits).sub(1 << (bits - 1))
    signed = signed.to(torch.int8).reshape(shape)
    original = pack_nz_codes(signed, bits)
    assert torch.equal(layout.unpack_nz(original, shape[-1]), signed)
    packed = layout.pack_cube(signed, bits)
    assert packed.shape == original.shape and packed.numel() == original.numel()
    assert torch.equal(layout.unpack_cube(packed, shape[-1]), signed)
    host = layout.convert_host_bytes(original, shape[-1], to_cube=True)
    assert torch.equal(host, packed)
    assert torch.equal(layout.convert_host_bytes(host, shape[-1], to_cube=False), original)
    if bits == 4:
        e, n, k = shape
        actual = signed_nibbles(packed).reshape(e, n // 128, k // 256, 4, 128, 64)
        channels = torch.cat((torch.arange(0, 128, 2), torch.arange(1, 128, 2)))
        logical = signed.reshape(e, n // 128, 128, k // 256, 4, 64).permute(0, 1, 3, 4, 2, 5)
        assert torch.equal(actual, logical[..., channels, :])


def bank_model():
    gate = torch.randint(-4, 4, (2, 512, 256), dtype=torch.int8)
    down = torch.randint(-8, 8, (2, 256, 256), dtype=torch.int8)
    banks = SimpleNamespace(
        gate_up_packed_bank=pack_nz_codes(gate, 3),
        gate_up_scale_bank=torch.ones(2, 16, 8),
        down_packed_bank=pack_nz_codes(down, 4),
        down_scale_bank=torch.ones(2, 8, 8),
    )
    module = SimpleNamespace(w2_experts=banks)
    return SimpleNamespace(modules=lambda: iter((module, module))), banks, gate, down


def test_preparation_preserves_storage_and_restores_all_bytes(tmp_path):
    model, banks, gate, down = bank_model()
    tensors = (banks.gate_up_packed_bank, banks.down_packed_bank)
    originals = [tensor.clone() for tensor in tensors]
    pointers = [tensor.data_ptr() for tensor in tensors]
    transaction = layout.PreparedWeightLayout(lambda: None, report_path=tmp_path / "rollback.json")
    transaction.prepare([model, model])
    assert transaction.receipt["prepared"] and transaction.receipt["banks"] == 2
    assert [tensor.data_ptr() for tensor in tensors] == pointers
    for tensor, signed in zip(tensors, (gate, down)):
        assert torch.equal(layout.unpack_cube(tensor, 256), signed)
    transaction.prepare([model])  # Retried capture never packs twice.
    transaction.restore()
    assert not transaction.backups and transaction.receipt["restored_exact"]
    assert all(torch.equal(tensor, original) for tensor, original in zip(tensors, originals))
    assert all(row["original"] == row["restored"] for row in transaction.receipt["bank_byte_hashes"])


def test_later_bank_preparation_failure_restores_already_written_bank(monkeypatch):
    model, banks, _, _ = bank_model()
    original = banks.gate_up_packed_bank.clone()
    convert = layout.convert_host_bytes
    calls = []

    def failing_convert(*args, **kwargs):
        calls.append(None)
        if len(calls) == 5:
            raise RuntimeError("injected preparation failure")
        return convert(*args, **kwargs)

    monkeypatch.setattr(layout, "convert_host_bytes", failing_convert)
    transaction = layout.PreparedWeightLayout(lambda: None)
    with pytest.raises(RuntimeError, match="injected"):
        transaction.prepare([model])
    assert torch.equal(banks.gate_up_packed_bank, original)
    assert transaction.receipt["restored_exact"] and not transaction.backups


def test_insufficient_host_memory_rejects_before_any_write():
    model, banks, _, _ = bank_model()
    original = banks.gate_up_packed_bank.clone()
    transaction = layout.PreparedWeightLayout(lambda: None, host_available=lambda: 0)
    with pytest.raises(MemoryError):
        transaction.prepare([model])
    assert not transaction.backups and torch.equal(banks.gate_up_packed_bank, original)


def test_failed_restore_keeps_backup_for_retry(monkeypatch):
    model, banks, _, _ = bank_model()
    original = banks.gate_up_packed_bank.clone()
    transaction = layout.PreparedWeightLayout(lambda: None)
    transaction.prepare([model])
    digest = layout.tensor_digest
    monkeypatch.setattr(layout, "tensor_digest", lambda cpu: "invalid readback")
    with pytest.raises(RuntimeError, match="retain rollback backup"):
        transaction.restore()
    assert len(transaction.backups) == 2 and not transaction.receipt["restored_exact"]
    monkeypatch.setattr(layout, "tensor_digest", digest)
    transaction.restore()
    assert torch.equal(banks.gate_up_packed_bank, original) and not transaction.backups


def test_apply_retry_and_mode_change_keep_prepared_bytes():
    restores = []
    transaction = SimpleNamespace(restore=lambda: restores.append(True), receipt={})
    worker_type = SimpleNamespace(
        resident_capture=lambda self: None,
        resident_apply=lambda self, generation: generation,
        resident_status=lambda self: {},
    )
    hooks = layout.wrap_worker_layout({}, transaction, worker_type)
    apply = hooks["vllm_ascend._310p.worker_310p:NPUWorker310.resident_apply"]
    current = SimpleNamespace(generation="a", digest="native")
    session = SimpleNamespace(current=current, pending=None)
    worker = SimpleNamespace(_resident_session=lambda: session)
    assert apply(worker, "a") == "a" and not restores
    session.pending = (SimpleNamespace(digest="native"), [])
    assert apply(worker, "b") == "b" and not restores
    session.pending = (SimpleNamespace(digest="baseline"), [])
    assert apply(worker, "c") == "c" and restores == [True]


@pytest.mark.parametrize("count", (1, 2, 3, 4, 7, 8, 15))
def test_sparse_int8_limbs_share_cube_fractal_and_preserve_bias(count):
    inputs = torch.linspace(-1, 1, count * 64).reshape(count, 64).half()
    low, high, _, quant = activation_limbs(inputs)
    weights = torch.arange(64).remainder(16).sub(8).int()
    sparse = count <= 7
    for group in (0, 1):
        a = torch.zeros(16 if sparse else 32, 64, dtype=torch.int32)
        selected = slice(group * 32, (group + 1) * 32)
        a[:count, selected] = low[:, group]
        high_row, bias_row = (count, 2 * count) if sparse else (16, 15)
        a[high_row : high_row + count, selected] = high[:, group]
        a[bias_row, selected] = 1
        products = a @ weights
        combined = products[:count] + 16 * products[high_row : high_row + count] + 8 * products[bias_row]
        assert torch.equal(combined, quant[:, group] @ weights[selected])


@pytest.mark.parametrize("activation_bits,count", ((8, 1), (8, 3), (4, 1), (4, 8)))
def test_two_scale_groups_share_one_cube_without_combining_scales(activation_bits, count):
    x = torch.linspace(-1, 1, count * 64).reshape(count, 64).half()
    low, high, _, quant = activation_limbs(x)
    if activation_bits == 4:
        grouped = x.float().reshape(count, 2, 32)
        quant = (grouped / (grouped.abs().amax(-1)[..., None] / 7)).round().clamp(-7, 7).int()
        low = quant
    weight = torch.arange(64).remainder(16).sub(8).int()
    packed = torch.zeros(16, 64, dtype=torch.int32)
    for group in (0, 1):
        base = group * (2 * count + 1 if activation_bits == 8 else count)
        selected = slice(group * 32, (group + 1) * 32)
        packed[base : base + count, selected] = low[:, group]
        if activation_bits == 8:
            packed[base + count : base + 2 * count, selected] = high[:, group]
            packed[base + 2 * count, selected] = 1
    cube = packed @ weight
    for group in (0, 1):
        base = group * (2 * count + 1 if activation_bits == 8 else count)
        actual = cube[base : base + count]
        if activation_bits == 8:
            actual = actual + 16 * cube[base + count : base + 2 * count] + 8 * cube[base + 2 * count]
        assert torch.equal(actual, quant[:, group] @ weight[group * 32 : (group + 1) * 32])


def test_permanent_layout_kernel_switch_never_transforms_or_backs_up(monkeypatch):
    model, banks, _, _ = bank_model()
    banks.native_weight_layout = layout.PERMANENT_LAYOUT
    monkeypatch.setattr(layout, "convert_host_bytes", lambda *args, **kwargs: pytest.fail("must not transform"))
    transaction = layout.PreparedWeightLayout(lambda: None, host_available=lambda: 0)
    transaction.prepare([model])
    assert transaction.receipt["persistent"] and transaction.receipt["backup_bytes"] == 0
    assert transaction.receipt["transformed_banks"] == 0 and not transaction.backups
    transaction.restore()
    assert not transaction.receipt["prepared"]
    transaction.prepare([model])
    assert transaction.receipt["prepared"] and not transaction.backups


def test_repeated_worker_wrapping_does_not_retain_previous_preparation():
    calls = []
    first = SimpleNamespace(prepare=lambda models: calls.append("first"), restore=lambda: None, receipt={})
    second = SimpleNamespace(prepare=lambda models: calls.append("second"), restore=lambda: None, receipt={})
    worker_type = SimpleNamespace(
        resident_capture=lambda self: calls.append("capture"),
        resident_apply=lambda self, generation: None,
        resident_status=lambda self: {},
    )
    hooks = layout.wrap_worker_layout({}, first, worker_type)
    worker_type.resident_capture = hooks["vllm_ascend._310p.worker_310p:NPUWorker310.resident_capture"]
    hooks = layout.wrap_worker_layout({}, second, worker_type)
    worker = SimpleNamespace(
        _resident_session=lambda: SimpleNamespace(graphs_dirty=True),
        _resident_wrappers=lambda: [SimpleNamespace(runnable=object())],
    )
    hooks["vllm_ascend._310p.worker_310p:NPUWorker310.resident_capture"](worker)
    assert calls == ["second", "capture"]
