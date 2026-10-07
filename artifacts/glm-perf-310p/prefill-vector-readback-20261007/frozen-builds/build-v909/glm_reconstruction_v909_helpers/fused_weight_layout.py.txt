# SPDX-License-Identifier: Apache-2.0
"""Lossless native Cube packing and reversible paused-worker bank preparation."""

import hashlib
import inspect
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .glm_int4 import packed_weight_bits

OUTPUT_TILE = 128
NZ_K = 256
CUBE_K = 64
NZ_N = 16
HOST_RESERVE_BYTES = 8 * 1024**3
MAX_PREPARE_THREADS = 4
PERMANENT_LAYOUT = "cube_n128_k256_v1"


def _geometry(packed, k):
    if packed.ndim != 3 or packed.dtype not in (torch.int8, torch.uint8) or not packed.is_contiguous():
        raise ValueError("weight layout requires contiguous byte [E,N,packedK]")
    e, n, _ = packed.shape
    bits = packed_weight_bits(k, packed.shape[-1])
    if e < 1 or n % OUTPUT_TILE or k % NZ_K:
        raise ValueError("weight layout requires complete 128x256 tiles")
    return e, n, bits


def _decode(words, bits, fields):
    values = torch.stack([(words >> (bits * field)) & ((1 << bits) - 1) for field in range(fields)], -2)
    sign = 1 << (bits - 1)
    return ((values + sign) % (1 << bits) - sign).to(torch.int8)


def unpack_nz(packed, k):
    e, n, bits = _geometry(packed, k)
    fields = 8 if bits == 3 else 8 // bits
    planes = 3 if bits == 3 else 1
    raw = packed.to(torch.int32).bitwise_and(255).reshape(e, n // NZ_N, k // NZ_K, planes, -1)
    words = raw[..., 0, :]
    for plane in range(1, planes):
        words = words | (raw[..., plane, :] << (8 * plane))
    values = _decode(words, bits, fields).reshape(e, n // NZ_N, k // NZ_K, NZ_K, NZ_N)
    return values.permute(0, 1, 4, 2, 3).reshape(e, n, k).contiguous()


def pack_cube(signed, bits):
    if signed.ndim != 3 or signed.dtype != torch.int8 or bits not in (2, 3, 4):
        raise ValueError("Cube packing requires signed INT8 [E,N,K], bits 2/3/4")
    e, n, k = signed.shape
    if e < 1 or n % OUTPUT_TILE or k % NZ_K:
        raise ValueError("Cube packing requires complete 128x256 tiles")
    values = signed.reshape(e, n // OUTPUT_TILE, OUTPUT_TILE, k // NZ_K, NZ_K // CUBE_K, CUBE_K)
    values = values.permute(0, 1, 3, 4, 2, 5)
    # Cast<int4b_t> and the existing L0B loader use even channels followed by odd.
    values = torch.cat((values[..., ::2, :], values[..., 1::2, :]), -2).contiguous()
    unsigned = values.to(torch.int32).bitwise_and((1 << bits) - 1)
    if bits == 4:
        pairs = unsigned.reshape(e, n // OUTPUT_TILE, k // NZ_K, -1, 2)
        packed = pairs[..., 0] | (pairs[..., 1] << 4)
    else:
        fields = 8 if bits == 3 else 4
        values = unsigned.reshape(e, n // OUTPUT_TILE, k // NZ_K, fields, -1)
        words = values[..., 0, :]
        for field in range(1, fields):
            words = words | (values[..., field, :] << (bits * field))
        packed = torch.stack([(words >> (8 * plane)) & 255 for plane in range(3 if bits == 3 else 1)], -2)
    return packed.to(torch.uint8).view(torch.int8).reshape(e, n, k * bits // 8).contiguous()


def unpack_cube(packed, k):
    e, n, bits = _geometry(packed, k)
    raw = packed.to(torch.int32).bitwise_and(255)
    if bits == 4:
        pairs = torch.stack((raw & 15, raw >> 4), -1)
        values = ((pairs + 8) % 16 - 8).to(torch.int8)
    else:
        planes = raw.reshape(e, n // OUTPUT_TILE, k // NZ_K, 3 if bits == 3 else 1, -1)
        words = planes[..., 0, :]
        if bits == 3:
            words = words | (planes[..., 1, :] << 8) | (planes[..., 2, :] << 16)
        values = _decode(words, bits, 8 if bits == 3 else 4)
    physical = values.reshape(e, n // OUTPUT_TILE, k // NZ_K, NZ_K // CUBE_K, OUTPUT_TILE, CUBE_K)
    logical = torch.stack((physical[..., : OUTPUT_TILE // 2, :], physical[..., OUTPUT_TILE // 2 :, :]), -2)
    logical = logical.reshape(e, n // OUTPUT_TILE, k // NZ_K, NZ_K // CUBE_K, OUTPUT_TILE, CUBE_K)
    return logical.permute(0, 1, 4, 2, 3, 5).reshape(e, n, k).contiguous()


def tensor_digest(cpu):
    return hashlib.sha256(memoryview(cpu.contiguous().numpy()).cast("B")).hexdigest()


def _byte_fields(planes, bits):
    fields = 8 if bits == 3 else 8 // bits
    decoded = []
    for field in range(fields):
        phase, plane = bits * field % 8, bits * field // 8
        low_bits = min(bits, 8 - phase)
        values = (planes[..., plane, :] >> phase) & ((1 << low_bits) - 1)
        if low_bits < bits:
            values = values | ((planes[..., plane + 1, :] & ((1 << (bits - low_bits)) - 1)) << low_bits)
        decoded.append(values)
    return np.stack(decoded, -2)


def _byte_planes(values, bits):
    fields = 8 if bits == 3 else 8 // bits
    values = values.reshape(*values.shape[:-1], fields, -1)
    output = np.zeros((*values.shape[:-2], 3 if bits == 3 else 1, values.shape[-1]), dtype=np.uint8)
    for field in range(fields):
        phase, plane = bits * field % 8, bits * field // 8
        low_bits = min(bits, 8 - phase)
        output[..., plane, :] |= (values[..., field, :] & ((1 << low_bits) - 1)) << phase
        if low_bits < bits:
            output[..., plane + 1, :] |= values[..., field, :] >> low_bits
    return output


def convert_host_bytes(packed, k, *, to_cube):
    """Reorder byte fields directly on CPU; avoid per-expert NPU tensor graphs.

    The inverse follows the byte layout independently of the signed-code Torch
    reference. Preparation verifies every original byte before any device write.
    """
    e, n, bits = _geometry(packed, k)
    if packed.device.type != "cpu":
        raise ValueError("host byte conversion requires CPU storage")
    raw = packed.numpy().view(np.uint8)
    planes = 3 if bits == 3 else 1
    if to_cube:
        codes = _byte_fields(raw.reshape(e, n // NZ_N, k // NZ_K, planes, -1), bits)
        codes = codes.reshape(e, n // OUTPUT_TILE, OUTPUT_TILE // NZ_N, k // NZ_K, NZ_K // CUBE_K, CUBE_K, NZ_N)
        codes = codes.transpose(0, 1, 3, 4, 2, 6, 5).reshape(
            e, n // OUTPUT_TILE, k // NZ_K, NZ_K // CUBE_K, OUTPUT_TILE, CUBE_K
        )
        physical = np.concatenate((codes[..., ::2, :], codes[..., 1::2, :]), -2)
        if bits == 4:
            pairs = physical.reshape(e, n // OUTPUT_TILE, k // NZ_K, -1, 2)
            output = pairs[..., 0] | (pairs[..., 1] << 4)
        else:
            output = _byte_planes(physical.reshape(e, n // OUTPUT_TILE, k // NZ_K, -1), bits)
    else:
        if bits == 4:
            codes = np.stack((raw & 15, raw >> 4), -1)
        else:
            codes = _byte_fields(raw.reshape(e, n // OUTPUT_TILE, k // NZ_K, planes, -1), bits)
        physical = codes.reshape(e, n // OUTPUT_TILE, k // NZ_K, NZ_K // CUBE_K, OUTPUT_TILE, CUBE_K)
        logical = np.stack((physical[..., : OUTPUT_TILE // 2, :], physical[..., OUTPUT_TILE // 2 :, :]), -2)
        logical = logical.reshape(e, n // OUTPUT_TILE, k // NZ_K, NZ_K // CUBE_K, OUTPUT_TILE // NZ_N, NZ_N, CUBE_K)
        nz = logical.transpose(0, 1, 4, 2, 3, 6, 5).reshape(e, n // NZ_N, k // NZ_K, -1)
        output = _byte_planes(nz, bits)
    output = np.ascontiguousarray(output).reshape(packed.shape)
    return torch.from_numpy(output.view(np.int8 if packed.dtype == torch.int8 else np.uint8))


def available_host_bytes():
    rows = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(rows["MemAvailable"].split()[0]) * 1024


class PreparedWeightLayout:
    """Keep CPU byte backups until every bank has been restored and verified.

    Only called by paused worker capture/apply RPCs, never model execution.
    Bank shapes, storage addresses, codes and scales are preserved. No persistent
    second device bank is allocated; transform and verify one expert at a time.
    """

    def __init__(self, synchronize, world_size=1, host_available=available_host_bytes, report_path=None):
        self.synchronize = synchronize
        self.world_size = world_size
        self.host_available = host_available
        self.backups = []
        self.report_path = report_path
        self.history = []
        self.receipt = {"prepared": False, "banks": 0, "bytes": 0, "restored_exact": True}

    def prepare(self, models):
        if self.backups:
            if not self.receipt["prepared"]:
                raise RuntimeError("incomplete weight transaction; restore before capture")
            return
        banks, seen, layouts = [], set(), []
        for model in models:
            for module in model.modules():
                module = getattr(module, "w2_experts", module)
                for prefix in ("gate_up", "down"):
                    codes = getattr(module, prefix + "_packed_bank", None)
                    scales = getattr(module, prefix + "_scale_bank", None)
                    if codes is None or scales is None or codes.data_ptr() in seen:
                        continue
                    seen.add(codes.data_ptr())
                    k = scales.shape[-1] * 32
                    _geometry(codes, k)
                    banks.append((codes, k))
                    layouts.append(getattr(module, "native_weight_layout", None))
        if not banks:
            raise ValueError("no resident routed weight banks found")
        total = sum(codes.numel() for codes, _ in banks)
        if any(value is not None for value in layouts):
            if any(value != PERMANENT_LAYOUT for value in layouts):
                raise ValueError("cannot mix permanent native and legacy packed banks")
            self.receipt = {
                "prepared": True,
                "persistent": True,
                "banks": len(banks),
                "bytes": total,
                "backup_bytes": 0,
                "transformed_banks": 0,
                "restored_exact": True,
            }
            self._report()
            return
        if self.host_available() < self.world_size * total + HOST_RESERVE_BYTES:
            raise MemoryError("insufficient host memory for all-rank exact bank backups")
        self.synchronize()
        self.receipt = {"prepared": False, "banks": len(banks), "bytes": total, "restored_exact": False}
        self.receipt["bank_byte_hashes"] = []
        started = time.monotonic()
        try:
            for bank_index, (codes, k) in enumerate(banks):
                bank_started = time.monotonic()
                original = codes.detach().cpu().clone() if codes.device.type == "cpu" else codes.detach().cpu()
                # Register rollback before the first write, including partial-bank failures.
                self.backups.append((codes, original, tensor_digest(original)))
                self.receipt["bank_byte_hashes"].append({"original": self.backups[-1][2], "restored": None})
                prepared_bank = torch.empty_like(original)

                def prepare_expert(expert, original=original, k=k, prepared_bank=prepared_bank):
                    source = original[expert : expert + 1]
                    prepared = convert_host_bytes(source, k, to_cube=True)
                    inverse = convert_host_bytes(prepared, k, to_cube=False)
                    if not np.array_equal(inverse.numpy(), source.numpy()):
                        raise RuntimeError("lossless Cube packing verification failed")
                    np.copyto(prepared_bank[expert : expert + 1].numpy(), prepared.numpy())

                # NumPy releases the GIL. Independent experts use disjoint CPU
                # output slices; only the owning RPC thread writes device memory.
                with ThreadPoolExecutor(max_workers=MAX_PREPARE_THREADS) as pool:
                    for _ in pool.map(prepare_expert, range(codes.shape[0])):
                        pass
                expected = tensor_digest(prepared_bank)
                codes.copy_(prepared_bank)
                self.synchronize()
                if tensor_digest(codes.detach().cpu()) != expected:
                    raise RuntimeError("prepared device bank differs from verified CPU bytes")
                self.receipt["bank_byte_hashes"][-1]["prepared"] = expected
                self.receipt["banks_completed"] = bank_index + 1
                self.receipt["prepare_elapsed_s"] = time.monotonic() - started
                self.receipt["last_bank_elapsed_s"] = time.monotonic() - bank_started
                self._report()
            self.synchronize()
            self.receipt["prepared"] = True
            self._report()
        except Exception:
            self.restore()
            raise

    def restore(self):
        if self.receipt.get("persistent"):
            # The default model already consumes these on-disk native bytes.
            # A kernel rollback changes dispatch, never the checkpoint layout.
            self.receipt["prepared"] = False
            self._report()
            return
        self.synchronize()
        while self.backups:
            codes, original, digest = self.backups[-1]
            codes.copy_(original)
            self.synchronize()
            actual = tensor_digest(codes.detach().cpu())
            if actual != digest:
                raise RuntimeError("resident bank byte restoration failed; retain rollback backup")
            self.receipt["bank_byte_hashes"][len(self.backups) - 1]["restored"] = actual
            self.backups.pop()
        self.receipt["prepared"] = False
        self.receipt["restored_exact"] = True
        self._report()

    def _report(self):
        self.history.append(json.loads(json.dumps(self.receipt)))
        if self.report_path is not None:
            temporary = self.report_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.history, indent=2) + "\n")
            temporary.replace(self.report_path)

    def wrap_worker_hooks(self, changes, worker_type):
        return wrap_worker_layout(changes, self, worker_type)


def wrap_worker_layout(changes, layout, worker_type):
    result = dict(changes)
    prefix = "vllm_ascend._310p.worker_310p:NPUWorker310."
    capture = original_layout_hook(result.get(prefix + "resident_capture", worker_type.resident_capture), "capture")
    apply = original_layout_hook(result.get(prefix + "resident_apply", worker_type.resident_apply), "apply")
    status = result.get(prefix + "resident_status", worker_type.resident_status)

    def prepare_capture(self):
        try:
            if self._resident_session().graphs_dirty:
                layout.prepare([wrapper.runnable for wrapper in self._resident_wrappers()])
            return capture(self)
        except Exception as error:
            return layout_error(error)

    def restore_apply(self, generation):
        try:
            # Restore before PatchSession removes these hooks or builds baseline graphs.
            session = self._resident_session()
            current = session.current
            pending = session.pending
            if current is None or (
                current.generation != generation and (pending is None or pending[0].digest != current.digest)
            ):
                layout.restore()
            return apply(self, generation)
        except Exception as error:
            return layout_error(error)

    def layout_status(self):
        receipt = status(self)
        receipt["prepared_weight_layout"] = dict(layout.receipt)
        return receipt

    result[prefix + "resident_capture"] = prepare_capture
    result[prefix + "resident_apply"] = restore_apply
    result[prefix + "resident_status"] = layout_status
    prepare_capture.__glm_layout_original__ = capture
    restore_apply.__glm_layout_original__ = apply
    return result


def original_layout_hook(function, role):
    """Remove earlier preparation hooks when constructing a new transaction.

    Legacy frozen helpers predate the marker. Only their known wrappers are
    unwrapped through the named closure; arbitrary worker functions are kept.
    """
    while True:
        original = getattr(function, "__glm_layout_original__", None)
        if original is None and function.__module__.endswith(".fused_weight_layout"):
            expected = "prepare_capture" if role == "capture" else "restore_apply"
            if function.__name__ == expected:
                original = inspect.getclosurevars(function).nonlocals.get(role)
        if original is None or original is function:
            return function
        function = original


def layout_error(error):
    # Older resident worker extensions do not expose _resident_error. Returning
    # every rank's reply prevents queued RPC errors contaminating model output.
    return {"rank": torch.distributed.get_rank(), "pid": os.getpid(), "error": f"{type(error).__name__}: {error}"}


def repair_prepared_capture(worker_type, layout):
    """Paused recovery: capture existing prepared banks without a second layout.

    The native recovery manifest records this explicit temporary hook. It does
    not change kernel registrations, buffers, bytes, or the active candidate.
    """
    original = original_layout_hook(worker_type.resident_capture, "capture")

    def capture_prepared(self):
        try:
            if not layout.receipt["prepared"]:
                raise RuntimeError("capture recovery requires already prepared banks")
            return original(self)
        except Exception as error:
            return layout_error(error)

    capture_prepared.__glm_layout_original__ = original
    worker_type.resident_capture = capture_prepared
