# SPDX-License-Identifier: Apache-2.0
"""Private eager HC combine: preserve separate FP32 multiply/add and FP16 cast."""

import hashlib
import json
import time
from pathlib import Path

import torch

NAMESPACE = "qwen_hc_combine_v1"
STREAMS = 4
WIDTH = 2560
MAX_ROWS = 2560
MAX_CONFIGS = 128
BLOCKS = 8
MIN_VALIDATION_CASES = 3
REQUIRED_SOURCES = frozenset(("native_hc_combine.cpp", "native_owned_bridge.cpp", "shared_launch_arguments.h"))


def _on_npu(device):
    return device.type == "npu"


def _is_capturing():
    return torch.npu.is_current_stream_capturing()


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_build(build):
    root = Path(build)
    provenance = json.loads((root / "provenance.json").read_text())
    if (
        provenance.get("namespace") != NAMESPACE
        or provenance.get("bridge_source") != "native_owned_bridge.cpp"
        or provenance.get("flush_host_queue") is not True
        or provenance.get("eager_only") is not True
    ):
        raise ValueError("HC combine requires the explicit owned, queue-draining eager bridge")
    if not REQUIRED_SOURCES.issubset(provenance["sources"]):
        raise ValueError("HC combine frozen native and ownership sources are required")
    if set(provenance["binaries"]) != {"native_hc_combine"}:
        raise ValueError("HC combine requires exactly its private native binary")
    for name, checksum in provenance["sources"].items():
        if Path(name).name != name or _digest(root / name) != checksum:
            raise ValueError("HC combine frozen source digest differs")
    for entry in (provenance["bridge"], provenance["binaries"]["native_hc_combine"]):
        path = Path(entry["path"])
        if not path.is_absolute() or path.resolve().parent != root.resolve() or _digest(path) != entry["sha256"]:
            raise ValueError("HC combine native artifact path or digest differs")
    return provenance


def eligible(residual, block, injection):
    if not all(isinstance(value, torch.Tensor) for value in (residual, block, injection)):
        return False
    if residual.ndim != 3 or not 0 < residual.shape[0] <= MAX_ROWS:
        return False
    rows = residual.shape[0]
    return (
        residual.shape == (rows, STREAMS, WIDTH)
        and block.shape == (rows, WIDTH)
        and injection.shape == (rows, STREAMS)
        and residual.dtype == torch.float16
        and block.dtype == injection.dtype == torch.float32
        and _on_npu(residual.device)
        and all(value.device == residual.device and value.is_contiguous() for value in (residual, block, injection))
    )


class HCCombine:
    """Append-only descriptors and strong failure ownership; never graph safe."""

    def __init__(self, kernel, launch):
        self.kernel, self.launch = kernel, launch
        self._configs = {}
        self.failed = False
        self.failure = None
        self.failure_traceback = None

    def __call__(self, residual, block, injection):
        if self.failed:
            raise RuntimeError("HC combine failed; retain its native resources and tensor frames")
        if not eligible(residual, block, injection):
            raise ValueError("HC combine requires contiguous FP16 residual and FP32 block/gates on one NPU")
        if _is_capturing():
            raise RuntimeError("queue-draining HC combine is eager-only, including cached shapes")
        key = (residual.shape[0], residual.device)
        if key not in self._configs and len(self._configs) >= MAX_CONFIGS:
            raise RuntimeError("HC combine descriptor capacity reached; never evict live descriptors")
        try:
            config = self._configs.get(key)
            if config is None:
                config = torch.tensor([key[0]], dtype=torch.int64, device=residual.device)
                self._configs[key] = config
            output = torch.empty(residual.shape, dtype=torch.float16, device=residual.device)
            self.launch(self.kernel, [residual, block, injection, output, config], BLOCKS)
            return output
        except BaseException as error:
            self.failed = True
            self.failure, self.failure_traceback = error, error.__traceback__
            raise


def load_resources(build, *, owner):
    if owner:
        raise ValueError("HC combine native owner must begin empty")
    provenance = verify_build(build)
    torch.ops.load_library(provenance["bridge"]["path"])
    owner["eager_only"] = True
    kernel = getattr(torch.classes, NAMESPACE).Kernel(provenance["binaries"]["native_hc_combine"]["path"], NAMESPACE)
    # Publish each successful native constructor before any later operation.
    owner["kernel"] = kernel
    owner["combine"] = HCCombine(kernel, getattr(torch.ops, NAMESPACE).launch)
    return owner


def baseline(residual, block, injection):
    """Pinned epilogue after projection/sigmoid, with separate FP32 operations."""
    return (residual.float() + block.float().unsqueeze(1) * injection.float().unsqueeze(-1)).half()


def _private_inputs(rows, step, generator, device):
    residual = torch.randn(rows, STREAMS, WIDTH, dtype=torch.float16, generator=generator)
    block = torch.randn(rows, WIDTH, dtype=torch.float32, generator=generator)
    # Preserve the real model's already-rounded FP16 injection values.
    injection = (torch.rand(rows, STREAMS, generator=generator) * 2).half().float()
    if step == 1:
        residual[0, :, :8] = torch.tensor([-0.0, 0.0, 2**-24, -(2**-24), 65504, -65504, 1, -1]).half()
        block[0, :8] = torch.tensor([-0.0, 0.0, 2**-25, -(2**-25), 65504, -65504, -1, 1])
        injection[0] = torch.tensor([0.0, 0.5, 1.0, 1.75])
        # A fused multiply-add would leave a representable FP16 subnormal;
        # the qualified separate FP32 product rounds to 1 before adding -1.
        residual[0, 3, 8], block[0, 8] = -1.0, 4.0 / 7.0
    elif step == 2:
        # Exercise final cast overflow and exceptional arithmetic explicitly.
        residual[0, :, :4] = torch.tensor([float("inf"), -float("inf"), float("nan"), -0.0]).half()
        block[0, :4] = torch.tensor([-float("inf"), float("inf"), 1.0, float("nan")])
        injection[0] = torch.tensor([0.0, 0.5, 1.0, 2.0])
    return tuple(value.to(device) for value in (residual, block, injection))


def validate(resource, device, *, rows=(68, 640, 2048), repeats=3):
    if type(repeats) is not int or not 1 <= repeats <= 5:
        raise ValueError("HC combine repeats require 1..5")
    if not rows or any(type(count) is not int or not 0 < count <= MAX_ROWS for count in rows):
        raise ValueError("HC combine validation requires nonempty supported rows")
    generator = torch.Generator().manual_seed(571)
    records = []
    for count in rows:
        for step in range(max(repeats, MIN_VALIDATION_CASES)):
            residual, block, injection = _private_inputs(count, step, generator, device)
            # CPU copies allow an exact input-preservation check without doubling
            # device inputs. Validation is intentionally a cold, synchronized gate.
            originals = tuple(value.cpu().clone() for value in (residual, block, injection))
            expected = baseline(residual, block, injection)
            actual = resource(residual, block, injection)
            exact = torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            preserved = all(
                torch.equal(value.cpu().view(torch.uint8), old.view(torch.uint8))
                for value, old in zip((residual, block, injection), originals, strict=True)
            )
            records.append(dict(rows=count, step=step, exact=exact, inputs_preserved=preserved))
            del residual, block, injection, expected, actual, originals
    return dict(
        passed=all(row["exact"] and row["inputs_preserved"] for row in records),
        scope="private HC epilogue bitwise parity, not full-model qualification",
        cases=records,
    )


def timing(resource, device, *, rows, repeats):
    generator = torch.Generator().manual_seed(572)
    records = []
    for count in rows:
        residual, block, injection = _private_inputs(count, 0, generator, device)
        for name, function in (("qualified", baseline), ("fused", resource)):
            output = function(residual, block, injection)
            torch.npu.synchronize()
            del output
            elapsed = []
            for _ in range(repeats):
                start = time.perf_counter()
                output = function(residual, block, injection)
                torch.npu.synchronize()
                elapsed.append((time.perf_counter() - start) * 1000)
                del output
            records.append(dict(rows=count, mode=name, wall_ms=elapsed))
        del residual, block, injection
    return dict(passed=True, scope="synchronized eager component timing, not model throughput", records=records)
