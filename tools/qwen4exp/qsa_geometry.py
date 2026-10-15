# SPDX-License-Identifier: Apache-2.0
"""Bounded INT32 QSA geometry experiment; private graph qualification required."""

import hashlib
import json
import time
from pathlib import Path

import torch

NAMESPACE = "qwen_qsa_geometry_v1"
ROWS = (3, 6)
PADDED_ROWS = 8
RATIO = 4
MAX_CONFIGS = 2
REQUIRED_SOURCES = frozenset(
    (
        "compile_reconstruction.cpp",
        "native_qsa_geometry.cpp",
        "qsa_geometry_integer.h",
        "native_owned_bridge.cpp",
        "shared_launch_arguments.h",
    )
)
TEST_POSITIONS = (
    -2147483648,
    -2147483647,
    -9,
    -5,
    -4,
    -3,
    -2,
    -1,
    0,
    1,
    2,
    3,
    2050,
    2051,
    2052,
    262143,
    262144,
    2147483647,
)
TEST_CAPACITIES = (0, 1, 512, 65536)


def _is_capturing():
    return torch.npu.is_current_stream_capturing()


def _on_npu(device):
    return device.type == "npu"


def verify_build(build):
    root = Path(build).resolve(strict=True)
    report = json.loads((root / "provenance.json").read_text())
    if (
        report.get("namespace") != NAMESPACE
        or report.get("entry") != NAMESPACE
        or report.get("bridge_source") != "native_owned_bridge.cpp"
        or report.get("flush_host_queue") is not False
        or report.get("eager_only") is not False
    ):
        raise ValueError("geometry requires its isolated graph-compatible no-drain owned bridge")
    if set(report["sources"]) != REQUIRED_SOURCES or set(report["binaries"]) != {"native_qsa_geometry"}:
        raise ValueError("geometry frozen source/binary set differs")
    entries = [(root / name, digest) for name, digest in report["sources"].items()]
    entries += [
        (Path(entry["path"]), entry["sha256"])
        for entry in (report["bridge"], report["binaries"]["native_qsa_geometry"])
    ]
    for path, digest in entries:
        if path.resolve().parent != root or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("geometry artifact path or digest differs")
    return report


def eligible(positions, compress_ratio):
    return (
        isinstance(positions, torch.Tensor)
        and positions.ndim == 1
        and positions.shape[0] in ROWS
        and positions.dtype == torch.int32
        and positions.is_contiguous()
        and _on_npu(positions.device)
        and type(compress_ratio) is int
        and compress_ratio == RATIO
    )


def reference(positions, compress_ratio, capacity):
    next_positions = positions + 1
    groups = torch.div(next_positions, compress_ratio, rounding_mode="floor")
    starts = groups * compress_ratio
    return groups.clamp_max(capacity), starts, next_positions - starts


class QSAGeometry:
    """Own both descriptors and failed tensor frames for one worker device."""

    def __init__(self, kernel, launch):
        self.kernel, self.launch = kernel, launch
        self._configs = {}
        self.device = None
        self.failed = False
        self.failure = self.failure_traceback = None

    def prewarm(self, device):
        if self.failed or _is_capturing():
            raise RuntimeError("geometry prewarm requires healthy eager resources")
        device = torch.device(device)
        if self.device is not None and device != self.device:
            raise ValueError("geometry owner belongs to one device")
        self.device = device
        try:
            for rows in ROWS:
                key = (rows, device)
                if key not in self._configs:
                    self._configs[key] = torch.tensor([rows], dtype=torch.int64, device=device)
        except BaseException as error:
            self._retain_failure(error)
            raise

    def _retain_failure(self, error):
        if not self.failed:
            self.failed = True
            self.failure, self.failure_traceback = error, error.__traceback__

    def __call__(self, positions, compress_ratio, capacity):
        if self.failed:
            raise RuntimeError("geometry failed; retain resources for inspection")
        if not eligible(positions, compress_ratio) or type(capacity) is not int:
            raise ValueError("geometry requires INT32 contiguous N3/6 positions, ratio4 and integer capacity")
        key = (positions.shape[0], positions.device)
        if key not in self._configs:
            raise RuntimeError("prewarm geometry descriptors before execution or capture")
        try:
            packed = torch.empty((3, PADDED_ROWS), dtype=torch.int32, device=positions.device)
            self.launch(self.kernel, [positions, packed, self._configs[key]], 1)
            groups, starts, counts = (packed[row, : positions.shape[0]] for row in range(3))
            return groups.clamp_max(capacity), starts, counts
        except BaseException as error:
            self._retain_failure(error)
            raise


def load_resources(build, *, owner):
    if owner:
        raise ValueError("geometry native owner must begin empty")
    report = verify_build(build)
    torch.ops.load_library(report["bridge"]["path"])
    owner["graph_compatible"] = True
    kernel = getattr(torch.classes, NAMESPACE).Kernel(report["binaries"]["native_qsa_geometry"]["path"], NAMESPACE)
    owner["kernel"] = kernel
    owner["geometry"] = QSAGeometry(kernel, getattr(torch.ops, NAMESPACE).launch)
    return owner


def _equal(actual, expected):
    return all(torch.equal(left.cpu(), right.cpu()) for left, right in zip(actual, expected, strict=True))


def validate_graphs(resource, device, *, owner, repeats=3, replays=20):
    """Private tensors only; publish every graph before capture and retain failures.

    The caller must pause/drain serving first and retain ``owner`` on its worker.
    Failed graphs are deliberately not reset here; no automatic retry or unload.
    Successful cleanup synchronizes before resetting these private graphs only.
    """
    if (
        owner
        or type(repeats) is not int
        or not 1 <= repeats <= 5
        or type(replays) is not int
        or not 1 <= replays <= 100
    ):
        raise ValueError("geometry gate requires fresh owner and bounded repeats/replays")
    if _is_capturing():
        raise RuntimeError("private geometry gate cannot run inside capture")
    owner.update(failed=True, graphs=[], tensors=[], outputs=[], receipt=None)
    try:
        resource.prewarm(device)
        caller = torch.npu.current_stream(device)
        stream = torch.npu.Stream(device=device)
        owner["stream"] = stream
        stream.wait_stream(caller)
        torch.npu.synchronize(device)
        checks, timings = [], []
        with torch.inference_mode(), torch.npu.stream(stream):
            for rows in ROWS:
                positions = torch.zeros(rows, dtype=torch.int32, device=device)
                owner["tensors"].append(positions)
                graphs, outputs = {}, {}
                # Warm up both call paths and descriptors before capture.
                for function in (reference, resource):
                    owner["outputs"].append(function(positions, RATIO, 512))
                torch.npu.synchronize(device)
                for name, function in (("reference", reference), ("native", resource)):
                    graph = torch.npu.NPUGraph()
                    owner["graphs"].append(graph)
                    with torch.npu.graph(graph, stream=stream):
                        output = function(positions, RATIO, 512)
                    outputs[name], graphs[name] = output, graph
                    owner["outputs"].append(output)
                for offset in range(0, len(TEST_POSITIONS), rows):
                    values = TEST_POSITIONS[offset : offset + rows]
                    source = torch.tensor(values, dtype=torch.int32)
                    positions.copy_(source)
                    for graph in graphs.values():
                        graph.replay()
                    torch.npu.synchronize(device)
                    expected = reference(source, RATIO, 512)
                    exact = all(_equal(value, expected) for value in outputs.values())
                    preserved = torch.equal(positions.cpu(), source)
                    checks.append(dict(rows=rows, offset=offset, captured_exact=exact, inputs_preserved=preserved))
                    for capacity in TEST_CAPACITIES:
                        result = resource(positions, RATIO, capacity)
                        owner["outputs"].append(result)
                        torch.npu.synchronize(device)
                        checks.append(
                            dict(
                                rows=rows,
                                offset=offset,
                                capacity=capacity,
                                eager_exact=_equal(result, reference(source, RATIO, capacity)),
                            )
                        )
                positions.zero_()
                torch.npu.synchronize(device)
                for name, graph in graphs.items():
                    elapsed = []
                    for _ in range(repeats):
                        start = time.perf_counter()
                        for _ in range(replays):
                            graph.replay()
                        torch.npu.synchronize(device)
                        elapsed.append((time.perf_counter() - start) * 1000 / replays)
                    timings.append(dict(rows=rows, mode=name, wall_ms_per_replay=elapsed))
        passed = all(
            row.get("captured_exact", True) and row.get("eager_exact", True) and row.get("inputs_preserved", True)
            for row in checks
        )
        receipt = dict(
            passed=passed,
            cases=checks,
            timings=timings,
            ratio=RATIO,
            rows=list(ROWS),
            scope="private geometry integer parity and graph replay; not full-model qualification",
        )
        owner["receipt"] = receipt
        if not passed:
            return receipt
        torch.npu.synchronize(device)
        for graph in owner["graphs"]:
            graph.reset()
        owner["failed"] = False
        owner["retired"] = True
        return receipt
    except BaseException as error:
        owner.setdefault("failure", error)
        owner.setdefault("failure_traceback", error.__traceback__)
        raise
