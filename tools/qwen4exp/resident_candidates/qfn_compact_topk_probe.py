# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold private-score gate against the resident qualified top-k implementation.

Install this diagnostic RPC alone while inference is paused and drained, before
installing qsa_compact_topk. Calling qfn_cpu_profile_stop() without arguments runs
the gate; normal calls with an output path delegate to the original profiler.
No model weights, request caches, native graphs or collective owners are used.
"""

import hashlib
import inspect
import statistics
import time

import torch

from tools.qwen4exp.resident_candidates.qsa_compact_topk import compact_topk_indices

TARGET = "vllm_ascend._310p.worker_310p:NPUWorker310.qfn_cpu_profile_stop"
DEFAULT_ROWS = (64, 2048)
DEFAULT_WIDTHS = (1024, 2048, 8192)
DEFAULT_TOP_K = 512
TIMING_REPEATS = 3
TIMING_ITERATIONS = 3


def score_case(rows, width, pattern, *, generator):
    """Prepare independent CPU scores; large tied cutoffs preserve top-k ties."""
    if pattern == "random":
        return torch.randn((rows, width), generator=generator)
    if pattern == "ties":
        return torch.randint(0, 4, (rows, width), generator=generator).float()
    if pattern == "infinities":
        scores = torch.randn((rows, width), generator=generator)
        scores[:, width // 2 :] = -torch.inf
        scores[::2, : width // 2] = torch.inf
        scores[1::4] = -torch.inf
        return scores
    raise ValueError("unknown score pattern")


def assert_exact_indices(actual, expected):
    if actual.shape != expected.shape or actual.dtype not in (torch.int32, torch.int64):
        raise AssertionError("candidate index shape or dtype changed")
    # Deliberate cold diagnostics: readbacks never enter the serving hot path.
    if not torch.equal(actual.cpu().to(torch.int64), expected.cpu().to(torch.int64)):
        raise AssertionError("candidate changed selected index values or order")


def paired_timing(scores, k, original, candidate):
    functions = {"qualified": original, "compact": candidate}
    wall = {name: [] for name in functions}
    device = {name: [] for name in functions}
    for function in functions.values():
        function(scores, k)
    torch.npu.synchronize()
    for repeat in range(TIMING_REPEATS):
        order = tuple(functions) if repeat % 2 == 0 else tuple(reversed(functions))
        for name in order:
            begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
            torch.npu.synchronize()
            started = time.perf_counter()
            begin.record()
            for _ in range(TIMING_ITERATIONS):
                functions[name](scores, k)
            end.record()
            end.synchronize()
            wall[name].append((time.perf_counter() - started) * 1000 / TIMING_ITERATIONS)
            device[name].append(begin.elapsed_time(end) / TIMING_ITERATIONS)
    return {
        "wall_median_ms": {name: statistics.median(values) for name, values in wall.items()},
        "event_median_ms": {name: statistics.median(values) for name, values in device.items()},
        "wall_samples_ms": wall,
        "event_samples_ms": device,
    }


@torch.inference_mode()
def run_probe():
    # Resolve the function in the loaded qualified runtime, not a local copy.
    from vllm_ascend.models.qwen4_exp.ops import qsa_indexer

    original = qsa_indexer._fast_topk_indices
    source = inspect.getsource(original)
    if original.__name__ != "_fast_topk_indices" or "compact_indices" in source:
        raise RuntimeError("run the probe against the original qualified INT64 selector before candidate installation")
    if "310" not in torch.npu.get_device_name(torch.npu.current_device()):
        raise RuntimeError("requires the assigned resident Ascend 310P worker")
    device = f"npu:{torch.npu.current_device()}"
    generator = torch.Generator().manual_seed(314159)
    records = []
    torch.npu.synchronize()
    for rows in DEFAULT_ROWS:
        for width in DEFAULT_WIDTHS:
            for pattern in ("random", "ties", "infinities"):
                scores = score_case(rows, width, pattern, generator=generator).to(device)
                expected = original(scores, DEFAULT_TOP_K)
                actual = compact_topk_indices(scores, DEFAULT_TOP_K)
                assert_exact_indices(actual, expected)
                if expected.dtype != torch.int64 or actual.dtype != torch.int32:
                    raise AssertionError("gate did not compare qualified INT64 indices against compact INT32 indices")
                record = dict(rows=rows, width=width, k=DEFAULT_TOP_K, pattern=pattern, exact_indices=True)
                if pattern == "random":
                    record["timing"] = paired_timing(scores, DEFAULT_TOP_K, original, compact_topk_indices)
                records.append(record)
                # Only this shape's private tensors remain live on the device.
                del scores, expected, actual
    torch.npu.synchronize()
    return dict(
        scope="private_score_eager_selector_only_not_model_or_graph_qualification",
        passed=True,
        original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        score_math_changed=False,
        selection_width_changed=False,
        records=records,
    )


def make_rpc(original_stop, *, probe=run_probe):
    def qfn_cpu_profile_stop(self, output=None):
        if output is not None:
            return original_stop(self, output)
        try:
            if getattr(self, "_qfn_cpu_profile", None) is not None:
                raise RuntimeError("CPU profiling is active; stop it normally before the component gate")
            if getattr(self, "_qfn_native_profile_session", None) is not None:
                raise RuntimeError("native profiling is active; finish its session before the component gate")
            return dict(rank=self.rank, qsa_compact_topk=probe())
        except Exception as error:
            return dict(
                rank=getattr(self, "rank", None),
                qsa_compact_topk=dict(passed=False, error=f"{type(error).__name__}: {error}"),
            )

    return qfn_cpu_profile_stop


def replacements():
    # Worker isolation: preparation on a CPU host need not import torch_npu.
    from vllm_ascend._310p.worker_310p import NPUWorker310

    return {TARGET: make_rpc(NPUWorker310.qfn_cpu_profile_stop)}
