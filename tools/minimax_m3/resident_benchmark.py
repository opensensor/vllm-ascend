# SPDX-License-Identifier: Apache-2.0
"""Benchmark sealed native candidates against loaded banks, preserving KV cache."""

import argparse
import dataclasses
import fcntl
import inspect
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.resident_native import NativeManifest
from tools.minimax_m3.resident import candidate_source


def benchmark_replacements(native_resources, baseline, candidate, iterations, attention=False, decode_only=False):
    # Worker-isolated imports: this function's source is sent to each rank.
    import hashlib
    import statistics
    import time

    import torch

    from tools.minimax_m3.byte_reuse_probe import byte_reuse_probe
    from tools.minimax_m3.resident_attention_benchmark import benchmark_attention
    from vllm_ascend._310p.ops.minimax_m3 import PackedM3Experts

    original, proposed = native_resources[baseline], native_resources[candidate]

    def project(x, codes, scales, ends):
        return original(x, codes, scales, ends)

    def cache_identity(value):
        if isinstance(value, torch.Tensor):
            return (value.untyped_storage().data_ptr(), tuple(value.shape), str(value.dtype))
        if isinstance(value, dict):
            return tuple((key, cache_identity(child)) for key, child in sorted(value.items()))
        return tuple(cache_identity(child) for child in value)

    def capture(self):
        self._m3_idle()
        torch.npu.synchronize()
        before = cache_identity(self.model_runner.kv_caches)
        records = []
        try:
            bank = next(module for module in self.model_runner.model.modules() if isinstance(module, PackedM3Experts))
            experts = bank.num_local_experts
            with torch.inference_mode():
                if attention:
                    report = benchmark_attention(bank.w1_codes.device, iterations)
                else:
                    byte_probe = byte_reuse_probe(proposed, bank.w1_codes.device)
                    for projection in ("w1", "w3", "w2"):
                        codes = getattr(bank, projection + "_codes")
                        scales = getattr(bank, projection + "_scales").float()
                        cases = ((4, False), (4, True))
                        if not decode_only:
                            cases += ((512, False), (2048, False), (2048, True))
                        for rows, concentrated in cases:
                            boundaries = (
                                [rows] * experts
                                if concentrated
                                else [min(rows, (i + 1) * max(1, rows // experts)) for i in range(experts)]
                            )
                            x = (
                                torch.randn(rows, codes.shape[-1] * 2, generator=torch.Generator().manual_seed(53))
                                .half()
                                .to(codes.device)
                            )
                            ends = torch.tensor(boundaries, dtype=torch.int64, device=codes.device)
                            timings, reference, error = {}, None, None
                            for name, kernel in ((baseline, original), (candidate, proposed)):
                                kernel_scales = scales.to(dtype=getattr(kernel, "scale_dtype", torch.float32))
                                for _ in range(2):
                                    kernel(x, codes, kernel_scales, ends)
                                torch.npu.synchronize()
                                samples = []
                                for _ in range(iterations):
                                    start = time.perf_counter()
                                    result = kernel(x, codes, kernel_scales, ends)
                                    torch.npu.synchronize()
                                    samples.append(time.perf_counter() - start)
                                actual = result.cpu()
                                timings[name] = {"median_seconds": statistics.median(samples), "samples": samples}
                                if reference is None:
                                    reference = actual
                                else:
                                    error = float((actual.float() - reference.float()).abs().max())
                            records.append(
                                {
                                    "projection": projection,
                                    "rows": rows,
                                    "concentrated": concentrated,
                                    "timings": timings,
                                    "max_abs": error,
                                    "passed": error == 0.0,
                                }
                            )
                    del result, x, ends, actual, reference, scales
                    report = {
                        "passed": byte_probe["passed"] and all(row["passed"] for row in records),
                        "byte_reuse": byte_probe,
                        "layer": bank.prefix,
                        "cases": records,
                    }
        except Exception as exc:
            report = {"passed": False, "error": f"{type(exc).__name__}: {exc}", "cases": records}
        torch.npu.synchronize()
        after = cache_identity(self.model_runner.kv_caches)
        report["kv_storage_unchanged"] = before == after
        report["kv_storage_digest"] = hashlib.sha256(repr(after).encode()).hexdigest()
        free, total = torch.npu.mem_get_info()
        report["memory"] = {
            "free_bytes": free,
            "total_bytes": total,
            "allocated_bytes": torch.npu.memory_allocated(),
            "reserved_bytes": torch.npu.memory_reserved(),
        }
        return {**self.resident_status(), "benchmark": report}

    return {
        "vllm_ascend._310p.ops.minimax_m3:grouped_w4_mm": project,
        "vllm_ascend._310p.worker_310p:NPUWorker310.resident_capture": capture,
    }


def run_benchmark(client, manifest, baseline, iterations, attention=False, decode_only=False):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    before = client.rpc("resident_status")
    source = candidate_source(native_name=baseline)
    restore = Control(before[0]["generation"], "direct-target", before[0]["candidate"], source)
    if any(row["digest"] != restore.digest or row["native_failed"] or row["graphs_dirty"] for row in before):
        raise ValueError("benchmark requires the named native-only baseline on every rank")
    if client.request("/is_paused", method="GET")["is_paused"]:
        raise ValueError("refusing to take ownership of an already paused server")
    hook = (
        inspect.getsource(benchmark_replacements)
        + "\ndef replacements(native_resources):\n"
        + f"    return benchmark_replacements(native_resources, {baseline!r}, {manifest.name!r}, "
        + f"{iterations!r}, {attention!r}, {decode_only!r})\n"
    )
    control = Control(uuid.uuid4().hex, "direct-target", "projection_benchmark", hook)
    applied = False
    try:
        # New HTTP requests can queue while this bounded device benchmark runs.
        # Neither scheduler prefixes nor physical KV buffers are reset.
        client.request("/pause?mode=wait&clear_cache=false")
        prepared = client.rpc("resident_native_prepare", manifest.payload)
        if any(row["native_digest"] != manifest.digest for row in prepared):
            raise RuntimeError("native manifests disagree")
        loaded = client.rpc("resident_native_load", manifest.digest)
        if any(not row["validation"]["passed"] for row in loaded):
            raise RuntimeError("native validation failed")
        staged = client.rpc("resident_prepare", json.dumps(dataclasses.asdict(control)))
        if any(row["digest"] != control.digest for row in staged):
            raise RuntimeError("benchmark hooks disagree")
        applied = True
        client.rpc("resident_apply", control.generation)
        measured = client.rpc("resident_capture")
    finally:
        if applied:
            client.rpc("resident_prepare", json.dumps(dataclasses.asdict(restore)))
            client.rpc("resident_apply", restore.generation)
        after = client.rpc("resident_status")
        fields = ("rank", "pid", "generation", "digest", "weight_storage_digest")
        if sorted(tuple(row[key] for key in fields) for row in before) != sorted(
            tuple(row[key] for key in fields) for row in after
        ):
            raise RuntimeError("baseline restoration failed; leaving the server paused")
        if any(row["native_failed"] or row["graphs_dirty"] for row in after):
            raise RuntimeError("worker native state is unhealthy; leaving the server paused")
        client.resume()
    return {
        "scope": "Private attention caches"
        if attention
        else "Resident full expert banks; projection microbenchmark, not serving throughput",
        "decode_only": decode_only,
        "baseline": baseline,
        "candidate": manifest.name,
        "before": before,
        "after": after,
        "workers": measured,
        "passed": all(row["benchmark"]["passed"] and row["benchmark"]["kv_storage_unchanged"] for row in measured),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", default="minimax_w4_v7")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention", action="store_true", help="Compare attention on bounded private scratch caches")
    parser.add_argument("--decode-only", action="store_true", help="Time only four-route expert cases")
    args = parser.parse_args()
    client = ResidentClient(args.url, 6)
    with Path("/tmp/minimax-m31-resident.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = run_benchmark(
            client,
            NativeManifest(json.loads(args.manifest.read_text())),
            args.baseline,
            args.iterations,
            args.attention,
            args.decode_only,
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "workers": len(report["workers"])}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
