# SPDX-License-Identifier: Apache-2.0
"""Qualify private TP6 state banks on six drained resident workers.

Does not replace serving methods or write live state banks. Native failures
remain fail-closed under the resident harness; inspect receipts before recovery.
"""

import argparse
import hashlib
import json
from pathlib import Path

from tools.glm_perf.resident_native import NativeManifest
from tools.sidecard.qfn_resident_server import QfnResidentClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing receipts")
    provenance = json.loads((args.build / "provenance.json").read_text())
    if provenance["namespace"] != "qwen_mamba_commit_v1" or provenance["flush_host_queue"] is not False:
        raise ValueError("requires the qualified graph-compatible state-commit bridge")
    source = """
import gc
import torch
from tools.qwen4exp.benchmark_mamba_commit_group_310 import gate
from tools.qwen4exp.runtime_geometry import state_geometry

def prepare():
    workers = [obj for obj in gc.get_objects() if type(obj).__name__ == 'NPUWorker310']
    if len(workers) != 1:
        raise RuntimeError('requires one resident NPU worker')
    runner = workers[0].model_runner
    if runner.execute_model_state is not None or runner.max_num_reqs != 6:
        raise RuntimeError('requires a drained six-request runner')
    group = runner._mamba_bufs.postprocess_align.mamba_group_ids[0]
    name = runner.kv_cache_config.kv_cache_groups[group].layer_names[0]
    states = runner.compilation_config.static_forward_context[name].kv_cache
    slots = state_geometry(*states)
    width, heads = states[0].shape[2], states[1].shape[1]
    records = []
    for pairs, pool_slots in ((36, 4), (1, slots)):
        gate(BUILD, str(runner.device), records.append, requests=6,
             width=width, heads=heads, pairs=pairs, slots=pool_slots)
    torch.npu.synchronize()
    torch.npu.empty_cache()
    return dict(records=records, slots=slots, width=width, heads=heads)

def validate(resource):
    completed = [row for row in resource['records'] if row.get('stage') == 'passed']
    if len(completed) != 2:
        raise RuntimeError('both full-bank and actual-slot gates must complete')
    return dict(passed=True, **resource)
"""
    source = "BUILD = " + repr(str(args.build.resolve(strict=True))) + "\n" + source
    modules = [
        Path(__file__).parent / name
        for name in (
            "benchmark_mamba_commit_group_310.py",
            "benchmark_mamba_state_commit_310.py",
            "mamba_commit.py",
            "mamba_commit_plan.py",
            "runtime_geometry.py",
        )
    ]
    assets = [dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in modules]
    manifest = NativeManifest(
        dict(
            name="qfn_tp6_private_state_gate_v1",
            libraries=[provenance["bridge"]],
            assets=[*provenance["binaries"].values(), *assets],
            operators=["qwen_mamba_commit_v1::launch"],
            validation_source=source,
        )
    )
    client = QfnResidentClient(args.url, args.run_root / "control.token", expected_workers=6, timeout=600)
    args.output.write_text(json.dumps(dict(manifest=manifest.value, passed=False), indent=2) + "\n")
    workers = client.load_native(manifest)
    args.output.write_text(json.dumps(dict(manifest=manifest.value, workers=workers, passed=True), indent=2) + "\n")
    print(json.dumps(dict(passed=True, workers=[dict(rank=w["rank"], validation=w["validation"]) for w in workers])))


if __name__ == "__main__":
    main()
