# SPDX-License-Identifier: Apache-2.0
"""Read-only progress report for an isolated Qwen serving profile."""

import argparse
import json
import os
import time
from pathlib import Path


def worker_fields(line, marker):
    if marker not in line or "Worker_TP" not in line:
        return None
    rank = line.partition("Worker_TP")[2].partition("_EP")[0]
    if not rank.isdigit():
        return None
    fields = dict(field.split("=", 1) for field in line.partition(marker)[2].split() if "=" in field)
    return int(rank), fields


def summarize(log, *, layers=48, banks_per_layer=2):
    completed = {}
    progress = {}
    for line in log.splitlines():
        parsed = worker_fields(line, "bank_done ")
        if parsed:
            rank, fields = parsed
            completed.setdefault(rank, {})[(int(fields["layer"]), fields["bank"])] = float(fields["elapsed_s"])
        parsed = worker_fields(line, "CONVERSION layer=")
        if parsed:
            rank, fields = worker_fields(line, "CONVERSION ")
            done, total = fields["experts"].split("/")
            progress[rank] = {
                "layer": int(fields["layer"]),
                "bank": fields["bank"],
                "experts": int(done),
                "total": int(total),
            }
    if "EngineCore encountered a fatal error" in log or "WorkerProc hit an exception" in log:
        phase = "failed"
    elif "Application startup complete" in log:
        phase = "API started; resident correctness gates may still be running"
    elif "Capturing CUDA graphs" in log or "Graph capturing" in log:
        phase = "capturing decode graphs"
    elif "INT8_EXPERT_CONVERSION complete" in log:
        phase = "conversion complete on some ranks; waiting for all ranks, KV profiling, or graphs"
    elif "INT8_EXPERT_CONVERSION" in log:
        phase = "converting expert weights"
    elif "Loading safetensors checkpoint shards" in log:
        phase = "loading checkpoint"
    else:
        phase = "initializing workers and tokenizer"
    ranks = []
    for rank in sorted(set(completed) | set(progress)):
        durations = list(completed.get(rank, {}).values())
        count = len(durations)
        # Estimate conversion only. It excludes profiling, graph capture, and
        # inference gates and is deliberately absent until two banks finish.
        remaining = (
            ((layers * banks_per_layer - count) * sum(durations[-8:]) / len(durations[-8:])) if count >= 2 else None
        )
        ranks.append(
            {
                "rank": rank,
                "banks_done": count,
                "banks_total": layers * banks_per_layer,
                "current": progress.get(rank),
                "conversion_remaining_s": remaining,
            }
        )
    return {"phase": phase, "ranks": ranks}


def cache_progress(root, *, started_after, layers=48, banks_per_layer=2):
    """Count finalized banks; only new writes establish conversion progress."""
    report = []
    for rank in range(6):
        paths = list((root / f"rank-{rank}").glob("layer-*-*.safetensors"))
        times = sorted(path.stat().st_mtime for path in paths if path.stat().st_mtime >= started_after)
        remaining = None
        if len(times) >= 2 and times[-1] > times[0]:
            remaining = (layers * banks_per_layer - len(times)) * (times[-1] - times[0]) / (len(times) - 1)
        report.append(
            {
                "rank": rank,
                "cached_banks": len(paths),
                "new_banks": len(times),
                "banks_total": layers * banks_per_layer,
                "conversion_remaining_s": remaining,
            }
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", required=True, type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    while True:
        path = args.profile_root / "server-8001.log"
        report = summarize(path.read_text(errors="replace") if path.exists() else "")
        report["log"] = str(path)
        report["last_log_age_s"] = round(time.time() - path.stat().st_mtime, 1) if path.exists() else None
        pid_file = args.profile_root / "api.pid"
        if pid_file.exists():
            pid = int(pid_file.read_text().strip())
            report["api_pid"] = pid
            try:
                os.kill(pid, 0)
                report["process_alive"] = True
            except ProcessLookupError:
                report["process_alive"] = False
        bootstrap_log = args.profile_root / "bootstrap.log"
        if bootstrap_log.exists():
            bootstrap = bootstrap_log.read_text(errors="replace")
            if "Traceback (most recent call last)" in bootstrap:
                report["phase"] = "resident bootstrap failed; service is not qualified"
                report["bootstrap_log"] = str(bootstrap_log)
        if args.cache_root and pid_file.exists():
            report["weight_cache"] = cache_progress(args.cache_root, started_after=pid_file.stat().st_mtime)
            if report["phase"] == "loading checkpoint" and any(rank["new_banks"] for rank in report["weight_cache"]):
                report["phase"] = "expert conversion active on some ranks; other ranks may still be loading"
        receipt = args.profile_root / "bootstrap-ready.json"
        if receipt.exists():
            ready = json.loads(receipt.read_text())
            report["gates_passed_for_pid"] = ready["api_pid"]
            if ready["api_pid"] == report.get("api_pid") and report.get("process_alive"):
                report["phase"] = "ready: resident correctness gates passed"
        print(json.dumps(report, indent=2), flush=True)
        if not args.watch:
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
