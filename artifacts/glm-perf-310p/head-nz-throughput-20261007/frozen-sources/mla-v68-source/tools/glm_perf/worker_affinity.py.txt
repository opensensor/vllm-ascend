# SPDX-License-Identifier: Apache-2.0
"""Plan/apply explicit Linux affinity to a server's workers and existing threads.

Run after worker initialization. Default is a read-only plan; --apply writes
the recorded CPU masks. No NPU API is called. Keep the JSON record to restore
the original masks with --restore. Bindings are PID=CPU_LIST, for example
--bind 1234=4-11,68-75. Choose masks from the host's physical-core/NUMA topology.
"""

import argparse
import json
import os
from pathlib import Path

import psutil


def parse_cpu_list(value: str) -> set[int]:
    cpus = set()
    for item in value.split(","):
        parts = item.split("-")
        if not 1 <= len(parts) <= 2 or any(not part.isdecimal() for part in parts):
            raise ValueError(f"invalid CPU list: {value}")
        first, last = int(parts[0]), int(parts[-1])
        if first > last or last > 65535:
            raise ValueError(f"invalid CPU range: {item}")
        cpus.update(range(first, last + 1))
    return cpus


def plan_bindings(server_pid: int, bindings: dict[int, set[int]]) -> dict:
    server = psutil.Process(server_pid)
    descendants = {server_pid, *(child.pid for child in server.children(recursive=True))}
    if not bindings or not bindings.keys() <= descendants:
        raise ValueError("all binding PIDs must belong to the selected server process tree")
    allowed = os.sched_getaffinity(0)
    processes = []
    for pid, cpus in bindings.items():
        if not cpus or not cpus <= allowed:
            raise ValueError(f"PID {pid}: requested CPUs must be within the caller's allowed CPU set")
        process = psutil.Process(pid)
        threads = [
            {"tid": thread.id, "before": sorted(os.sched_getaffinity(thread.id))} for thread in process.threads()
        ]
        processes.append(
            {
                "pid": pid,
                "created": process.create_time(),
                "name": process.name(),
                "cpus": sorted(cpus),
                "threads": threads,
            }
        )
    return {"server_pid": server_pid, "applied": False, "processes": processes}


def apply_bindings(plan: dict, *, restore: bool = False) -> None:
    # Check every process identity before the first mutation; never bind a
    # recycled PID from a stale experiment record.
    for item in plan["processes"]:
        if psutil.Process(item["pid"]).create_time() != item["created"]:
            raise RuntimeError(f"PID {item['pid']} has been recycled")
    for item in plan["processes"]:
        live_threads = {thread.id for thread in psutil.Process(item["pid"]).threads()}
        for thread in item["threads"]:
            if thread["tid"] not in live_threads:
                continue
            cpus = thread["before"] if restore else item["cpus"]
            try:
                os.sched_setaffinity(thread["tid"], cpus)
                if os.sched_getaffinity(thread["tid"]) != set(cpus):
                    raise RuntimeError(f"affinity verification failed for thread {thread['tid']}")
            except ProcessLookupError:
                # A worker may retire a helper thread between enumeration and
                # the syscall. New threads inherit their creator's mask.
                continue
    plan["applied"] = not restore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--bind", action="append", default=[], metavar="PID=CPU_LIST")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--restore", type=Path)
    args = parser.parse_args()
    if args.restore:
        if args.apply or args.bind or args.server_pid or args.output:
            parser.error("--restore takes only the saved record")
        plan = json.loads(args.restore.read_text())
        apply_bindings(plan, restore=True)
        print(json.dumps(plan, indent=2))
        return
    if not args.server_pid or not args.output or not args.bind:
        parser.error("--server-pid, --bind, and --output are required")
    if args.output.exists():
        parser.error("output already exists")
    try:
        bindings = {}
        for binding in args.bind:
            pid, cpus = binding.split("=", 1)
            pid = int(pid)
            if pid in bindings:
                raise ValueError(f"duplicate PID {pid}")
            bindings[pid] = parse_cpu_list(cpus)
        plan = plan_bindings(args.server_pid, bindings)
    except (ValueError, psutil.Error) as exc:
        parser.error(str(exc))
    # Save original masks before applying; even a partial failure is reversible.
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    if args.apply:
        apply_bindings(plan)
        args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
