"""Pause, change, and compare a running GLM server without loading weights."""

from __future__ import annotations

import argparse
import dataclasses
import json
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from tools.glm_perf.resident_control import MODES, Control


class ResidentClient:
    def __init__(self, base_url: str, expected_workers: int = 4, timeout: float = 900):
        self.base_url = base_url.rstrip("/")
        self.expected_workers = expected_workers
        self.timeout = timeout

    def request(self, path: str, payload: Any = None, method: str = "POST") -> Any:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = response.read()
        return json.loads(data) if data else None

    def rpc(self, method: str, *args: str) -> list[dict[str, Any]]:
        response = self.request("/collective_rpc", {"method": method, "args": list(args), "timeout": self.timeout})
        results = response.get("results") if isinstance(response, dict) else None
        if not isinstance(results, list) or len(results) != self.expected_workers:
            raise RuntimeError(f"{method}: expected {self.expected_workers} worker acknowledgments")
        if not all(isinstance(result, dict) for result in results):
            raise RuntimeError(f"{method}: malformed worker acknowledgment")
        errors = [(result.get("rank"), result["error"]) for result in results if "error" in result]
        if errors:
            raise RuntimeError(f"{method}: worker errors after collecting all acknowledgments: {errors}")
        return results

    def switch(self, control: Control) -> list[dict[str, Any]]:
        before = self.rpc("resident_status")
        was_paused = self.request("/is_paused", method="GET").get("is_paused")
        if not isinstance(was_paused, bool):
            raise RuntimeError("server did not report its pause state")
        # An existing pause may have frozen requests with mode=keep or retained
        # prefixes. Drain and invalidate scheduler caches in either case.
        if self.request("/pause?mode=wait&clear_cache=true").get("status") != "paused":
            raise RuntimeError("server did not finish draining requests")
        mutating = False
        try:
            prepared = self.rpc("resident_prepare", json.dumps(dataclasses.asdict(control)))
            if any(
                item.get("generation") != control.generation or item.get("digest") != control.digest
                for item in prepared
            ):
                raise RuntimeError("workers prepared different candidate source")
            if any(item.get("targets") != prepared[0].get("targets") for item in prepared):
                raise RuntimeError("workers prepared different replacement targets")
            mutating = True
            self.rpc("resident_reset")
            self.rpc("resident_apply", control.generation)
            self.rpc("resident_capture")
            results = self.rpc("resident_reset")
            identity_fields = ("rank", "pid", "weight_storage_digest")
            if {tuple(r.get(field) for field in identity_fields) for r in before} != {
                tuple(r.get(field) for field in identity_fields) for r in results
            }:
                raise RuntimeError("workers or resident weight storage changed during the switch")
            if len({r.get("rank") for r in results}) != self.expected_workers:
                raise RuntimeError("worker ranks are not unique")
            expected = {
                "generation": control.generation,
                "mode": control.mode,
                "candidate": control.candidate,
                "digest": control.digest,
                "graphs_dirty": False,
            }
            if any(any(result.get(key) != value for key, value in expected.items()) for result in results):
                raise RuntimeError("workers did not finish the same generation and graph capture")
        except Exception as error:
            if mutating:
                raise RuntimeError(
                    "resident switch failed; server remains paused. Apply baseline to recover, then resume"
                ) from error
            if not was_paused:
                self.request("/resume")
            raise
        if not was_paused:
            self.request("/resume")
        return results

    def resume(self) -> Any:
        workers = self.rpc("resident_status")
        if any(worker.get("graphs_dirty") is not False for worker in workers):
            raise RuntimeError("graphs are incomplete; apply baseline before resuming")
        fields = ("generation", "mode", "candidate", "digest")
        if len({tuple(worker.get(field) for field in fields) for worker in workers}) != 1:
            raise RuntimeError("workers disagree; apply baseline before resuming")
        return self.request("/resume")


def make_control(args: argparse.Namespace, mode: str) -> Control:
    source = args.candidate.read_text() if args.candidate else ""
    name = args.candidate_name or (args.candidate.stem if args.candidate else "baseline")
    return Control.from_dict(
        {"generation": uuid.uuid4().hex, "mode": mode, "candidate": name, "source": source, "recapture": args.recapture}
    )


def compare(args: argparse.Namespace, client: ResidentClient) -> bool:
    from tools.glm_perf.suite import make_groups, run_groups, summarize  # CLI-only workloads.

    if client.request("/is_paused", method="GET").get("is_paused") is not False:
        raise RuntimeError("resume the server before running comparisons")
    groups = make_groups(args.workloads, [])
    if len(set(args.modes)) != len(args.modes):
        raise ValueError("comparison modes must be unique")
    selected = make_control(args, args.modes[0])
    summaries = {}
    identities = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for mode in args.modes:
            control = dataclasses.replace(selected, generation=uuid.uuid4().hex, mode=mode)
            workers = client.switch(control)
            current_identities = {(r["rank"], r["pid"], r["weight_storage_digest"]) for r in workers}
            if identities is not None and identities != current_identities:
                raise RuntimeError("workers restarted between comparisons")
            identities = current_identities

            metadata = {
                "mode": mode,
                "candidate": control.candidate,
                "generation": control.generation,
                "digest": control.digest,
                "workers": workers,
            }

            def record(row: dict[str, Any], resident: dict[str, Any] = metadata) -> None:
                row["resident"] = resident
                output.write(json.dumps(row) + "\n")
                output.flush()

            rows = run_groups(groups, args.base_url, args.model, args.seed, on_result=record, timeout_s=args.timeout)
            summaries[mode] = summarize(rows)
            # A failed HTTP request can still be executing in the engine.
            # Stop comparisons; the next switch will explicitly drain it.
            if not summaries[mode]["valid"]:
                break
    args.output.with_suffix(".summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    return len(summaries) == len(args.modes) and all(summary["passed"] for summary in summaries.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--expected-workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900)
    commands = parser.add_subparsers(dest="command", required=True)
    switch = commands.add_parser("switch", help="drain requests and apply one generation")
    switch.add_argument("--mode", choices=MODES, required=True)
    comparison = commands.add_parser("compare", help="run the same workloads in each execution mode")
    comparison.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    comparison.add_argument(
        "--workloads", choices=("quality", "fault", "fault4", "short", "tool"), nargs="+", default=["fault", "fault4"]
    )
    comparison.add_argument("--model", default="glm53-flash-selective-w3")
    comparison.add_argument("--seed", type=int, default=42)
    comparison.add_argument("--output", type=Path, required=True)
    for command in (switch, comparison):
        command.add_argument("--candidate", type=Path, help="Python file defining replacements()")
        command.add_argument("--candidate-name", help="label for the candidate source")
        command.add_argument("--recapture", action="store_true", help="recapture even if source is unchanged")
    commands.add_parser("status")
    commands.add_parser("resume")
    args = parser.parse_args()
    if args.expected_workers < 1 or args.timeout <= 0:
        parser.error("expected workers and timeout must be positive")
    client = ResidentClient(args.base_url, args.expected_workers, args.timeout)
    if args.command == "switch":
        print(json.dumps(client.switch(make_control(args, args.mode)), indent=2))
    elif args.command == "compare":
        return 0 if compare(args, client) else 1
    elif args.command == "status":
        print(json.dumps(client.rpc("resident_status"), indent=2))
    else:
        print(json.dumps(client.resume()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
