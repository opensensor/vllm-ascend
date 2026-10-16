# SPDX-License-Identifier: Apache-2.0
"""Join whole-stage identities; preserve host overlap and unmapped device clocks."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .execution_trace import SCHEMA, device_interval_error, timing_error_counts, unavailable_device_interval


def summarize(paths):
    groups, metadata = defaultdict(list), []
    for path in paths:
        lines = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if not lines or lines[0].get("schema") != SCHEMA or lines[0].get("kind") != "metadata":
            raise ValueError("not a whole-stage execution trace")
        header = lines[0]
        for record in lines[1:]:
            if record.get("kind") != "span" or record["host_end_ns"] < record["host_start_ns"]:
                raise ValueError("invalid execution span")
            if record.get("device_ms") is not None:
                error = device_interval_error(record["device_ms"])
                if error is not None:
                    unavailable_device_interval(record, error)
            key = (
                header["run_id"],
                header["host_clock"],
                *[record[k] for k in ("request", "sequence", "position", "rows")],
            )
            groups[key].append(dict(record, owner=header["owner"], pid=header["pid"]))
        errors = timing_error_counts(lines[1:])
        metadata.append(
            dict(
                header,
                invalid_device_timings=sum(errors.values()),
                device_timing_errors=errors,
                device_timing_reliability="unreliable" if errors else "unverified",
            )
        )
    iterations = []
    for key, spans in sorted(groups.items()):
        outer = [s for s in spans if s["owner"] == "cuda" and s["name"] == "iteration"]
        if len(outer) > 1:
            raise ValueError("duplicate coordinator iteration identity")
        first = min(s["host_start_ns"] for s in spans)
        timeline = [
            dict(
                s,
                relative_host_start_ms=(s["host_start_ns"] - first) / 1e6,
                host_ms=(s["host_end_ns"] - s["host_start_ns"]) / 1e6,
            )
            for s in sorted(spans, key=lambda s: s["host_start_ns"])
        ]
        chain = [s for s in timeline if s["owner"] == "cuda" and s["name"] in ("prefix", "middle", "suffix")]
        iterations.append(
            dict(
                run_id=key[0],
                host_clock=key[1],
                request=key[2],
                sequence=key[3],
                position=key[4],
                rows=key[5],
                observed_iteration_ms=(outer[0]["host_end_ns"] - outer[0]["host_start_ns"]) / 1e6 if outer else None,
                observed_host_dependency_chain=chain,
                complete_boundary_chain={s["name"] for s in chain} == {"prefix", "middle", "suffix"},
                timeline=timeline,
                participating_owners=sorted({s["owner"] for s in spans}),
                attribution=(
                    "coordinator host critical path observed; device intervals unmapped; graph collectives unattributed"
                    if outer
                    else "partial/pipelined host timeline; no serial iteration envelope; device intervals unmapped"
                ),
            )
        )
    return dict(
        schema=SCHEMA + "-summary",
        metadata=metadata,
        invalid_device_timings=sum(header["invalid_device_timings"] for header in metadata),
        device_timing_reliability=(
            "unreliable" if any(header["invalid_device_timings"] for header in metadata) else "unverified"
        ),
        iterations=iterations,
        limitations=[
            "Host enqueue spans overlap device execution and must not be summed with device durations.",
            "Device event durations are stream intervals including queued work; no kernel-only attribution.",
            "No graph-internal collective arrival/transit decomposition or expert-only rank-counter joins.",
            "Validate traced/untraced vocabulary and wall-time overhead before using these diagnostics.",
            "Negative/nonfinite or failed event timings are unavailable; remaining device samples remain unverified.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.traces), indent=2) + "\n")


if __name__ == "__main__":
    main()
