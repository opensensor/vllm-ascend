# SPDX-License-Identifier: Apache-2.0
"""Untimed actual-input MoE shadow for short and mixed c4 prefills."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

from tools.glm_perf.streaming_protocol import file_digest, verify_reference
from tools.glm_perf.streaming_resident import GuardedResidentController, StreamingTransport, load_candidate

SHADOW_LITERAL = "SHADOW_CHUNKS = (640, 1280)"
SHORT_SHADOW_LITERAL = "SHADOW_CHUNKS = tuple(range(17, 1281))"
MAX_OUTPUT_TOKENS = 16
BANK_COUNT = 43
WORKER_COUNT = 4


def extend_shadow_source(source):
    """Expand observation only; decode shapes remain outside the shadow range."""
    if source.count(SHADOW_LITERAL) != 1:
        raise ValueError("frozen shadow observation boundary changed")
    result = source.replace(SHADOW_LITERAL, SHORT_SHADOW_LITERAL, 1)
    ast.parse(result)
    return result


def check_comparisons(workers):
    if len(workers) != WORKER_COUNT or {row.get("rank") for row in workers} != set(range(WORKER_COUNT)):
        raise ValueError("missing or duplicate short-shadow rank")
    summary = []
    for row in workers:
        state = row.get("streaming", {})
        cases, whole = state.get("cases", []), state.get("whole_moe", [])
        if (
            state.get("shadow") is not True
            or state.get("mismatches") != 0
            or not cases
            or not whole
            or not any(type(case.get("tokens")) is int and 17 <= case["tokens"] < 640 for case in cases)
            or {case.get("bank") for case in cases} != set(range(BANK_COUNT))
            or any(
                case.get("finite") is not True
                or case.get("bitwise_equal") is not True
                or type(case.get("tokens")) is not int
                or not 17 <= case["tokens"] <= 1280
                for case in cases + whole
            )
        ):
            raise ValueError("short-prefill actual-input comparison failed or is incomplete")
        summary.append(
            {
                "rank": row["rank"],
                "native_comparisons": len(cases),
                "whole_moe_comparisons": len(whole),
                "observed_rows": sorted({case["tokens"] for case in cases}),
                "all_bitwise": True,
            }
        )
    return summary


def workflow(controller, workload):
    report = {
        "kind": "short_prefill_shadow_diagnostic",
        "state": "pending",
        "identity": controller.candidate.identity,
        "scope": "untimed actual-input correctness; not throughput or complete promotion evidence",
        "errors": [],
    }
    try:
        with controller.lease():
            controller.load()
            source, _ = controller.source("candidate", shadow=True)
            observed_source = extend_shadow_source(source)
            path = controller.output_dir / "short-shadow-source.py.txt"
            path.write_text(observed_source)
            report["observer"] = {
                "original_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "source_sha256": file_digest(path),
                "path": path.name,
                "only_change": [SHADOW_LITERAL, SHORT_SHADOW_LITERAL],
            }
            controller.switch(observed_source, "short_prefill_shadow")
            control = copy.deepcopy(workload)
            control["max_tokens"] = MAX_OUTPUT_TOKENS
            samples, workers = controller.request_workload(control, variant="candidate", concurrency=WORKER_COUNT)
            report.update(samples=samples, comparison_workers=workers, comparisons=check_comparisons(workers))
            controller.restore()
            report.update(
                state="complete", final_generation=controller.expected_generation, final_workers=controller.last_workers
            )
    except BaseException as error:
        report.update(state="failed", errors=[repr(error)])
        raise
    finally:
        report["journal"] = controller.journal
        (controller.output_dir / "short-shadow.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "gates", "workloads", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-generation", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    reference = verify_reference(json.loads(args.reference.read_text()), args.reference.parent)
    candidate = load_candidate(reference, args.gates)
    workloads = json.loads(args.workloads.read_text())
    control = next(row for row in workloads["workloads"] if row["id"] == "decode_c4")
    if not args.execute:
        print(json.dumps({"dry_run": True, "identity": candidate.identity, "scope": "untimed short-prefill shadow"}))
        return
    controller = GuardedResidentController(
        StreamingTransport("http://127.0.0.1:8001"),
        reference,
        args.reference.parent,
        args.output_dir,
        candidate=candidate,
        expected_generation=args.expected_generation,
    )
    report = workflow(controller, control)
    print(json.dumps({"state": report["state"], "generation": report["final_generation"]}))


if __name__ == "__main__":
    main()
