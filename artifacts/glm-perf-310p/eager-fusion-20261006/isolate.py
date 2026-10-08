# SPDX-License-Identifier: Apache-2.0
"""Isolate which eager fusion component regresses concurrent decode.

Builds the fusion candidate with exactly one of SwiGLU / route-combine /
mHC-post enabled, then measures the standard short workload (one and four
streams) against the qualified `completed_pools_sinkhorn` baseline. The
`glm_eager_fusions_v1` native library must already be loaded and validated;
this script only switches Python dispatch and never reloads weights.
"""

import argparse
import dataclasses
import hashlib
import json
import re
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.suite import make_groups, run_groups, summarize

# (fuse_swiglu, fuse_combine, fuse_post) matching candidate.FUSION_SELECTION.
SELECTIONS = {
    "swiglu": (True, False, False),
    "combine": (False, True, False),
    "post": (False, False, True),
    "all": (True, True, True),
}

BASELINE_PATH = "/home/matteius/experiments/glm-prompt-profile-20261005/fused-baseline-candidate.py"
BASELINE_SHA = "348947ff4f965d2b6af532269f4b0e90eee1ba201d3e16dd60a063b8de423282"
MODEL = "glm53-flash-selective-w3"


def load_baseline() -> str:
    baseline = Path(BASELINE_PATH).read_text()
    if hashlib.sha256(baseline.encode()).hexdigest() != BASELINE_SHA:
        raise RuntimeError("baseline candidate source changed; refusing to compare against a new baseline")
    return baseline


def build_source(root: Path, baseline: str, component: str) -> str:
    candidate = (root / "candidate.py").read_text()
    selection = SELECTIONS[component]
    candidate, count = re.subn(
        r"FUSION_SELECTION = \(.*?\)",
        f"FUSION_SELECTION = {selection!r}",
        candidate,
        count=1,
    )
    if count != 1:
        raise RuntimeError("could not rewrite FUSION_SELECTION in candidate.py")
    source = baseline.replace(
        "def replacements(native_resources):", "def qualified_replacements(native_resources):"
    )
    return source + "\n" + candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=sorted(SELECTIONS), required=True)
    parser.add_argument("--full", action="store_true", help="baseline/candidate twice for drift control")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    baseline = load_baseline()
    source = build_source(root, baseline, args.component)
    client = ResidentClient("http://127.0.0.1:8001", timeout=900)

    before = client._acknowledged_rpc("resident_status", matches=client._status_matches)
    if not all(
        w.get("candidate") == "completed_pools_sinkhorn"
        and w.get("graphs_dirty") is False
        and not w.get("native_failed")
        for w in before
    ):
        raise RuntimeError("server is not in the qualified completed_pools_sinkhorn baseline state")
    if not all(
        w.get("native_loaded", {}).get("glm_eager_fusions_v1", {}).get("native_digest")
        == "58531a0b2177abb4a6d33d55a32ce59efea4f0a3d4e519db9aa9b9b8d7e7af49"
        for w in before
    ):
        raise RuntimeError("glm_eager_fusions_v1 native library is not loaded on every rank")

    short = [
        (name, [dataclasses.replace(case, max_tokens=args.max_tokens) for case in cases])
        for name, cases in make_groups(["short"], [])
    ]

    def switch(label: str) -> None:
        if label == "baseline":
            control = Control(uuid.uuid4().hex, candidate="completed_pools_sinkhorn", source=baseline)
        else:
            control = Control(uuid.uuid4().hex, candidate=f"fusion_{args.component}", source=source)
        client.switch(control)
        client.request("/resume")

    labels = ["baseline", "candidate", "baseline", "candidate"] if args.full else ["baseline", "candidate"]
    summaries = {}
    results_path = root / f"isolation-{args.component}-results.jsonl"
    try:
        with results_path.open("w") as output:
            for label in labels:
                switch(label)

                def record(row: dict) -> None:
                    row["isolation_label"] = label
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                    print(
                        json.dumps(
                            {
                                "component": args.component,
                                "label": label,
                                "case": row["case_id"],
                                "valid": row["valid"],
                                "decode_tokens_per_s": row.get("decode_tokens_per_s"),
                            }
                        ),
                        flush=True,
                    )

                rows = run_groups(short, client.base_url, MODEL, 42, on_result=record, timeout_s=900)
                summary = summarize(rows)
                summaries[label] = summary
                if not summary["valid"]:
                    raise RuntimeError(f"{label} run produced invalid requests: {summary}")
    finally:
        try:
            switch("baseline")
        except Exception:
            print(json.dumps(client.rpc("resident_status"), indent=2), flush=True)
            raise
    summary_path = root / f"isolation-{args.component}-summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
