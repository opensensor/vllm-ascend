# SPDX-License-Identifier: Apache-2.0
"""Select the existing decode dispatch gates and recapture without reloading."""

import argparse
import dataclasses
import hashlib
import json
import uuid
from pathlib import Path

from qualified_harness import ResidentClient

from tools.glm_perf.resident_control import Control
from tools.glm_perf.suite import make_groups, run_groups, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", choices=("swiglu", "combine", "both"))
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    qualified = Path("/home/matteius/experiments/glm-prompt-profile-20261005/fused-baseline-candidate.py").read_text()
    assert (
        hashlib.sha256(qualified.encode()).hexdigest()
        == "348947ff4f965d2b6af532269f4b0e90eee1ba201d3e16dd60a063b8de423282"
    )
    selection = {"swiglu": (True, False), "combine": (False, True), "both": (True, True)}[args.selection]
    selector = (
        (root / "resident_selector.py")
        .read_text()
        .replace("DECODE_SELECTION = (True, True)", f"DECODE_SELECTION = {selection!r}")
    )
    source = qualified.replace("def replacements(native_resources):", "def qualified_replacements(native_resources):")
    source += "\n" + selector
    (root / f"resident-{args.selection}.py").write_text(source)
    client = ResidentClient("http://127.0.0.1:8001", timeout=900)
    before = client._acknowledged_rpc("resident_status", matches=client._status_matches)
    after = client.switch(Control(uuid.uuid4().hex, candidate=f"decode_{args.selection}", source=source))
    identity = lambda rows: sorted((row["rank"], row["pid"], row["weight_storage_digest"]) for row in rows)
    assert identity(before) == identity(after), "resident weights or workers changed"
    assert all(row.get("decode_selection") == {"swiglu": selection[0], "combine": selection[1]} for row in after)
    client.resume()
    receipt = {"before": before, "after": after, "unchanged_workers_and_weights": True}
    if args.probe:
        groups = [
            (name, [dataclasses.replace(case, max_tokens=16) for case in cases])
            for name, cases in make_groups(["short"], [])
        ]
        rows = run_groups(groups, client.base_url, "glm53-flash-selective-w3", 42, timeout_s=900)
        assert all(row["valid"] and row["passed"] for row in rows)
        receipt["probe"] = summarize(rows)
    (root / f"hot-swap-{args.selection}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "selected": args.selection,
                "worker_pids": [row["pid"] for row in after],
                "unchanged_workers_and_weights": True,
                "probe": args.probe,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
