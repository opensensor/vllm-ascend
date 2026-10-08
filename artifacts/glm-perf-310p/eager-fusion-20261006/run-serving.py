# SPDX-License-Identifier: Apache-2.0
"""Resident fusion loading, numerical quality and paired serving checks."""

import dataclasses
import hashlib
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.resident_native import NativeManifest
from tools.glm_perf.suite import make_groups, run_groups, summarize


def main():
    root = Path(__file__).resolve().parent
    baseline = Path("/home/matteius/experiments/glm-prompt-profile-20261005/fused-baseline-candidate.py").read_text()
    assert (
        hashlib.sha256(baseline.encode()).hexdigest()
        == "348947ff4f965d2b6af532269f4b0e90eee1ba201d3e16dd60a063b8de423282"
    )
    source = baseline.replace("def replacements(native_resources):", "def qualified_replacements(native_resources):")
    source += "\n" + (root / "candidate.py").read_text()
    (root / "applied-candidate.py").write_text(source)
    manifest = NativeManifest(json.loads((root / "manifest.json").read_text()))
    client = ResidentClient("http://127.0.0.1:8001", timeout=900)
    before = client._acknowledged_rpc("resident_status", matches=client._status_matches)
    assert all(
        w["candidate"] == "completed_pools_sinkhorn" and not w["graphs_dirty"] and not w.get("native_failed")
        for w in before
    )
    (root / "before.json").write_text(json.dumps(before, indent=2) + "\n")
    client.request("/pause?mode=wait&clear_cache=true")
    for attempt in range(16):
        try:
            prepared = client.rpc("resident_native_prepare", manifest.payload)
        except RuntimeError as error:
            if "malformed worker acknowledgment" not in str(error):
                raise
        else:
            if all(w.get("native_digest") == manifest.digest for w in prepared):
                break
    else:
        raise RuntimeError("native preparation acknowledgments did not converge")
    client._acknowledged_rpc(
        "resident_native_load",
        manifest.digest,
        matches=lambda rows: all(
            (w.get("native_digest") == manifest.digest and w.get("validation", {}).get("passed") is True)
            or (
                w.get("native_loaded", {}).get(manifest.name, {}).get("native_digest") == manifest.digest
                and not w.get("native_failed")
            )
            for w in rows
        ),
    )
    loaded = client._acknowledged_rpc("resident_status", matches=client._status_matches)
    (root / "loaded.json").write_text(json.dumps(loaded, indent=2) + "\n")
    print("all four resident native gates passed", flush=True)
    selected = False
    summaries = {}

    def switch(label):
        return client.switch(
            Control(
                uuid.uuid4().hex,
                candidate="all_batch_fusions" if label == "candidate" else "completed_pools_sinkhorn",
                source=source if label == "candidate" else baseline,
            )
        )

    def run(label, groups):
        path = root / f"{label}-results.jsonl"
        with path.open("w") as output:

            def record(row):
                output.write(json.dumps(row) + "\n")
                output.flush()
                print(
                    json.dumps(
                        {
                            "run": label,
                            "case": row["case_id"],
                            "valid": row["valid"],
                            "decode_tokens_per_s": row.get("decode_tokens_per_s"),
                            "ttft_s": row.get("ttft_s"),
                        }
                    ),
                    flush=True,
                )

            rows = run_groups(groups, client.base_url, "glm53-flash-selective-w3", 42, on_result=record, timeout_s=900)
        result = summarize(rows)
        (root / f"{label}-summary.json").write_text(json.dumps(result, indent=2) + "\n")
        assert all(row["valid"] for row in rows), result
        summaries[label] = result
        return rows

    try:
        switch("candidate")
        client.request("/resume")
        # Numerical changes need complete-answer checks. Batch four cases to
        # exercise concurrent state/routing without changing their token caps.
        quality = [case for _, cases in make_groups(["quality"], []) for case in cases]
        groups = [(f"quality_{offset}", quality[offset : offset + 4]) for offset in range(0, len(quality), 4)]
        run("candidate-quality", groups + make_groups(["tool"], []))
        short = [
            (name, [dataclasses.replace(case, max_tokens=128) for case in cases])
            for name, cases in make_groups(["short"], [])
        ]
        for label in ("baseline1", "candidate1", "baseline2", "candidate2"):
            switch("candidate" if label.startswith("candidate") else "baseline")
            client.request("/resume")
            run(label, short)
        selected = True
    finally:
        # Keep only a candidate that passes complete-answer and serving gates.
        final = client.rpc("resident_status") if selected else switch("baseline")
        client.request("/resume")
        (root / "final.json").write_text(
            json.dumps(
                {"selected": selected, "workers": final, "pause": client.request("/is_paused", method="GET")}, indent=2
            )
            + "\n"
        )
        assert sorted((w["pid"], w["weight_storage_digest"]) for w in final) == sorted(
            (w["pid"], w["weight_storage_digest"]) for w in before
        )
        print("serving restored with unchanged resident weights", flush=True)
    (root / "summaries.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
