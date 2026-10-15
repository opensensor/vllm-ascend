"""Keep the decode tail measurements and longest bounded host spans."""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/srv/ai/storage/artifacts/qfn-high-end-20261015/decode-tail-v2")
BENCH = ROOT.parent.parent / "npu-collective-qfn-20261014-024609/qsa-resident-v1"


def main():
    receipt = json.loads((BENCH / "decode-benchmark-high-end-tail-v2.json").read_text())
    result = dict(
        source=str(ROOT),
        trials=[
            {key: row[key] for key in ("trial", "seconds", "output_tokens_per_second")} for row in receipt["trials"]
        ],
        spans={},
    )
    for path in sorted(ROOT.glob("*-spans.json")):
        snapshot = json.loads(path.read_text())
        labels = defaultdict(list)
        for row in snapshot["records"]:
            labels[row["label"]].append(row)
        result["spans"][path.name] = dict(
            total=snapshot["total"],
            dropped=snapshot["dropped"],
            labels={
                label: dict(
                    count=len(rows),
                    max_ms=max(r["duration_ns"] for r in rows) / 1e6,
                    slowest=sorted(rows, key=lambda r: -r["duration_ns"])[:3],
                )
                for label, rows in labels.items()
            },
        )
    stop = json.loads((ROOT / "stop/receipt.json").read_text())
    result["resume"] = stop["resume"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
