import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

root = Path(__file__).parent
summary = []
for p in sorted(root.glob("rank*-spans.json")):
    s = json.loads(p.read_text())
    groups = defaultdict(list)
    for r in s["records"]:
        key = (r["label"], r.get("tokens"), r.get("mode"), r.get("elements"), r.get("dtype"), r.get("native"))
        groups[key].append(r["duration_ns"] / 1e6)
    rows = [
        dict(
            label=k[0],
            tokens=k[1],
            mode=k[2],
            elements=k[3],
            dtype=k[4],
            native=k[5],
            calls=len(v),
            total_ms=sum(v),
            median_ms=statistics.median(v),
            max_ms=max(v),
        )
        for k, v in groups.items()
    ]
    rows.sort(key=lambda r: -r["total_ms"])
    out = dict(
        rank=s["rank"], total=s["total"], dropped=s["dropped"], failed=sum(r["failed"] for r in s["records"]), rows=rows
    )
    summary.append(out)
    print("RANK", s["rank"], "dropped", s["dropped"], "failed", out["failed"])
    for r in rows[:14]:
        print(r)
    if s["rank"] == 0:
        print("schedule", Counter(tuple(r["scheduled"]) for r in s["records"] if r["label"] == "execute_model"))
(root / "summary.json").write_text(json.dumps(summary, indent=2))
