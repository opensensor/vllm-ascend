"""Summarize native kernel evidence without copying the large raw traces."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/srv/ai/storage/artifacts/qfn-pack-followup-20261016")
LABELS = ("retained-native-v16",)


def main():
    output = []
    for label in LABELS:
        for path in sorted((ROOT / label / "native").glob("rank[03]/**/kernel_details.csv")):
            sums, shapes, counts = defaultdict(float), defaultdict(float), Counter()
            for row in csv.DictReader(path.open()):
                name, duration = row["Type"], float(row["Duration(us)"])
                sums[name] += duration
                counts[name] += 1
                if name in ("IndexPut", "MatMulV2", "QsaSparseAttentionV310", "Cast", "GatherV2"):
                    shapes[(name, row.get("Input Shapes", ""), row.get("Input Data Types", ""))] += duration
            events = json.loads(path.with_name("trace_view.json").read_text())
            if isinstance(events, dict):
                events = events["traceEvents"]
            apis, api_counts = defaultdict(float), Counter()
            for event in events:
                name = event.get("name", "")
                if event.get("ph") == "X" and name.startswith("AscendCL@"):
                    apis[name] += float(event["dur"])
                    api_counts[name] += 1
            output.append(
                dict(
                    label=label,
                    source=str(path),
                    kernels=[
                        dict(name=n, ms=v / 1000, count=counts[n]) for n, v in sorted(sums.items(), key=lambda x: -x[1])
                    ],
                    host_apis=[
                        dict(name=n, ms=v / 1000, count=api_counts[n])
                        for n, v in sorted(apis.items(), key=lambda x: -x[1])
                    ],
                    shapes=[
                        dict(name=n, shape=s, dtypes=t, ms=v / 1000)
                        for (n, s, t), v in sorted(shapes.items(), key=lambda x: -x[1])[:30]
                    ],
                )
            )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
