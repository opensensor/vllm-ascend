"""Summarize native kernel evidence without copying the large raw traces."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/srv/ai/storage/artifacts/qfn-mixed-prefill-20261015")
LABELS = ("fresh-retained-native-20261015", "fresh-retained-late-native-20261015", "gdn-merge-native-v1")


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
            output.append(
                dict(
                    label=label,
                    source=str(path),
                    kernels=[
                        dict(name=n, ms=v / 1000, count=counts[n]) for n, v in sorted(sums.items(), key=lambda x: -x[1])
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
