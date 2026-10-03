"""Count neighboring tasks for AI-CPU Cast events in a CANN kernel CSV.

This is a stream-local launch-order heuristic, not a dependency graph.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel_csv", type=Path)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    streams = defaultdict(list)
    with args.kernel_csv.open(newline="") as source:
        for row in csv.DictReader(source):
            streams[row["Stream ID"]].append(row)

    patterns = Counter()
    for events in streams.values():
        events.sort(key=lambda row: float(row["Start Time(us)"]))
        for index, event in enumerate(events):
            if event["Name"] != "Cast" or event["Accelerator Core"] != "AI_CPU":
                continue
            previous = events[index - 1]["Name"] if index else "<start>"
            following = events[index + 1]["Name"] if index + 1 < len(events) else "<end>"
            patterns[(previous, following, event["Input Shapes"])] += 1

    print(f"AI-CPU Cast count: {sum(patterns.values())}")
    for pattern, count in patterns.most_common(args.limit):
        print(f"{count:6d}  {pattern}")


if __name__ == "__main__":
    main()
