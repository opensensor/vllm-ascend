"""Read all-rank device task envelopes from CANN task_time.csv exports.

This works when a malformed timeline JSON prevents kernel_details.csv output.
The envelope includes the prompt/prefill tasks; it is not a decode-only time.
"""

import argparse
import csv
import json
import re
from pathlib import Path

RANK_RE = re.compile(r"(?:^|_)rank(\d+)(?:_|$)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root", type=Path)
    args = parser.parse_args()

    ranks = {}
    for path in sorted(args.trace_root.rglob("task_time.csv")):
        match = next((RANK_RE.search(part) for part in path.parts if RANK_RE.search(part)), None)
        if match is None:
            raise ValueError(f"cannot identify rank from {path}")
        rank = int(match.group(1))
        if rank in ranks:
            raise ValueError(f"multiple task exports for rank {rank}")
        first = float("inf")
        last = float("-inf")
        count = 0
        with path.open(newline="") as source:
            for row in csv.DictReader(source):
                first = min(first, float(row["task_start(us)"]))
                last = max(last, float(row["task_stop(us)"]))
                count += 1
        if not count:
            raise ValueError(f"no tasks in {path}")
        ranks[rank] = {"task_count": count, "first_us": first, "last_us": last, "envelope_ms": (last - first) / 1000}

    if set(ranks) != {0, 1, 2, 3}:
        raise ValueError(f"expected ranks 0..3, found {sorted(ranks)}")
    first = min(item["first_us"] for item in ranks.values())
    last = max(item["last_us"] for item in ranks.values())
    print(json.dumps({"all_rank_envelope_ms": (last - first) / 1000, "ranks": ranks}, indent=2))


if __name__ == "__main__":
    main()
