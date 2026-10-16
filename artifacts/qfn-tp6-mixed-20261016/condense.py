# SPDX-License-Identifier: Apache-2.0
"""Keep runtime identity/graph evidence without historical per-shape counters."""

import argparse
import hashlib
import json
from pathlib import Path


def condense(value):
    if isinstance(value, list):
        return [condense(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: condense(item) for key, item in value.items() if key != "discard"}
    if result.get("rank", 0) != 0 and isinstance(result.get("results"), list):
        for row in result["results"]:
            if isinstance(row, dict) and "geometry" in row and "timings" in row:
                row.pop("timings")
                row["cross_rank_timings_recorded_on_rank"] = 0
    if "custom_collective" in result:
        collective = result["custom_collective"]
        fallbacks = collective.pop("fallback_schemas", [])
        collective["fallback_shape_count"] = len(fallbacks)
        collective["fallback_calls"] = sum(row["calls"] for row in fallbacks)
        collective["observed_mixed_fallbacks"] = [row for row in fallbacks if row["elements"] in (335360, 343040)]
    if "native_ple" in result:
        fallbacks = result["native_ple"].pop("fallbacks", {})
        result["native_ple"]["fallback_shape_count"] = len(fallbacks)
        result["native_ple"]["fallback_calls"] = sum(fallbacks.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    result = dict(
        raw_filename=args.source.name, raw_sha256=hashlib.sha256(raw).hexdigest(), receipt=condense(json.loads(raw))
    )
    args.destination.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
