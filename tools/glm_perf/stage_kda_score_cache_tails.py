# SPDX-License-Identifier: Apache-2.0
"""Stage a diagnostic partial cache; this variant failed complete KDA gates.

This tool only writes an isolated source file. It does not admit a serving
package. The archived failure affects recurrent state at 63 tokens; preserve
this disabled candidate for diagnosis, and never treat it as qualified.
"""

import argparse
import hashlib
import json
from pathlib import Path

FLAG = "GLM_KDA_SCORE_CACHE_TAILS"
PARENT = """            curT == scoreCacheMaxRows &&
            K_ >= 16 && K_ <= scoreCacheMaxChannels && K_ % 16 == 0 &&"""
CANDIDATE = """#ifdef GLM_KDA_SCORE_CACHE_TAILS
            curT > 0 && curT <= scoreCacheMaxRows &&
#else
            curT == scoreCacheMaxRows &&
#endif
            K_ >= 16 && K_ <= scoreCacheMaxChannels && K_ % 16 == 0 &&"""


def transform(source):
    if FLAG in source:
        raise ValueError("KDA partial cache already staged")
    if source.count(PARENT) != 1 or "#ifdef GLM_KDA_SCORE_CACHE_COLUMNS" not in source:
        raise ValueError("qualified KDA column cache source changed")
    return source.replace(PARENT, CANDIDATE)


def stage(source, destination, expected_sha256):
    original = source.read_bytes()
    if hashlib.sha256(original).hexdigest() != expected_sha256:
        raise ValueError("KDA partial cache source differs from its qualified parent")
    candidate = transform(original.decode()).encode()
    with destination.open("xb") as output:
        output.write(candidate)
    return dict(
        source=str(source),
        source_sha256=expected_sha256,
        destination=str(destination),
        candidate_sha256=hashlib.sha256(candidate).hexdigest(),
        compiler_define=FLAG,
        extra_ub_bytes=0,
        full_kda_evaluated=False,
        serving_evaluated=False,
        serving_eligible=False,
        known_gate_failure="partial-cache-v1 recurrent-state mismatch at 63 tokens",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = stage(args.source, args.destination, args.expected_sha256)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
