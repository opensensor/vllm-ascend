# SPDX-License-Identifier: Apache-2.0
"""Stage shared K/V tile reuse in a private QSA source copy."""

import argparse
import hashlib
import json
from pathlib import Path

FLAG = "GLM_QSA_SHARED_CACHE"


def transform(source):
    if FLAG in source:
        raise ValueError("QSA shared-cache reuse already staged")
    anchors = {
        "        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));": 1,
        "                DataCopy(valueGather[localToken * NZ_INNER], valueCacheGm_[cacheOffset], groupCopy);": 1,
        "                DataCopy(valueGather[localToken * NZ_INNER], valueCacheGm_[cacheOffset], tokenCopy);": 1,
        "            DataCopy(valueGather[localToken * NZ_INNER], valueCacheGm_[cacheOffset], tokenCopy);": 1,
        "            DataCopy(valueGather[localToken * NZ_INNER], valueCacheGm_[fallback], tokenCopy);": 1,
        "            CopyGatheredKvToL1(kvTransposeBuf_.Get<half>());": 1,
        "    TPipe *pipe_ = nullptr;": 1,
    }
    # Match complete lines: the short-indented token copy is also a substring
    # of the deeper-indented branch, but represents a different DMA site.
    lines = source.splitlines()
    if any(lines.count(line) != count for line, count in anchors.items()):
        raise ValueError("qualified QSA gather/lifetime anchors changed")
    result = []
    for line in lines:
        if line == "    TPipe *pipe_ = nullptr;":
            result += ["#ifdef " + FLAG, "    bool sharedKeyValue_ = false;", "#endif"]
        if "DataCopy(valueGather[" in line or line == "            CopyGatheredKvToL1(kvTransposeBuf_.Get<half>());":
            indentation = line[: len(line) - len(line.lstrip())]
            result += ["#ifdef " + FLAG, indentation + "if (!sharedKeyValue_)", "#endif"]
        result.append(line)
        if line == "        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));":
            result += ["#ifdef " + FLAG, "        sharedKeyValue_ = keyCache == valueCache;", "#endif"]
    return "\n".join(result) + "\n"


def stage(source, destination, expected_sha256):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("QSA source differs from qualified parent")
    candidate = transform(raw.decode()).encode()
    with destination.open("xb") as out:
        out.write(candidate)
    return dict(
        source_sha256=expected_sha256,
        candidate_sha256=hashlib.sha256(candidate).hexdigest(),
        compiler_define=FLAG,
        extra_ub_bytes=0,
        full_operator_evaluated=False,
        serving_evaluated=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "destination", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    report = stage(args.source, args.destination, args.expected_sha256)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
