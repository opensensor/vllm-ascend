# SPDX-License-Identifier: Apache-2.0
"""Preparation-only pool diagnostic; parent must already pause(wait) and drain."""

import argparse
import dataclasses
import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.sidecard.qfn_resident_server import QfnResidentClient

SOURCE = """
import json
from vllm.distributed import get_tp_group
from tools.sidecard.qfn_allocator_diagnostic import inspect_allocator

def replacements():
    result = inspect_allocator(get_tp_group().device_communicator.custom_sum)
    raise RuntimeError("READ_ONLY_ALLOCATOR_STATE=" + json.dumps(result))
"""
MARKER = "READ_ONLY_ALLOCATOR_STATE="


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = QfnResidentClient(args.base_url, args.token_file)
    if client.request("/is_paused", method="GET").get("is_paused") is not True:
        raise RuntimeError("parent must establish pause(wait) and drain before this read-only diagnostic")
    control = Control(generation=uuid.uuid4().hex, candidate="allocator_read_only", source=SOURCE, recapture=False)
    result = client.request(
        "/collective_rpc", dict(method="resident_prepare", args=[json.dumps(dataclasses.asdict(control))], timeout=60)
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    records = []
    for row in result.get("results", []):
        message = row.get("error", "")
        if MARKER not in message:
            raise RuntimeError("diagnostic failed; inspect saved all-rank receipt; no automatic resume")
        record = json.loads(message.split(MARKER, 1)[1])
        if record["rank"] != row["rank"]:
            raise RuntimeError("diagnostic rank differs from worker acknowledgment")
        records.append(record)
    if sorted(record["rank"] for record in records) != [0, 1, 2, 3]:
        raise RuntimeError("expected exactly four diagnostic rank receipts")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
