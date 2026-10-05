# SPDX-License-Identifier: Apache-2.0
"""Remove stale baseline metadata from this isolated, renamed-operator package."""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_operator import rename

BASELINE_OP = "QsaGatherValueNzV310"
CANDIDATE_OP = "QsaGatherValueNzZeroV310"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vendor", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    root = args.vendor / "op_impl/ai_core/tbe"
    receipt = {}
    for relative in (
        "config/ascend310p/aic-ascend310p-ops-info.json",
        "kernel/config/ascend310p/binary_info_config.json",
    ):
        path = root / relative
        original = path.read_bytes()
        data = json.loads(original)
        entry = data.get(CANDIDATE_OP)
        if entry is None:
            if "binary_info" in relative or set(data) != {BASELINE_OP}:
                raise ValueError(f"unexpected operator metadata: {relative}: {list(data)}")
            # Both operators have the same inputs, outputs, attributes and
            # tiling schema. Only their registered names and kernel differ.
            entry = json.loads(rename(json.dumps(data[BASELINE_OP])))
        normalized = (json.dumps({CANDIDATE_OP: entry}, indent=2) + "\n").encode()
        path.write_bytes(normalized)
        receipt[relative] = {
            "previous_ops": list(data),
            "installed_ops": [CANDIDATE_OP],
            "previous_sha256": hashlib.sha256(original).hexdigest(),
            "installed_sha256": hashlib.sha256(normalized).hexdigest(),
        }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
