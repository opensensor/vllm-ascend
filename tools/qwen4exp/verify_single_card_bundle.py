# SPDX-License-Identifier: Apache-2.0
"""Verify the frozen single-card snapshot before launching or native loading."""

import argparse
import hashlib
import json
from pathlib import Path


def verify_bundle(root: Path):
    root = root.resolve()
    records = json.loads((root / "files.sha256.json").read_text())
    if not isinstance(records, dict) or not records:
        raise ValueError("bundle file manifest must be a nonempty mapping")
    failures = []
    for name, expected in records.items():
        if not isinstance(name, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("invalid file manifest entry")
        relative = Path(name)
        path = root / relative
        if relative.is_absolute() or ".." in relative.parts or not path.resolve().is_relative_to(root):
            raise ValueError(f"manifest path escapes bundle: {name}")
        if path.is_symlink() or not path.is_file():
            failures.append(f"missing or linked file: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"SHA-256 mismatch: {name}")
    if failures:
        raise ValueError("\n".join(failures))
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    try:
        count = verify_bundle(args.bundle)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Verified {count} frozen files. Run this before native.sh modifies its receipt.")


if __name__ == "__main__":
    main()
