# SPDX-License-Identifier: Apache-2.0
"""Generate a hashed append-only native manifest after standalone gates pass."""

import hashlib
import json
from pathlib import Path


def asset(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    root = Path(__file__).resolve().parent
    for device in range(4):
        assert json.loads((root / f"qualification-device-{device}.json").read_text())["passed"]
    validation = (root / "qualify.py").read_text()
    # The loader executes in a named module; __file__ identifies qualify.py for
    # import and binary paths. Qualification itself contains no serving changes.
    source = f"__file__ = {str(root / 'qualify.py')!r}\n" + validation
    manifest = {
        "name": "glm_eager_fusions_v1",
        "libraries": [asset(Path("/home/matteius/experiments/glm-eager-fusion-20261006/glm_eager_fusions_v1.so"))],
        "assets": [
            asset(root / name) for name in ("native.py", "qualify.py", "swiglu.bin", "combine.bin", "mhc_post.bin")
        ],
        "operators": ["glm_eager_fusions_v1::launch"],
        "validation_source": source,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
