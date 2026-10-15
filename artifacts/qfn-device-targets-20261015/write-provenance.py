"""Record the staged sources and binaries actually admitted by the live gates."""

import hashlib
import json
from pathlib import Path

ROOT = Path("/srv/ai/storage/artifacts/qfn-device-targets-20261015")
files = []
for directory in ("shared-mapping-v1", "qsa-table-v1"):
    files.extend(p for p in (ROOT / directory).iterdir() if p.is_file())
files.append(
    Path(
        "/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/checkpoint-native-build-v1/compile-checkpoint-preparation"
    )
)
result = dict(
    base_commit="fe5d4cd60eb9371123a213170075c0001d6988c2",
    note="Staged hardware-validated sources; repository formatting may differ without behavioral changes.",
    files=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)],
)
(ROOT / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
