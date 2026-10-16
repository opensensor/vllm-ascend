"""Bind the shared table to the TP6 overlay only after proving identical PLE files."""

import hashlib
import json
from pathlib import Path

ROOT = Path("/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016")
source = Path("/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i").resolve()
target = Path("/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-TP6-MTP-20261010").resolve()
index = "model.safetensors.index.json"
assert (source / index).resolve() == (target / index).resolve()
weights = json.loads((source / index).read_text())["weight_map"]
files = {name for key, name in weights.items() if "ple_embedding." in key}
assert files
assert all((source / name).resolve() == (target / name).resolve() for name in files)
metadata = json.loads((ROOT / "ple-table-source.json").read_text())
assert Path(metadata["model"]) == source
metadata["source_model"] = str(source)
metadata["model"] = str(target)
metadata["overlay_ple_files_verified"] = sorted(files)
(ROOT / "ple-table.json").write_text(json.dumps(metadata, indent=2) + "\n")
build = ROOT / "ple-selector-six-v2"
manifest = json.loads((build / "provenance.json").read_text())
# This build preceded adding the capacity field to the builder. Admit it using
# its exact source identity and the completed six-request hardware gate.
compiled_source = build / "native_ple_request.cpp"
assert hashlib.sha256(compiled_source.read_bytes()).hexdigest() == manifest["sources"][compiled_source.name]
assert compiled_source.read_bytes() == (ROOT / "runtime/tools/qwen4exp/native_ple_request.cpp").read_bytes()
assert "kMaxRequests = 6;" in compiled_source.read_text()
gate = json.loads((ROOT / "ple-gate-rank0.json").read_text())
assert gate["passed"] and any(case["lengths"] == [3] * 6 for case in gate["graph_shapes"])
manifest.update(max_requests=6, hardware_validated=True)
(build / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("TP6 overlay/table identity and six-request selector capacity verified")
