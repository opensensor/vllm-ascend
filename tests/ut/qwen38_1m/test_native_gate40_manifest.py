# SPDX-License-Identifier: Apache-2.0

import pytest

from tools.glm_perf.resident_native import NativeManifest
from tools.qwen4exp.prepare_native_gate40 import RESOURCE_SOURCES, make_manifest


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_manifest_covers_native_binaries_and_imported_validation_code(tmp_path, version):
    library = tmp_path / "native.so"
    library.write_bytes(b"fake library; no registration")
    binaries = {}
    for profile in ("baseline", "tile40", "stream40"):
        path = tmp_path / f"{profile}.bin"
        path.write_bytes(profile.encode())
        binaries[profile] = str(path)
    for source in RESOURCE_SOURCES:
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test source\n")
    value = make_manifest(library, binaries, tmp_path, f"qwen_gate40_{version}", f"qwen_native_gate40_{version}")
    manifest = NativeManifest(value)
    manifest.verify_files()
    assert len(value["assets"]) == len(RESOURCE_SOURCES) + 3
    assert value["operators"] == [f"qwen_native_gate40_{version}::launch"]
    # Building/compiling a manifest must never execute its validation source.
    compile(value["validation_source"], "<native validation>", "exec")
    library.write_bytes(b"changed after preparation")
    with pytest.raises(ValueError, match="digest"):
        manifest.verify_files()


def test_missing_profile_is_rejected_before_hashing(tmp_path):
    with pytest.raises(ValueError, match="binaries"):
        make_manifest(tmp_path / "missing.so", {"tile40": "missing.bin"}, tmp_path)
