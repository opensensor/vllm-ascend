import hashlib
import json

import pytest

from tools.rtsp_scene.prefill_probe import PREFILL_WIDTHS, checked_build


def manifest(directory):
    hashes = {}
    for inner in PREFILL_WIDTHS:
        path = directory / f"prefill-k{inner}.bin"
        path.write_bytes(str(inner).encode())
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"columns": 128, "row_tile": 64, "arguments": 6, "sha256": hashes}


def test_prefill_binary_fingerprint_and_tile_contract(tmp_path):
    expected = manifest(tmp_path)
    (tmp_path / "build.json").write_text(json.dumps(expected))
    assert checked_build(tmp_path) == expected
    (tmp_path / "prefill-k4096.bin").write_bytes(b"replacement")
    with pytest.raises(ValueError, match="fingerprint"):
        checked_build(tmp_path)


@pytest.mark.parametrize("field,value", [("columns", 256), ("row_tile", 16), ("arguments", 4)])
def test_prefill_rejects_a_different_native_contract(tmp_path, field, value):
    expected = manifest(tmp_path)
    expected[field] = value
    (tmp_path / "build.json").write_text(json.dumps(expected))
    with pytest.raises(ValueError, match="contract"):
        checked_build(tmp_path)
