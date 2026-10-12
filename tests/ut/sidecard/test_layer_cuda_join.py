# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest

from tools.sidecard.layer_cuda_join import NATIVE_GATE_FILES, join


@pytest.mark.parametrize("corrupt", (None, "cuda", "npu"))
def test_join_authenticates_both_architectures_before_creating_output(tmp_path, corrupt):
    native = tmp_path / "original"
    native.mkdir()
    hashes = {}
    for name in NATIVE_GATE_FILES:
        (native / name).write_bytes(name.encode())
        hashes[name] = hashlib.sha256(name.encode()).hexdigest()
    clipped = native / "build.json"
    clipped.write_text(json.dumps({"swiglu_limit": 10.0, "native_gate_files": hashes}))
    library = tmp_path / "cuda.so"
    library.write_bytes(b"qualified CUDA binary")
    rebuilt = tmp_path / "cuda-build.json"
    rebuilt.write_text(
        json.dumps(
            {
                "swiglu_limit": 10.0,
                "library": str(library),
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            }
        )
    )
    output = tmp_path / "joined"
    if corrupt == "cuda":
        library.write_bytes(b"wrong CUDA binary")
    if corrupt == "npu":
        (native / sorted(NATIVE_GATE_FILES)[0]).write_bytes(b"wrong Ascend binary")
    if corrupt:
        with pytest.raises(ValueError, match="authentication"):
            join(output, clipped, rebuilt)
        assert not output.exists()
    else:
        receipt = join(output, clipped, rebuilt)
        assert receipt["hardware_validated"] is False
        assert (output / "cuda.so").read_bytes() == library.read_bytes()
        for name in NATIVE_GATE_FILES:
            assert (output / name).read_bytes() == (native / name).read_bytes()
        assert json.loads((output / "build.json").read_bytes()) == receipt
