# SPDX-License-Identifier: Apache-2.0
"""Private build provenance and failure retention; no accelerator runtime."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.sidecard import pair_add_build as module


def setup(tmp_path, monkeypatch, *, fail_at=None, changed=False, malformed=False):
    sdk = tmp_path / "sdk"
    for relative in (
        "lib64/libascendcl.so",
        "lib64/libacl_rtc.so",
        "include/acl/acl.h",
        "include/acl/acl_rt.h",
        "include/acl/acl_rt_compile.h",
    ):
        path = sdk / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sdk-input")
    compiler = tmp_path / "compiler"
    compiler.write_bytes(b"compiler-input")
    monkeypatch.setattr(module.shutil, "which", lambda _: str(compiler))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == fail_at:
            return SimpleNamespace(returncode=7)
        product = Path(command[-1] if len(calls) <= 2 else command[2])
        product.write_bytes(b"bad" if malformed else b"\x7fELF" + bytes(28))
        if changed:
            (sdk / "include/acl/acl.h").write_bytes(b"changed-sdk")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    return sdk, calls


def test_fixed_targets_and_private_source_snapshot(tmp_path, monkeypatch):
    sdk, calls = setup(tmp_path, monkeypatch)
    output = tmp_path / "out"
    receipt = module.build(sdk, output)
    assert receipt["build_complete"] and not receipt["hardware_validated"]
    assert receipt["completed_commands"] == 5
    assert set(receipt["source_sha256"]) == set(module.SOURCES)
    assert len(receipt["products_sha256"]) == 4
    assert all("--npu-arch=dav-2002" in c for c in calls[2:])
    assert [c[-1] for c in calls[2:]] == ["-DSIDECARD_PAIR_ROWS=1", "-DSIDECARD_PAIR_ROWS=4", "-DSIDECARD_PAIR_ROWS=32"]
    assert not any("SetDevice" in arg or "opp" in arg for c in calls for arg in c)
    assert json.loads((output / "build.json").read_text()) == receipt
    with pytest.raises(FileExistsError):
        module.build(sdk, output)


@pytest.mark.parametrize("fail_at", (1, 2, 3, 4, 5))
def test_failed_command_stops_build_and_preserves_receipt(tmp_path, monkeypatch, fail_at):
    sdk, calls = setup(tmp_path, monkeypatch, fail_at=fail_at)
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="build failed"):
        module.build(sdk, output)
    receipt = json.loads((output / "build.json").read_text())
    assert len(calls) == fail_at
    assert receipt["returncode"] == 7 and receipt["completed_commands"] == fail_at - 1
    assert not receipt["build_complete"]
    assert (output / "source/pair_add_kernel.cpp").is_file()


@pytest.mark.parametrize("changed,malformed", ((True, False), (False, True)))
def test_input_mutation_and_non_elf_product_rejected(tmp_path, monkeypatch, changed, malformed):
    sdk, _ = setup(tmp_path, monkeypatch, changed=changed, malformed=malformed)
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="build failed"):
        module.build(sdk, output)
    assert not json.loads((output / "build.json").read_text())["build_complete"]


def test_missing_sdk_input_rejected_before_output(tmp_path, monkeypatch):
    sdk, calls = setup(tmp_path, monkeypatch)
    (sdk / "include/acl/acl_rt_compile.h").unlink()
    with pytest.raises(ValueError, match="SDK inputs"):
        module.build(sdk, tmp_path / "out")
    assert not calls and not (tmp_path / "out").exists()
