# SPDX-License-Identifier: Apache-2.0
"""Catch deployment failures before loading weights or initializing an NPU."""

import hashlib
import json

import pytest

from tools.qwen4exp.single_card_preflight import (
    CORE_PACKAGES,
    HCCL_SYMBOLS,
    compare_packages,
    preflight,
    prepare_build_info,
    sdk_checks,
)


@pytest.fixture
def deployment(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "runtime/vllm_ascend").mkdir(parents=True)
    expected = {name: "1.0" for name in CORE_PACKAGES}
    (bundle / "environment.json").write_text(json.dumps({"python": "3.12.13"}))
    (bundle / "python-packages.json").write_text(
        json.dumps([{"name": name, "version": version} for name, version in expected.items()])
    )
    content = b"frozen asset"
    (bundle / "asset.bin").write_bytes(content)
    (bundle / "files.sha256.json").write_text(json.dumps({"asset.bin": hashlib.sha256(content).hexdigest()}))
    sdk = tmp_path / "sdk"
    for directory in ("opp", "devlib", "compiler/bin", "python/site-packages/tbe/tvm/topi", "lib64"):
        (sdk / directory).mkdir(parents=True)
    (sdk / "compiler/bin/asc_opc").write_bytes(b"fake compiler entry point")
    for name in ("libascendcl.so", "libhccl.so", "libatb.so", "libacl_dvpp.so"):
        (sdk / "lib64" / name).write_bytes(b"fake ELF; symbols supplied by test")
    environment = {
        "ASCEND_HOME_PATH": str(sdk),
        "LD_LIBRARY_PATH": str(sdk / "lib64"),
        "VLLM_USE_V2_MODEL_RUNNER": "1",
    }
    return bundle, sdk, environment, expected


def check(deployment, **changes):
    bundle, sdk, environment, installed = deployment
    return preflight(
        bundle,
        environment=environment,
        installed=installed,
        machine="x86_64",
        python_version="3.12.13",
        symbol_reader=lambda path: set(HCCL_SYMBOLS),
        **changes,
    )


def test_omitted_metadata_is_detected_and_prepared_with_provenance(deployment):
    bundle = deployment[0]
    assert check(deployment)["status"] == "blocked"
    original = (bundle / "files.sha256.json").read_bytes()
    assert prepare_build_info(bundle)
    assert not prepare_build_info(bundle)
    assert (bundle / "files.sha256.json").read_bytes() == original
    assert json.loads((bundle / "build-info-preparation.json").read_text())["verified_original_files"] == 1
    assert check(deployment)["status"] == "passed"


def test_prepare_refuses_tampered_original(deployment):
    bundle = deployment[0]
    (bundle / "asset.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="mismatch"):
        prepare_build_info(bundle)
    assert not (bundle / "runtime/vllm_ascend/_build_info.py").exists()


def test_prepare_never_overwrites_another_hardware_build(deployment):
    bundle = deployment[0]
    (bundle / "runtime/vllm_ascend/_build_info.py").write_text("__device_type__ = 'A3'\n")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        prepare_build_info(bundle)


def test_critical_package_and_triton_drift_block_the_frozen_profile():
    errors, warnings = compare_packages(
        {"torch_npu": "2.13", "psutil": "7.2"},
        {"torch-npu": "2.10", "psutil": "7.0", "triton-ascend": "3.2.2", "harmless-extra": "1"},
        CORE_PACKAGES,
    )
    assert len(errors) == 2
    assert any("psutil" in warning for warning in warnings)
    assert any("harmless-extra" in warning for warning in warnings)


def test_missing_core_package_is_distinct_from_missing_auxiliary_package():
    errors, warnings = compare_packages({"torch": "2.13", "py-spy": "0.4"}, {}, CORE_PACKAGES)
    assert errors == ["torch: expected 2.13; installed MISSING"]
    assert warnings == ["py-spy: expected 0.4; installed MISSING"]


def test_selected_hccl_family_must_have_required_symbols(deployment):
    _, sdk, environment, _ = deployment
    overlay = sdk / "overlay"
    overlay.mkdir()
    (overlay / "libhccl.so").write_bytes(b"selected overlay")
    environment["LD_LIBRARY_PATH"] = f"{overlay}:{sdk / 'lib64'}"
    inspected = []

    def symbols(path):
        inspected.append(path)
        return {"HcclBroadcastV2"}

    result = sdk_checks(environment, symbols, dependency_reader=lambda path: [])
    assert inspected == [(overlay / "libhccl.so").resolve()]
    assert any("HcclBroadcast" in error for error in result["errors"])


def test_preload_and_incomplete_compiler_are_detected(deployment):
    _, sdk, environment, _ = deployment
    environment["LD_PRELOAD"] = "/elsewhere/libhccl.so"
    (sdk / "compiler/bin/asc_opc").unlink()
    result = sdk_checks(environment, lambda path: set(HCCL_SYMBOLS))
    assert any("LD_PRELOAD" in error for error in result["errors"])
    assert any("compiler entry point" in error for error in result["errors"])


def test_hccl_root_info_can_be_exported_by_a_needed_companion(deployment):
    _, sdk, environment, _ = deployment
    companion = sdk / "lib64/libhccl_compat.so"
    companion.write_bytes(b"fake companion ELF")

    def symbols(path):
        return {"HcclGetRootInfo"} if path == companion.resolve() else {"HcclAllReduce", "HcclBroadcast"}

    result = sdk_checks(environment, symbols, dependency_reader=lambda path: ["libhccl_compat.so"])
    assert not result["errors"]
    assert str(companion.resolve()) in result["hccl_symbol_files"]


def test_wrong_runner_is_not_hidden_by_matching_flags(deployment):
    bundle, _, environment, _ = deployment
    prepare_build_info(bundle)
    environment["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    result = check(deployment)
    assert any("VLLM_USE_V2_MODEL_RUNNER" in error for error in result["errors"])
    assert result["environment"]["VLLM_USE_V2_MODEL_RUNNER"] == "0"
