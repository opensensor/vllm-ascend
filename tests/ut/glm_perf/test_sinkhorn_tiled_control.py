# SPDX-License-Identifier: Apache-2.0
"""Append-only frozen build and exact hardware-gate admission for normalization."""

import json
from pathlib import Path

import pytest

from tools.glm_perf import build_sinkhorn_tiled as builder
from tools.glm_perf.sinkhorn_tiled import reduction_order
from tools.glm_perf.sinkhorn_tiled_control import manifest
from tools.glm_perf.sinkhorn_tiled_probe import COUNTS, ITERATIONS, LOGIT_SCALES, verify


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    compiler = tmp_path / "compiler"
    compiler.write_text("mock; never executed")
    cann = tmp_path / "cann"
    cann.mkdir()
    monkeypatch.setattr(
        builder.subprocess, "run", lambda command, **kwargs: Path(command[2]).write_bytes(b"mock kernel")
    )
    monkeypatch.setattr(builder, "compile_bridge", lambda output, namespace, cann: output.write_bytes(b"mock bridge"))
    root = tmp_path / "build"
    builder.build(root, compiler, "glm_sinkhorn_tiled_v971", 971, cann)
    gates = dict(
        complete=True,
        provenance=verify(root),
        order=None,
        row_orders={str(rows): reduction_order(rows) for rows in COUNTS},
        records=[
            dict(
                rows=rows,
                iterations=iterations,
                logit_scale=scale,
                passed=True,
                exact_fp32=True,
                changed_replay=True,
                guards_checked=True,
            )
            for rows in COUNTS
            for iterations in ITERATIONS
            for scale in LOGIT_SCALES
        ],
    )
    report = tmp_path / "gates.json"
    report.write_text(json.dumps(gates))
    return root, report, gates


def test_complete_qualification_pins_all_frozen_assets_and_unique_namespace(frozen):
    root, report, _ = frozen
    result = manifest(root, report)
    result.verify_files()
    assert result.name == "sinkhorn_tiled_v971"
    assert result.value["operators"] == ["glm_sinkhorn_tiled_v971::launch"]
    assert {Path(x["path"]).name for x in result.value["assets"]} == {
        "glm_sinkhorn_tiled.cpp",
        "sinkhorn_tiled.py",
        "sinkhorn_tiled_probe.py",
        "reconstruction_bridge.cpp",
        "glm_sinkhorn_tiled.bin",
    }


@pytest.mark.parametrize("broken", ["complete", "rows", "replay", "order", "provenance", "asset"])
def test_partial_or_mismatched_probes_never_admit_serving(frozen, broken):
    root, report, gates = frozen
    if broken == "complete":
        gates["complete"] = False
    elif broken == "rows":
        gates["records"].pop()
    elif broken == "replay":
        gates["records"][-1]["changed_replay"] = False
    elif broken == "order":
        gates["row_orders"]["128"] = 2
    elif broken == "provenance":
        gates["provenance"]["version"] = 970
    else:
        (root / "glm_sinkhorn_tiled.bin").write_bytes(b"changed kernel")
    report.write_text(json.dumps(gates))
    with pytest.raises(ValueError):
        manifest(root, report)


def test_builder_refuses_namespace_reuse_and_existing_destination(frozen, tmp_path):
    root, _, _ = frozen
    with pytest.raises(ValueError, match="unique"):
        builder.build(tmp_path / "wrong", tmp_path / "compiler", "glm_sinkhorn_tiled_v970", 971, tmp_path / "cann")
    with pytest.raises(FileExistsError):
        builder.build(root, tmp_path / "compiler", "glm_sinkhorn_tiled_v971", 971, tmp_path / "cann")
