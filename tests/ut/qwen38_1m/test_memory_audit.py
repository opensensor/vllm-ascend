# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline coverage and provenance regression checks for the audit inventory."""

import hashlib
import runpy
from pathlib import Path

import pytest

from tools.qwen4exp.memory_audit import native_sites, python_sites, scan_root


def test_python_ignores_comments_and_strings_and_keeps_nested_scope():
    source = """# x.cpu()
class Model:
    def forward(self, x):
        message = "x.item()"
        return x.to("cpu").tolist()
"""
    sites = python_sites(source, "model.py")
    assert [site["operation"] for site in sites] == ["tolist", "to"]
    assert all(site["scope"] == "Model.forward" and site["line"] == 5 for site in sites)
    assert {category for site in sites for category in site["categories"]} == {"host_boundary", "copy_or_cast"}


@pytest.mark.parametrize(
    "operation,category",
    [
        ("wait_event", "barrier_or_stream"),
        ("all_reduce", "collective"),
        ("nonzero", "dynamic_shape"),
        ("contiguous", "materialization"),
        ("empty_like", "allocation_or_fill"),
    ],
)
def test_candidate_categories_do_not_claim_a_transfer_direction(operation, category):
    site = python_sites(f"torch.{operation}(x)", "candidate.py")[0]
    assert site["categories"] == [category]
    assert "direction" not in site and "device" not in site


def test_native_comments_and_strings_do_not_create_false_barriers():
    source = """// DataCopy(x,y,4);
/* PipeBarrier<PIPE_ALL>(); */
const char *message = "aclrtMemcpy(x,y)";
DataCopy(x,y,4); PipeBarrier<PIPE_ALL>();
aclrtSynchronizeStream(stream);
"""
    sites = native_sites(source)
    assert [(site["operation"], site["line"]) for site in sites] == [
        ("DataCopy", 4),
        ("PipeBarrier", 4),
        ("aclrtSynchronizeStream", 5),
    ]


def test_scan_manifest_includes_zero_site_sources_and_reports_parse_failures(tmp_path):
    (tmp_path / "plain.py").write_text("VALUE = 1\n")
    (tmp_path / "broken.py").write_text("def broken(\n")
    raw = b"def call(x):\n    return x.cpu()\n"
    (tmp_path / "call.py").write_bytes(raw)
    (tmp_path / "binary.so").write_bytes(b"not source")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "ignored.py").write_text("x.item()")
    (tmp_path / "alias.py").symlink_to(tmp_path / "call.py")
    manifest, sites, errors = scan_root(tmp_path, "snapshot")
    assert len(manifest) == 3 and len(sites) == 1 and len(errors) == 1
    assert sites[0]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert sites[0]["root"] == "snapshot" and sites[0]["scope"] == "call"
    assert errors[0]["path"] == "broken.py"


def test_profiler_receipt_preserves_self_time_and_filters_other_operations():
    script = Path(__file__).resolve().parents[3] / "artifacts/qwen38-memory-audit-20261008/reproduce_evidence.py"
    receipt = runpy.run_path(str(script))["host_receipt"]
    raw = (
        b"Name,Host Self Duration(us),Host Total Duration(us),Device Self Duration(us)\n"
        b"Event::synchronize,100,900,0\nEvent::synchronize,200,900,0\naten::matmul,999,999,999\n"
    )
    result = receipt(raw, "trusted.csv")
    assert result["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["operations"] == {
        "Event::synchronize": {"count": 2, "host_self_ms": pytest.approx(0.3), "device_self_ms": 0.0}
    }


@pytest.mark.parametrize("duration", ["nan", "-1"])
def test_profiler_receipt_rejects_unusable_durations(duration):
    script = Path(__file__).resolve().parents[3] / "artifacts/qwen38-memory-audit-20261008/reproduce_evidence.py"
    receipt = runpy.run_path(str(script))["host_receipt"]
    raw = f"Name,Host Self Duration(us),Device Self Duration(us)\nEvent::synchronize,{duration},0\n".encode()
    with pytest.raises(ValueError, match="Invalid profiler duration"):
        receipt(raw, "invalid.csv")
