# SPDX-License-Identifier: Apache-2.0
"""Report generation must retain failures and separate precision diagnostics."""

import csv
import json

import pytest

from tools.raw_math.report import RECEIPTS, export_csv, load_receipts, select


def row(*, status="passed", validated=True):
    return dict(
        name="vector",
        op="gemm",
        dtype="fp32",
        m=1,
        n=4,
        k=8,
        layout="ND",
        pinned=False,
        status=status,
        error="unsupported" if status == "failed" else "",
        initial_validation=dict(passed=validated),
        changed_input_validation=dict(passed=validated),
        final_validation=dict(passed=validated),
        timings=dict(
            single=dict(
                repeats_per_sample=1,
                device=dict(median_ms=1),
                wall=dict(median_ms=2),
                rates=dict(tera_ops_per_second=0.01, operations="floating multiply/add"),
            )
        ),
    )


def test_exports_failed_rows_and_marks_rounding_diagnostics(tmp_path):
    data = dict(completed=True, backend="npu", reference_fp16_inputs=True, cases=[row(), row(status="failed")])
    path = tmp_path / "results.csv"
    export_csv({"diagnostic": data}, path)
    assert b"\r" not in path.read_bytes()
    with path.open() as file:
        records = list(csv.DictReader(file))
    assert len(records) == 2
    assert records[0]["reference_fp16_inputs"] == "True"
    assert records[0]["device_median_ms"] == "1"
    assert records[1]["status"] == "failed"
    assert records[1]["device_median_ms"] == ""
    assert records[1]["error"] == "unsupported"


def test_select_cannot_hide_duplicate_or_failed_measurements():
    assert select(dict(cases=[row()]), "vector", "fp32")["status"] == "passed"
    for cases in ([row(), row()], [row(status="failed")], []):
        with pytest.raises(ValueError, match="unique validated"):
            select(dict(cases=cases), "vector", "fp32")


@pytest.mark.parametrize("failure", ["incomplete", "invalid_gate"])
def test_receipt_loading_rejects_incomplete_or_invalid_gates(tmp_path, failure):
    for key, filename in RECEIPTS.items():
        report = dict(completed=True, backend="npu", cases=[] if key == "npu_card" else [row()])
        if key == "cuda_matrix":
            if failure == "incomplete":
                report["completed"] = False
            else:
                report["cases"] = [row(validated=False)]
        (tmp_path / filename).write_text(json.dumps(report))
    with pytest.raises(ValueError):
        load_receipts(tmp_path)
