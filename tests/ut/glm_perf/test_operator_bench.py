# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json

import pytest
import torch

from tools.glm_perf.operator_bench import (
    WORKSPACE_OUTPUT_TILE,
    Case,
    cases,
    compare,
    compare_outputs,
    describe_package,
    route_counts,
    sha256_file,
    summarize_timings,
)


def test_case_matrix_covers_real_projections_and_route_patterns():
    assert WORKSPACE_OUTPUT_TILE * 4096 * 2 == 1024 * 1024
    matrix = cases()
    assert len(matrix) == 36
    assert {case.routed_rows for case in matrix} == {8, 32, 416}
    assert {(case.projection, case.bits, case.output_width, case.input_width) for case in matrix} == {
        ("w4_gate_up", 4, 4096, 4096),
        ("w2_down", 2, 4096, 2048),
    }
    for case in matrix:
        counts = route_counts(case)
        assert len(counts) == 72
        assert (
            sum(counts)
            == {
                "zero_local": 0,
                "peer_owned": case.routed_rows // 2,
                "distributed": case.routed_rows,
                "repeated": case.routed_rows,
                "singleton": min(72, case.routed_rows),
                "mixed": case.routed_rows,
            }[case.route_pattern]
        )
        if case.route_pattern == "repeated":
            assert counts[17] == case.routed_rows
        if case.route_pattern == "singleton":
            assert max(counts) == 1
        if case.route_pattern == "mixed":
            assert [(expert, count) for expert, count in enumerate(counts) if count] == [
                (0, 1),
                (17, case.routed_rows - 2),
                (71, 1),
            ]
    with pytest.raises(ValueError, match="prefill_rows"):
        cases(32)
    assert {case.routed_rows for case in cases(5120)} == {8, 32, 5120}
    with pytest.raises(ValueError, match="prefill_rows"):
        cases(5121)
    with pytest.raises(ValueError, match="unknown route pattern"):
        route_counts(Case("w4_gate_up", 4, 4096, 4096, 8, "unknown"))


def test_prefill_sweep_covers_all_experts_and_peer_owned_routes():
    matrix = cases(row_sizes=(512, 1024, 4096), route_patterns=("uniform_all_experts_quarter",))
    assert len(matrix) == 6
    for case in matrix:
        counts = route_counts(case)
        assert sum(counts) == case.routed_rows // 4
        assert max(counts) - min(counts) <= 1
        assert min(counts) > 0 or case.routed_rows // 4 < len(counts)
    full = route_counts(Case("w4_gate_up", 4, 4096, 4096, 4096, "uniform_all_experts"))
    assert sum(full) == 4096
    assert min(full) > 0
    with pytest.raises(ValueError, match="row_sizes"):
        cases(row_sizes=(0,))
    with pytest.raises(ValueError, match="route_patterns"):
        cases(route_patterns=("unknown",))


def test_package_hashes_binary_and_source_and_checks_search_path(tmp_path, monkeypatch):
    root = tmp_path / "opp"
    root.mkdir()
    binary = root / "w2_grouped_blocked_dequant_matmul_v310.o"
    binary.write_bytes(b"real-binary")
    binary.with_suffix(".json").write_text(
        json.dumps({"supportInfo": {"inputs": [{"name": "codes", "dtype": "int8"}]}})
    )
    source = tmp_path / "kernel.cpp"
    source.write_bytes(b"source")
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", str(root))
    package = describe_package(root, binary, [source], "nzpacked")
    assert package["binary"] == {"path": str(binary), "sha256": hashlib.sha256(b"real-binary").hexdigest()}
    assert package["sources"][0]["sha256"] == sha256_file(source)
    assert package["binary_metadata"]["code_dtype"] == "int8"
    assert package["operator"] == "npu_w2_grouped_blocked_dequant_matmul_310"
    with pytest.raises(ValueError, match="inside"):
        describe_package(root, source, [source], "nzpacked")
    with pytest.raises(ValueError, match="source"):
        describe_package(root, binary, [], "nzpacked")
    other_root = tmp_path / "other_opp"
    other_root.mkdir()
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", f"{root}:{other_root}")
    with pytest.raises(ValueError, match="only --opp-root"):
        describe_package(root, binary, [source], "nzpacked")
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", str(root))
    wrong_binary = root / "unrelated_operator.o"
    wrong_binary.write_bytes(b"unrelated")
    with pytest.raises(ValueError, match="grouped W2/W4"):
        describe_package(root, wrong_binary, [source], "nzpacked")
    fake_binary = root / "w2_grouped_blocked_dequant_matmul_v310_fake.json"
    fake_binary.write_text("{}")
    with pytest.raises(ValueError, match="compiled operator binary"):
        describe_package(root, fake_binary, [source], "nzpacked")
    with pytest.raises(ValueError, match="does not match canonical"):
        describe_package(root, binary, [source], "canonical")
    monkeypatch.delenv("ASCEND_CUSTOM_OPP_PATH")
    with pytest.raises(ValueError, match="ASCEND_CUSTOM_OPP_PATH"):
        describe_package(root, binary, [source], "nzpacked")


def test_timing_summary_and_bitwise_or_bounded_parity():
    timing = summarize_timings([1.0, 2.0, 3.0])
    assert timing["median_ms"] == 2.0
    assert timing["mean_ms"] == 2.0
    assert timing["stdev_ms"] == 1.0
    assert timing["samples_ms"] == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="nonempty"):
        summarize_timings([])
    reference = torch.tensor([1.0, -0.0], dtype=torch.float16)
    candidate = torch.tensor([1.0, 0.0], dtype=torch.float16)
    assert not compare_outputs(reference, candidate, atol=0, rtol=0)["passed"]
    assert compare_outputs(reference, candidate, atol=0.001, rtol=0)["passed"]
    with pytest.raises(ValueError, match="dtype"):
        compare_outputs(reference, candidate.float(), atol=0, rtol=0)
    with pytest.raises(ValueError, match="non-finite"):
        compare_outputs(reference, torch.tensor([float("nan"), 0], dtype=torch.float16), atol=0, rtol=0)


def _payload(binary_hash, sample, code_hash="same"):
    record = {
        "key": "w4_gate_up_8_distributed",
        "projection": "w4_gate_up",
        "bits": 4,
        "output_width": 4096,
        "input_width": 4096,
        "routed_rows": 8,
        "route_pattern": "distributed",
        "expert_counts": [8] + [0] * 71,
        "local_rows": 8,
        "output_dtype": "torch.float16",
        "logical_code_sha256": code_hash,
        "packed_code_sha256": code_hash,
        "scale_sha256": "scales",
        "input_sha256": "inputs",
        "operator_latency": {"median_ms": sample},
    }
    return {
        "schema_version": 1,
        "kind": "isolated_operator_measurement",
        "package": {"binary": {"path": binary_hash, "sha256": binary_hash}},
        "layout": "canonical",
        "seed": 1,
        "warmup": 4,
        "repeats": 20,
        "records": [record],
        "outputs": {record["key"]: torch.ones(8, 4096, dtype=torch.float16)},
    }


def test_comparison_summary_separates_latency_from_throughput(tmp_path):
    from argparse import Namespace

    before, after, output = (tmp_path / name for name in ("before.pt", "after.pt", "summary.json"))
    torch.save(_payload("aaa", 2.0), before)
    torch.save(_payload("bbb", 1.5), after)
    compare(Namespace(baseline=before, candidate=after, output=output, atol=0.0, rtol=0.0))
    summary = json.loads(output.read_text())
    assert summary["all_parity_passed"]
    assert summary["results"][0]["speedup_fraction"] == 0.25
    assert "serving throughput" in summary["metric_scope"]
    assert summary["baseline"]["package"]["binary"]["sha256"] == "aaa"
    with pytest.raises(FileExistsError):
        compare(Namespace(baseline=before, candidate=after, output=output, atol=0.0, rtol=0.0))


def test_comparison_reads_legacy_torch_version_metadata(tmp_path):
    from argparse import Namespace

    before, after, output = (tmp_path / name for name in ("before.pt", "after.pt", "summary.json"))
    baseline = _payload("aaa", 2.0)
    candidate = _payload("bbb", 1.5)
    baseline["torch"] = torch.__version__
    candidate["torch"] = torch.__version__
    torch.save(baseline, before)
    torch.save(candidate, after)

    compare(Namespace(baseline=before, candidate=after, output=output, atol=0.0, rtol=0.0))

    assert json.loads(output.read_text())["all_parity_passed"]


def test_comparison_rejects_input_and_binary_identity_mismatch(tmp_path):
    from argparse import Namespace

    before, after, output = (tmp_path / name for name in ("before.pt", "after.pt", "summary.json"))
    torch.save(_payload("aaa", 2.0), before)
    torch.save(_payload("aaa", 1.5), after)
    args = Namespace(baseline=before, candidate=after, output=output, atol=0.0, rtol=0.0)
    with pytest.raises(ValueError, match="same OPP binary"):
        compare(args)
    torch.save(_payload("bbb", 1.5, code_hash="different"), after)
    with pytest.raises(ValueError, match="logical_code_sha256"):
        compare(args)
    wrong_geometry = _payload("bbb", 1.5)
    wrong_geometry["records"][0]["input_width"] = 2048
    torch.save(wrong_geometry, after)
    with pytest.raises(ValueError, match="input_width"):
        compare(args)
    wrong_routes = _payload("bbb", 1.5)
    wrong_routes["records"][0]["local_rows"] = 4
    torch.save(wrong_routes, after)
    with pytest.raises(ValueError, match="local_rows"):
        compare(args)
