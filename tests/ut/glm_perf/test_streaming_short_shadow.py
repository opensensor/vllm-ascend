# SPDX-License-Identifier: Apache-2.0
"""Reject partial actual-input shadow coverage and changed observer boundaries."""

import copy

import pytest

from tools.glm_perf.streaming_short_shadow import check_comparisons, extend_shadow_source


def workers():
    return [
        {
            "rank": rank,
            "streaming": {
                "shadow": True,
                "mismatches": 0,
                "cases": [
                    {"bank": bank, "tokens": tokens, "finite": True, "bitwise_equal": True}
                    for tokens in (128, 385)
                    for bank in range(43)
                ],
                "whole_moe": [{"tokens": 385, "finite": True, "bitwise_equal": True}] * 43,
            },
        }
        for rank in range(4)
    ]


def test_observer_expands_inside_sealed_source_without_changing_dispatch():
    source = "capsule = 'SHADOW_CHUNKS = (640, 1280)\\nunchanged_dispatch = True'\n"
    changed = extend_shadow_source(source)
    assert changed == source.replace("SHADOW_CHUNKS = (640, 1280)", "SHADOW_CHUNKS = tuple(range(17, 1281))")


@pytest.mark.parametrize("source", ["pass", "SHADOW_CHUNKS = (640, 1280)\n" * 2])
def test_changed_or_ambiguous_observer_refused(source):
    with pytest.raises(ValueError, match="boundary"):
        extend_shadow_source(source)


def test_real_rank_bank_and_mixed_shape_coverage():
    result = check_comparisons(workers())
    assert len(result) == 4
    assert all(row["observed_rows"] == [128, 385] and row["native_comparisons"] == 86 for row in result)


@pytest.mark.parametrize("field,value", [("finite", False), ("bitwise_equal", False), ("tokens", 8), ("tokens", 1281)])
@pytest.mark.parametrize("kind", ["cases", "whole_moe"])
def test_failed_or_decode_comparisons_refused(kind, field, value):
    rows = copy.deepcopy(workers())
    rows[2]["streaming"][kind][0][field] = value
    with pytest.raises(ValueError, match="comparison"):
        check_comparisons(rows)


@pytest.mark.parametrize("fault", ["missing_rank", "duplicate_rank", "missing_bank", "empty_whole", "mismatch"])
def test_incomplete_coverage_refused(fault):
    rows = workers()
    if fault == "missing_rank":
        rows.pop()
    elif fault == "duplicate_rank":
        rows[3]["rank"] = 2
    elif fault == "missing_bank":
        rows[1]["streaming"]["cases"] = [c for c in rows[1]["streaming"]["cases"] if c["bank"] != 42]
    elif fault == "empty_whole":
        rows[1]["streaming"]["whole_moe"] = []
    else:
        rows[1]["streaming"]["mismatches"] = 1
    with pytest.raises(ValueError):
        check_comparisons(rows)


def test_large_prompt_only_cannot_qualify_short_prefill():
    rows = workers()
    for row in rows:
        for case in row["streaming"]["cases"]:
            case["tokens"] = 640
    with pytest.raises(ValueError):
        check_comparisons(rows)


@pytest.mark.parametrize("mismatch", [False, True])
def test_guarded_workflow_restores_and_releases_after_success_or_mismatch(tmp_path, monkeypatch, mismatch):
    from test_streaming_resident import workflow_controller

    from tools.glm_perf.streaming_protocol import file_digest, make_workloads
    from tools.glm_perf.streaming_short_shadow import workflow

    guard, client = workflow_controller(tmp_path)
    original = client.stream

    def stream(*args, **kwargs):
        sample = original(*args, **kwargs)
        observed = workers()
        for row in client.workers:
            for field in ("cases", "whole_moe", "mismatches"):
                row["streaming"][field] = copy.deepcopy(observed[row["rank"]]["streaming"][field])
            if mismatch:
                row["streaming"]["cases"][0]["bitwise_equal"] = False
        return sample

    monkeypatch.setattr(client, "stream", stream)
    control = next(row for row in make_workloads()["workloads"] if row["id"] == "decode_c4")
    if mismatch:
        with pytest.raises(ValueError, match="comparison"):
            workflow(guard, control)
    else:
        report = workflow(guard, control)
        assert report["state"] == "complete" and len(report["samples"]) == 4
        assert report["observer"]["source_sha256"] == file_digest(guard.output_dir / "short-shadow-source.py.txt")
    assert len(client.inference) == 4 and all(call["payload"]["max_tokens"] == 16 for call in client.inference)
    assert client.version == 1021 and not client.maintenance and not client.paused
    assert (guard.output_dir / "short-shadow.json").is_file()
