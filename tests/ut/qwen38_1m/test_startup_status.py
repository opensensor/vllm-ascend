# SPDX-License-Identifier: Apache-2.0
"""Check reported startup phases and bounded conversion estimates."""

from tools.qwen4exp.startup_status import summarize


def test_bank_progress_estimates_conversion_only_after_two_samples():
    line = "(Worker_TP0_EP0 pid=10) INFO INT8_EXPERT_CONVERSION bank_done layer=0 bank=gate_up_proj elapsed_s=2.0"
    report = summarize(line)
    assert report["phase"] == "converting expert weights"
    assert report["ranks"][0]["conversion_remaining_s"] is None
    report = summarize(line + "\n" + line.replace("gate_up_proj", "down_proj").replace("2.0", "4.0"))
    assert report["ranks"][0]["banks_done"] == 2
    assert report["ranks"][0]["conversion_remaining_s"] == 94 * 3


def test_repeated_messages_do_not_overcount_banks():
    line = "(Worker_TP2_EP2 pid=10) INFO INT8_EXPERT_CONVERSION bank_done layer=4 bank=down_proj elapsed_s=2.0"
    report = summarize(line + "\n" + line)
    assert report["ranks"][0]["banks_done"] == 1


def test_failure_overrides_api_ready():
    assert summarize("Application startup complete\nWorkerProc hit an exception")["phase"] == "failed"


def test_expert_progress_without_completed_bank():
    report = summarize("(Worker_TP4_EP4 pid=10) INFO INT8_EXPERT_CONVERSION layer=0 bank=gate_up_proj experts=32/85")
    assert report["ranks"][0]["current"] == {"layer": 0, "bank": "gate_up_proj", "experts": 32, "total": 85}
    assert report["ranks"][0]["banks_done"] == 0


def test_saved_bank_counter_does_not_claim_old_cache_as_new_progress(tmp_path):
    import os

    from tools.qwen4exp.startup_status import cache_progress

    rank = tmp_path / "rank-0"
    rank.mkdir()
    for name, modified in (("layer-00-gate_up_proj", 10), ("layer-00-down_proj", 20)):
        path = rank / (name + ".safetensors")
        path.write_bytes(b"test")
        os.utime(path, (modified, modified))
    report = cache_progress(tmp_path, started_after=0)
    assert report[0]["new_banks"] == 2
    assert report[0]["conversion_remaining_s"] == 940
    assert len(report) == 6 and report[2]["new_banks"] == 0
    report = cache_progress(tmp_path, started_after=30)
    assert report[0]["cached_banks"] == 2 and report[0]["new_banks"] == 0
    assert report[0]["conversion_remaining_s"] is None
