"""CPU checks for GLM all-rank profiler attribution."""

import csv
import json

import pytest

from tools.glm_perf.analyze_trace import (
    analyze,
    kernel_category,
    microseconds_to_ns,
    read_collectives,
    read_kernels,
    union_ns,
)


def _rank_export(root, rank, token, tasks, collective):
    directory = root / f"dp0_rank{rank}_{token}_ascend_pt" / "ASCEND_PROFILER_OUTPUT"
    directory.mkdir(parents=True)
    with (directory / "kernel_details.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=["Device_id", "Name", "Start Time(us)", "Duration(us)", "Input Shapes"]
        )
        writer.writeheader()
        for name, start, duration in tasks:
            writer.writerow(
                {"Device_id": rank, "Name": name, "Start Time(us)": start, "Duration(us)": duration, "Input Shapes": ""}
            )
    (directory / "communication.json").write_text(
        json.dumps(
            {
                "step": {
                    "collective": {
                        "hcom_allReduce__0_0_1@0": {
                            "Communication Time Info": {
                                "Start Timestamp(us)": collective,
                                "Elapse Time(ms)": 0.2,
                                "Wait Time(ms)": 0.15,
                                "Transit Time(ms)": 0.02,
                            }
                        }
                    }
                }
            }
        )
    )


def test_four_rank_windows_separate_elapsed_time_from_summed_tasks(tmp_path):
    start = 1790795144682500
    for rank in range(4):
        _rank_export(
            tmp_path,
            rank,
            "capture_a",
            [
                ("W2GroupedBlockedDequantMatmul", start, 1000 + rank * 100),
                ("hcom_allReduce", start + 400 + rank * 20, 200),
                ("npu_qsa_sparse_attention_310", start + 2000, 500),
            ],
            start + 400 + rank * 20,
        )
    windows = [
        {"label": "decode_step_0", "start_us": start, "end_us": start + 1800},
        {"label": "decode_step_1", "start_us": start + 1800, "end_us": start + 2600},
    ]
    report = analyze(tmp_path, "capture_a", windows)
    first, second = report["windows"]
    assert first["tail_rank"] == 3
    assert first["cross_rank_task_envelope_ms"] == pytest.approx(1.3)
    assert first["ranks"]["0"]["summed_task_ms"] > first["ranks"]["0"]["task_union_ms"]
    assert first["ranks"]["0"]["categories"]["grouped_w2_w4"]["count"] == 1
    assert first["ranks"]["0"]["categories"]["grouped_w2_w4"]["task_union_ms"] == pytest.approx(1.0)
    assert first["ranks"]["0"]["top_kernel_shapes"][0]["name"] == "W2GroupedBlockedDequantMatmul"
    assert first["collectives_matched_all_ranks"] == 1
    assert first["collective_arrival_spread_max_ms"] == pytest.approx(0.06)
    assert first["ranks"]["0"]["collective_summed_wait_ms"] == pytest.approx(0.15)
    assert first["ranks"]["0"]["collective_summed_transit_ms"] == pytest.approx(0.02)
    assert second["ranks"]["2"]["categories"]["sparse_attention"]["count"] == 1
    assert second["ranks"]["2"]["collective_count"] == 0


def test_rank_validation_and_exact_epoch_conversion(tmp_path):
    assert microseconds_to_ns("1790795144682503.331\t") == 1790795144682503331
    assert union_ns([(0, 10), (5, 15), (20, 25)]) == 20
    _rank_export(tmp_path, 0, "one", [("other", 1, 1)], 1)
    with pytest.raises(ValueError, match="expected ranks"):
        analyze(tmp_path, "one", [])


def test_default_window_keeps_epoch_microseconds_exact(tmp_path):
    start = "1790795144682503.331"
    for rank in range(4):
        _rank_export(tmp_path, rank, "exact", [("other", start, "2.031")], start)
    window = analyze(tmp_path, "exact", [])["windows"][0]
    assert window["window_start_us"] == start
    assert window["window_end_us"] == "1790795144682505.362"
    assert window["cross_rank_task_envelope_ms"] == pytest.approx(0.002031)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("MhcSinkhornV310", "mhc_sinkhorn"),
        ("aclnnTopK", "topk_all"),
        ("aclnnBitwiseAnd", "integer_bit_ops"),
        ("aclnnRightShift", "integer_bit_ops"),
        ("aclnnMatmul", "other_matmul"),
        ("W2GroupedBlockedDequantMatmul", "grouped_w2_w4"),
    ],
)
def test_graph_trace_name_categories(name, expected):
    assert kernel_category(name) == expected


def test_ai_cpu_cast_is_separate_from_device_cast():
    assert kernel_category("Cast", "AI_CPU") == "ai_cpu_cast"
    assert kernel_category("Cast", "AI_CORE") == "other"


def test_cann_truncated_last_kernel_row_is_ignored(tmp_path):
    path = tmp_path / "kernel_details.csv"
    path.write_text("Device_id,Name,Start Time(us),Duration(us)\n0,valid,1791076846137613.608\t,0.99\n0,incomplete,,\n")
    assert [event["name"] for event in read_kernels(path)] == ["valid"]


def test_cann_total_row_is_not_a_collective(tmp_path):
    path = tmp_path / "communication.json"
    path.write_text(
        json.dumps(
            {
                "step": {
                    "collective": {
                        "Total Op Info": {"Communication Time Info": {"Elapse Time(ms)": 100}},
                        "hcom_allReduce__0_0_1@0": {
                            "Communication Time Info": {
                                "Start Timestamp(us)": "100.25",
                                "Elapse Time(ms)": 0.2,
                                "Wait Time(ms)": 0.15,
                                "Transit Time(ms)": 0.02,
                            }
                        },
                    }
                }
            }
        )
    )
    collectives = read_collectives(path)
    assert len(collectives) == 1
    assert collectives[0]["key"] == "hcom_allReduce__0_0_1"


@pytest.mark.parametrize(
    "name,core,expected",
    [
        ("glm_fused_gate_up_w4_v1", "AI_CORE", "moe_gate_up"),
        ("glm_fused_gate_up_v1", "AI_CORE", "moe_gate_up"),
        ("glm_fused_down_v1", "AI_CORE", "moe_down"),
        ("glm_fused_reduce_half_v1", "AI_CORE", "moe_reduce"),
        ("glm_fused_route_input_v1", "AI_CORE", "moe_prepare"),
        ("glm_fused_pack_v1", "AI_CORE", "moe_prepare"),
        ("aclnnInplaceCopy_CastAiCpu_Cast", "AI_CPU", "ai_cpu_cast"),
        ("aclnnMatmul_CastAiCore_Cast", "AI_CORE", "other_matmul"),
        ("Broadcast", "AI_CPU", "other"),
    ],
)
def test_actual_native_stage_names_and_cpu_casts(name, core, expected):
    assert kernel_category(name, core) == expected
