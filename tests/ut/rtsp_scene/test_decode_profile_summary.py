import pytest

from tools.rtsp_scene.decode_profile_summary import summarize_text_decode


def task(kind, start, duration, device="0"):
    return {"Type": kind, "Start Time(us)": str(start), "Duration(us)": str(duration), "Device_id": device}


def test_excludes_prefill_and_interleaved_decode_and_counts_overlaps():
    rows = [task("rtsp_dense_i8", 0, 1), task("QuantBatchMatmulV3", 2, 8)]
    rows += [task("rtsp_dense_i8", 11 + i, 2) for i in range(4)]
    rows += [task("rtsp_nz_swiglu", 15, 1), task("MatMulV2", 16, 2)]
    result = summarize_text_decode(rows, layers=1)
    assert result["decode_forwards"] == 1 and result["custom_matrices"] == 4
    assert result["active_union_ms"] == 0.007 and result["task_sum_ms"] == 0.011
    assert result["profiled_not_throughput_baseline"]


def test_timestamp_offsets_keep_sub_microsecond_precision():
    base = "1791923146924578"
    rows = [task("QuantBatchMatmulV3", base + ".000", ".125")]
    rows += [task("rtsp_dense_i8", base + ".250", ".125") for _ in range(4)]
    rows += [task("rtsp_nz_swiglu", base + ".375", ".125")]
    result = summarize_text_decode(rows, layers=1)
    assert result["active_union_ms"] == 0.00025


@pytest.mark.parametrize("bad", ["device", "partial", "activation", "boundary"])
def test_reject_unproven_decode_phase(bad):
    rows = [task("QuantBatchMatmulV3", 0, 1)]
    rows += [task("rtsp_dense_i8", 2, 1) for _ in range(4)]
    rows += [task("rtsp_nz_swiglu", 3, 1)]
    if bad == "device":
        rows[-1]["Device_id"] = "1"
    elif bad == "partial":
        rows.pop(1)
    elif bad == "activation":
        rows.pop()
    else:
        rows.pop(0)
    with pytest.raises(ValueError):
        summarize_text_decode(rows, layers=1)
