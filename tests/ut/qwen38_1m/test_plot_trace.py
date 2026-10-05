# SPDX-License-Identifier: Apache-2.0
"""Host-only checks for standalone Qwen trace figures."""

import pytest

pytest.importorskip("matplotlib")

from tools.qwen4exp.plot_trace import (
    load_tasks,
    merge_intervals,
    plot_host_copy_tail,
    plot_host_ops,
    plot_mix,
    plot_occupancy,
    plot_timeline,
    plot_top,
)


def _write_trace(path, device):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Device_id,Name,Start Time(us),Duration(us)\n"
        f"{device},QwenW4A8Int4MatmulV310,1000000.000,500.000\n"
        f"{device},QwenQSASelectV310,1000400.000,150.000\n"
    )


def test_plot_trace_writes_pngs_from_two_device_timelines(tmp_path):
    root = tmp_path / "traces"
    _write_trace(root / "rank0" / "kernel_details.csv", 0)
    _write_trace(root / "rank1" / "kernel_details.csv", 1)
    traces = load_tasks(root)
    assert [label for label, _ in traces] == ["NPU 0", "NPU 1"]
    assert len(traces[0][1]) == 2

    for name, render in (
        ("operator-mix.png", plot_mix),
        ("top-kernels.png", plot_top),
        ("task-timeline.png", plot_timeline),
        ("device-occupancy.png", plot_occupancy),
    ):
        output = tmp_path / name
        render(traces, output)
        assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    host_csv = tmp_path / "operator_details.csv"
    host_csv.write_text("Name,Host Self Duration(us)\naclnnInplaceCopy,100.0\nEvent::synchronize,200.0\n")
    host_png = tmp_path / "host-ops.png"
    plot_host_ops(host_csv, host_png)
    assert host_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    tail_png = tmp_path / "host-copy-tail.png"
    plot_host_copy_tail(host_csv, tail_png)
    assert tail_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_overlapping_kernels_count_as_one_busy_interval():
    tasks = [
        {"start": 0, "duration": 20},
        {"start": 10, "duration": 15},
        {"start": 40, "duration": 10},
    ]
    assert merge_intervals(tasks) == [(0, 25), (40, 50)]


def test_plot_trace_rejects_mixed_devices_in_one_worker_file(tmp_path):
    path = tmp_path / "kernel_details.csv"
    _write_trace(path, 0)
    with path.open("a") as file:
        file.write("1,QwenW4A8Int4MatmulV310,1000550.000,10.000\n")
    with pytest.raises(ValueError, match="multiple device IDs"):
        load_tasks(tmp_path)
