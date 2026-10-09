# SPDX-License-Identifier: Apache-2.0

import io
import tarfile
from decimal import Decimal

import pytest

from tools.glm_perf.memory_peak_audit import Allocation, audit_archive, audit_device, read_allocations


def allocation(name, size, start, end, peak, released=None, device="NPU:0"):
    return Allocation(
        name,
        device,
        Decimal(size),
        Decimal(start),
        Decimal(end) if end else None,
        Decimal(peak),
        Decimal(released) if released else None,
        Decimal(9000),
    )


def test_other_operator_limits_peak_saving_and_reserved_is_not_allocated():
    rows = [allocation("mixer", 480, 10, 20, 4100, 3620), allocation("matmul", 220, 30, 40, 4070, 3850)]
    result = audit_device(rows, selected_name="mixer", minimum_mib=Decimal(400))
    assert result["peak_allocated_mib"] == 4100
    assert result["peak_owner_reserved_at_allocation_mib"] == 9000
    assert result["ideal_snapshot_peak_without_complete_selected_buffers_mib"] == 4070
    assert result["ideal_snapshot_peak_reduction_mib"] == 30
    assert result["peak_outside_complete_selected_lifetimes"]["snapshot_owner"]["name"] == "matmul"


def test_overlapping_selected_lifetimes_subtract_both_at_peak():
    rows = [allocation("mixer", 480, 10, 30, 4000, 4000), allocation("mixer", 480, 20, 25, 4480, 4000)]
    result = audit_device(rows, selected_name="mixer", minimum_mib=Decimal(400))
    assert result["ideal_snapshot_peak_reduction_mib"] == 480
    assert len(result["largest_recorded_allocations_live_at_peak"]) == 2


def test_incomplete_selected_lifetime_is_not_assumed_removable():
    rows = [allocation("mixer", 480, 10, None, 4100)]
    result = audit_device(rows, selected_name="mixer", minimum_mib=Decimal(400))
    assert result["selected_records"] == 1
    assert result["selected_complete_lifetimes"] == 0
    assert result["incomplete_lifetimes"] == 1
    assert result["ideal_snapshot_peak_reduction_mib"] == 0


def test_epoch_timestamp_precision_and_tab_suffix():
    text = (
        "Name,Device Type,Size(KB),Allocation Time(us),Release Time(us),Allocation Total Allocated(MB)\n"
        "mixer,NPU:0,491520,1791506401715928.118\t,1791506401715928.119\t,4100\n"
    )
    row = read_allocations(io.StringIO(text))[0]
    assert row.live_at(Decimal("1791506401715928.1185"))
    assert not row.live_at(Decimal("1791506401715928.119"))
    assert row.size_mib == 480


@pytest.mark.parametrize("size,start,end", [("-1", "10", "11"), ("NaN", "10", "11"), ("1", "11", "10")])
def test_malformed_record_rejected(size, start, end):
    text = f"Name,Device Type,Size(KB),Allocation Time(us),Release Time(us)\nmixer,NPU:0,{size},{start},{end}\n"
    with pytest.raises(ValueError):
        read_allocations(io.StringIO(text))


def test_archive_audits_devices_separately_without_extracting_members(tmp_path):
    csv = (
        b"Name,Device Type,Size(KB),Allocation Time(us),Release Time(us),Allocation Total Allocated(MB)\n"
        b"mixer,NPU:0,491520,10,20,4100\n"
        b"mixer,NPU:1,491520,10,20,5100\n"
        b"matmul,NPU:0,1024,30,40,4000\n"
    )
    path = tmp_path / "memory.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("../../must-not-extract/operator_memory.csv")
        info.size = len(csv)
        archive.addfile(info, io.BytesIO(csv))
    result = audit_archive(path, selected_name="mixer")
    assert [table["device"] for table in result["tables"]] == ["NPU:0", "NPU:1"]
    assert [table["peak_allocated_mib"] for table in result["tables"]] == [4100, 5100]
    assert [table["ideal_snapshot_peak_reduction_mib"] for table in result["tables"]] == [100, 480]
    assert len(result["archive_sha256"]) == 64
    assert list(tmp_path.iterdir()) == [path]
