# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the 310P hardware/topology probe (plan T0.3).

The probe is exercised entirely with mocked collectors so no NPU is required.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[3]
_MODULE_PATH = _REPO_ROOT / "tools" / "qwen38_1m" / "hw_probe.py"


def _load_hw_probe():
    spec = importlib.util.spec_from_file_location("qwen38_hw_probe", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass annotation resolution can find the module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hw_probe = _load_hw_probe()

_GIB = 1024**3


def _four_chip_inventory():
    # Two 300I Duo cards (card 0 and card 1), two chips each.
    return [
        hw_probe.ChipInfo(
            chip_id=0,
            card_id=0,
            name="Ascend310P",
            firmware_version="1.0",
            driver_version="24.1",
            total_bytes=48 * _GIB,
            free_bytes=46 * _GIB,
            numa_node=0,
        ),
        hw_probe.ChipInfo(
            chip_id=1,
            card_id=0,
            name="Ascend310P",
            firmware_version="1.0",
            driver_version="24.1",
            total_bytes=48 * _GIB,
            free_bytes=45 * _GIB,
            numa_node=0,
        ),
        hw_probe.ChipInfo(
            chip_id=2,
            card_id=1,
            name="Ascend310P",
            firmware_version="1.0",
            driver_version="24.1",
            total_bytes=48 * _GIB,
            free_bytes=46 * _GIB,
            numa_node=1,
        ),
        hw_probe.ChipInfo(
            chip_id=3,
            card_id=1,
            name="Ascend310P",
            firmware_version="1.0",
            driver_version="24.1",
            total_bytes=48 * _GIB,
            free_bytes=46 * _GIB,
            numa_node=1,
        ),
    ]


def test_probe_records_all_four_chips():
    report = hw_probe.probe(chip_collector=_four_chip_inventory, now=lambda: 123.0)
    assert report.schema_version == hw_probe.SCHEMA_VERSION
    assert report.timestamp == 123.0
    assert len(report.chips) == 4
    assert report.min_free_bytes() == 45 * _GIB


def test_link_classification_within_and_cross_card():
    report = hw_probe.probe(chip_collector=_four_chip_inventory)
    # 4 chips -> 12 directed links. Each chip has 1 within-card peer, 2 cross-card.
    assert len(report.links) == 12
    within = [link for link in report.links if link.link_class == "within_card"]
    cross = [link for link in report.links if link.link_class == "cross_card"]
    assert len(within) == 4  # (0<->1) and (2<->3), both directions
    assert len(cross) == 8
    # Chips 0 and 1 share card 0 -> within-card.
    assert any(link.src_chip == 0 and link.dst_chip == 1 and link.link_class == "within_card" for link in report.links)
    # Chips 0 and 2 are on different cards -> cross-card.
    assert any(link.src_chip == 0 and link.dst_chip == 2 and link.link_class == "cross_card" for link in report.links)


def test_optional_collectors_default_empty_but_schema_valid():
    report = hw_probe.probe(chip_collector=_four_chip_inventory)
    assert report.pcie == []
    assert report.bandwidth == []
    assert report.ddr_channels_populated is None
    # Still round-trips.
    assert hw_probe.ProbeReport.from_dict(report.to_dict()).chips[0].chip_id == 0


def test_full_probe_with_all_collectors():
    def pcie_collector():
        return [hw_probe.PCIeInfo(chip_id=i, generation=4, width=16) for i in range(4)]

    def ddr_collector():
        return (8, 8)

    def bandwidth_collector(chips):
        return [hw_probe.BandwidthSample(chip_id=c.chip_id, h2d_gib_s=24.0, d2h_gib_s=22.0) for c in chips]

    report = hw_probe.probe(
        chip_collector=_four_chip_inventory,
        pcie_collector=pcie_collector,
        ddr_collector=ddr_collector,
        bandwidth_collector=bandwidth_collector,
    )
    assert len(report.pcie) == 4
    assert report.ddr_channels_populated == 8
    assert len(report.bandwidth) == 4
    assert report.bandwidth[0].h2d_gib_s == 24.0


def test_json_schema_round_trip():
    def pcie_collector():
        return [hw_probe.PCIeInfo(chip_id=i, generation=4, width=16) for i in range(4)]

    original = hw_probe.probe(
        chip_collector=_four_chip_inventory,
        pcie_collector=pcie_collector,
        ddr_collector=lambda: (4, 8),
        bandwidth_collector=lambda chips: [
            hw_probe.BandwidthSample(chip_id=c.chip_id, h2d_gib_s=20.0, d2h_gib_s=18.0) for c in chips
        ],
        now=lambda: 7.0,
    )
    import json

    restored = hw_probe.ProbeReport.from_dict(json.loads(original.to_json()))
    assert restored.to_dict() == original.to_dict()
    assert restored.timestamp == 7.0
    assert restored.ddr_channels_populated == 4
    assert restored.links[0].link_class in {"within_card", "cross_card"}


def test_human_summary_reports_free_and_links():
    report = hw_probe.probe(
        chip_collector=_four_chip_inventory,
        bandwidth_collector=lambda chips: [
            hw_probe.BandwidthSample(chip_id=c.chip_id, h2d_gib_s=24.0, d2h_gib_s=22.0) for c in chips
        ],
    )
    summary = report.human_summary()
    assert "min free per chip: 45.00 GiB" in summary
    assert "within-card" in summary and "cross-card" in summary
    assert "H2D 24.0 GiB/s" in summary


def test_used_bytes_property():
    chip = _four_chip_inventory()[1]
    assert chip.used_bytes == 3 * _GIB


def test_default_chip_collector_raises_until_finalized():
    with pytest.raises(NotImplementedError, match="pinned CANN"):
        hw_probe._default_chip_collector()
