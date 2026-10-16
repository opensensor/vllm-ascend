# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from tools.qwen4exp.tp6_mixed_profile import collective_metadata, ple_metadata, schedule_metadata


def test_schedule_records_only_counts():
    schedule = SimpleNamespace(
        num_scheduled_tokens={"private-request": 3, "another": 128},
        scheduled_spec_decode_tokens={"private-request": [999, 1000]},
    )
    assert schedule_metadata(schedule) == dict(scheduled=[3, 128], drafts=[2])


def test_ple_records_input_geometry_without_tensor_reads():
    assert ple_metadata(None, object(), SimpleNamespace(shape=(131,))) == dict(tokens=131)


def test_collective_reports_contiguous_admission():
    collective = SimpleNamespace(owners={("float16", 335360): object()})
    tensor = SimpleNamespace(dtype="torch.float16", numel=lambda: 335360, is_contiguous=lambda: True)
    assert collective_metadata(collective, tensor, None) == dict(elements=335360, dtype="float16", native=True)
    tensor.is_contiguous = lambda: False
    assert not collective_metadata(collective, tensor, None)["native"]
    tensor.numel = lambda: 1
    tensor.is_contiguous = lambda: True
    assert not collective_metadata(collective, tensor, None)["native"]
