# SPDX-License-Identifier: Apache-2.0
"""Offline frozen-source and DMA layout checks, not native numerical gates."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.qwen4exp import build_qsa_gather_staging as builder


def test_frozen_headers_differ_only_in_staging_width(tmp_path):
    root = Path(__file__).resolve().parents[3]
    hashes = builder.freeze_headers(root, tmp_path)
    original = (tmp_path / "qsa_gather_value_nz_v310.h").read_text()
    assert (tmp_path / "qsa_gather_staging_4.h").read_text() == original
    eight = (tmp_path / "qsa_gather_staging_8.h").read_text()
    assert eight.replace("TOKEN_BLOCKS_PER_TILE = 8;", "TOKEN_BLOCKS_PER_TILE = 4;") == original
    assert all(builder.digest(tmp_path / name) == value for name, value in hashes.items())


def test_header_drift_rejected_before_copy(tmp_path):
    headers = tmp_path / "csrc/attention/qsa_gather_value_nz_v310/op_kernel"
    headers.mkdir(parents=True)
    (headers / "qsa_gather_value_nz_v310.h").write_text("changed header")
    destination = tmp_path / "frozen"
    destination.mkdir()
    with pytest.raises(ValueError, match="qualified QSA"):
        builder.freeze_headers(tmp_path, destination)
    assert not list(destination.iterdir())


def test_build_requires_disabled_backend_autoload(monkeypatch, tmp_path):
    monkeypatch.delenv("TORCH_DEVICE_BACKEND_AUTOLOAD", raising=False)
    with pytest.raises(RuntimeError, match="AUTOLOAD=0"):
        builder.build(tmp_path, tmp_path, tmp_path, tmp_path)


def test_build_two_frozen_entries_and_owned_queue_drained_bridge(monkeypatch, tmp_path):
    import torch.utils.cpp_extension

    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    root = Path(__file__).resolve().parents[3]
    actual_find = builder.importlib.util.find_spec
    monkeypatch.setattr(
        builder.importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(origin=str(tmp_path / "torch_npu/__init__.py"))
            if name == "torch_npu"
            else actual_find(name)
        ),
    )
    monkeypatch.setattr(torch.utils.cpp_extension, "include_paths", lambda: [])
    monkeypatch.setattr(torch.utils.cpp_extension, "library_paths", lambda: [])
    calls = []

    def compile_mock(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("-o") + 1]) if command[0] == "c++" else Path(command[2])
        output.write_bytes(b"mock artifact")

    monkeypatch.setattr(builder.subprocess, "run", compile_mock)
    destination = tmp_path / "fresh"
    result = builder.build(destination, tmp_path, root, root)
    assert result["namespace"] == "qwen_qsa_gather_staging_v1"
    assert result["native_task_type"] == "KERNEL_TYPE_AICORE"
    assert result["eager_only"] is result["flush_host_queue"] is True
    assert result["qualified_headers"] == builder.QUALIFIED_HEADERS
    assert "-DQSA_STAGING_BLOCKS=4" in calls[1]
    assert "-DQSA_STAGING_BLOCKS=8" in calls[2]
    assert "-DQWEN_NATIVE_FLUSH_HOST_QUEUE=1" in calls[3]
    assert "-DGLM_RECONSTRUCTION_NAMESPACE=qwen_qsa_gather_staging_v1" in calls[3]
    for width in (4, 8):
        binary = result["binaries"][f"native_qsa_gather_staging_{width}"]
        assert binary["entry"] == f"qwen_qsa_gather_staging_{width}_v1"
        assert binary["sha256"] == builder.digest(binary["path"])
    assert json.loads((destination / "provenance.json").read_text()) == result
    with pytest.raises(FileExistsError):
        builder.build(destination, tmp_path, root, root)
    assert len(calls) == 4


def gathered_rows(count, tail_count, heads):
    # Distinct integer payloads detect physical pages, heads, dimensions and lanes.
    cache = np.arange(5 * heads * 16 * 64 * 16).reshape(5, heads, 16, 64, 16)
    table = np.array([4, 2, 0, 3])
    groups = np.random.default_rng(174).integers(0, 64, size=512)
    tokens = []
    for rank in range(516):
        if rank < count:
            tokens.extend(range(int(groups[rank]) * 4, int(groups[rank]) * 4 + 4))
        elif rank == 512 and tail_count:
            # Tail crosses a logical page boundary; invalid tail lanes use token0.
            tokens.extend(63 + lane if lane < tail_count else 0 for lane in range(4))
        else:
            tokens.extend(range(4))
    return np.stack([cache[table[token // 64], :, :, token % 64, :] for token in tokens], axis=1)


def staged_raw_output(rows, staging, transpose):
    heads, _, dimensions, _ = rows.shape
    result = np.full(heads * dimensions * 129 * 256, -1, dtype=np.int64)
    for head in range(heads):
        for block in range(0, 129, staging):
            count = min(staging, 129 - block)
            tile = np.full((dimensions, staging, 16, 16), -2, dtype=np.int64)
            tile[:, :count] = (
                rows[head, block * 16 : (block + count) * 16].transpose(1, 0, 2).reshape(dimensions, count, 16, 16)
            )
            if transpose:
                for token_block in range(count):
                    offset = ((head * 129 + block + token_block) * dimensions) * 256
                    result[offset : offset + dimensions * 256] = tile[:, token_block].transpose(0, 2, 1).reshape(-1)
            else:
                # Exact outputCopy: head-dimension blocks are separated by
                # (129-count)*256 elements; UB stride is (staging-count)*256.
                for dimension in range(dimensions):
                    offset = ((head * dimensions + dimension) * 129 + block) * 256
                    result[offset : offset + count * 256] = tile[dimension, :count].reshape(-1)
    return result


@pytest.mark.parametrize("count", [0, 1, 15, 16, 17, 255, 256, 511, 512])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("heads", [1, 2])
def test_staging_four_eight_and_direct_nz_layout_agree(count, transpose, heads):
    rows = gathered_rows(count, count % 4, heads)
    logical = rows.reshape(heads, 129, 16, 16, 16)  # H, token block, token lane, D block, D lane
    expected = (logical.transpose(0, 1, 3, 4, 2) if transpose else logical.transpose(0, 3, 1, 2, 4)).reshape(-1)
    for staging in (4, 8):
        np.testing.assert_array_equal(staged_raw_output(rows, staging, transpose), expected)


def test_native_descriptor_and_launch_guard_match_pinned_abi():
    root = Path(__file__).resolve().parents[3]
    source = (root / "tools/qwen4exp/native_qsa_gather_staging.cpp").read_text()
    assert all(f"raw[{i}]" in source for i in range(11))
    assert "td.taskCount != td.numTokens * td.numKvHeads" in source
    assert "td.tasksPerCore != (td.taskCount + blocks - 1) / blocks" in source
    assert "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);" in source
    assert "KERNEL_TYPE_AIV_ONLY" not in source
