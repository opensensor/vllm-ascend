# SPDX-License-Identifier: Apache-2.0
"""Admission tests for the serving boundary and shared-table lifetime."""

import fcntl
import os
from types import SimpleNamespace

import pytest

from tools.qwen4exp.ple_serving import load_library, open_table, request_count
from tools.qwen4exp.ple_sources import IMMUTABLE_SEALS


@pytest.mark.parametrize("tokens", [1, 2, 3, 6, 9, 12])
def test_decode_layout_including_empty_padding_groups(tokens):
    history = SimpleNamespace(shape=(5, 2), ndim=2)
    starts = SimpleNamespace(numel=lambda: 6)
    assert request_count(tokens, starts, history, [0, tokens, tokens, tokens, tokens, tokens]) == 4


@pytest.mark.parametrize("tokens,end", [(13, 13), (2048, 2048), (0, 0), (6, 3)])
def test_prefill_and_fifth_padding_group_fall_back(tokens, end):
    history = SimpleNamespace(shape=(5, 2), ndim=2)
    starts = SimpleNamespace(numel=lambda: 6)
    assert request_count(tokens, starts, history, [0, end, end, end, end, tokens]) is None


def test_missing_host_layout_falls_back():
    assert request_count(3, None, None, None) is None


def test_shared_mapping_survives_owner_descriptor_close():
    original = os.memfd_create("ple-test", os.MFD_ALLOW_SEALING)
    try:
        os.write(original, b"x" * 320)
        fcntl.fcntl(original, fcntl.F_ADD_SEALS, IMMUTABLE_SEALS)
        info = os.fstat(original)
        metadata = dict(fd_path=f"/proc/self/fd/{original}", device=info.st_dev, inode=info.st_ino, bytes=320)
        owned = open_table(metadata)
        with pytest.raises(ValueError, match="identity changed"):
            open_table(dict(metadata, inode=info.st_ino + 1))
    finally:
        os.close(original)
    try:
        assert os.pread(owned, 320, 0) == b"x" * 320
        assert fcntl.fcntl(owned, fcntl.F_GET_SEALS) & IMMUTABLE_SEALS == IMMUTABLE_SEALS
    finally:
        os.close(owned)


def test_reject_library_change_before_loading(tmp_path):
    library = tmp_path / "library.so"
    library.write_bytes(b"not the admitted library")
    with pytest.raises(ValueError, match="digest differs"):
        load_library(library, "0" * 64)
