# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import socket
import struct
from pathlib import Path

import pytest

from tools.sidecard.pair_collective_probe import FRAME_BYTES, exchange, pattern, receive, send, summarize, validate


def config(tmp_path):
    libraries = {}
    for name in ("acl", "native"):
        path = tmp_path / (name + ".so")
        path.write_bytes(name.encode())
        libraries[name + "_library"] = str(path)
        libraries[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(libraries, physical_chips=[2, 3], rows=1)


def test_configuration_is_pinned_and_defaults_bounded(tmp_path):
    value = validate(config(tmp_path))
    assert value["batch"] == 30 and value["pairs"] == 5
    assert value["warmup"] == 10 and value["physical_chips"] == [2, 3]


@pytest.mark.parametrize(
    "field,value",
    (
        ("rows", True),
        ("rows", 33),
        ("batch", 65),
        ("pairs", 0),
        ("physical_chips", [0, 4]),
        ("physical_chips", [2, 2]),
        ("physical_chips", [False, 1]),
        ("unknown", 1),
        ("physical_chips", [[0], 1]),
        ("handshake", "unsafe"),
    ),
)
def test_invalid_configuration_rejected_before_launch(tmp_path, field, value):
    data = config(tmp_path)
    data[field] = value
    with pytest.raises(ValueError):
        validate(data)


def test_changed_binary_rejected_before_launch(tmp_path):
    data = config(tmp_path)
    (tmp_path / "native.so").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        validate(data)


def test_ping_pong_configuration_is_explicit_and_local_reducer_cannot_be_combined(tmp_path):
    data = config(tmp_path)
    assert validate(data)["handshake"] == "consumed"
    data["handshake"] = "ping-pong"
    assert validate(data)["handshake"] == "ping-pong"
    data["local_add"] = {}
    with pytest.raises(ValueError, match="requires ACLNN"):
        validate(data)


def local_add_config(tmp_path, *, rows=1, blocks=1):
    data = config(tmp_path)
    selected = {"blocks": blocks}
    for name in ("library", "binary"):
        path = tmp_path / ("local-add-" + name)
        path.write_bytes(("pinned local add " + name).encode())
        selected[name], selected[name + "_sha256"] = str(path), hashlib.sha256(path.read_bytes()).hexdigest()
    data.update(rows=rows, local_add=selected)
    return data


@pytest.mark.parametrize("rows", (1, 4, 32))
@pytest.mark.parametrize("blocks", (1, 2, 4, 8))
def test_probe_authenticates_optional_local_add_for_fixed_shapes(tmp_path, rows, blocks):
    data = local_add_config(tmp_path, rows=rows, blocks=blocks)
    result = validate(data)
    assert result["rows"] == rows and result["local_add"] == data["local_add"]
    assert result["exchange_mode"] == "push"


@pytest.mark.parametrize("raw", (None, [], "local", {}, {"extra": 1}))
def test_probe_rejects_malformed_local_add_object(tmp_path, raw):
    data = config(tmp_path)
    data["local_add"] = raw
    with pytest.raises(ValueError, match="local_add"):
        validate(data)


@pytest.mark.parametrize("rows", (2, 3, 31))
def test_probe_rejects_local_add_unsupported_shape_before_launch(tmp_path, rows):
    with pytest.raises(ValueError, match="fixed-row"):
        validate(local_add_config(tmp_path, rows=rows))


@pytest.mark.parametrize("blocks", (True, 0, 3, 1.0, "1"))
def test_probe_rejects_local_add_grid_coercions(tmp_path, blocks):
    with pytest.raises(ValueError, match="grid"):
        validate(local_add_config(tmp_path, blocks=blocks))


@pytest.mark.parametrize("name", ("library", "binary"))
def test_probe_rejects_local_add_payload_drift(tmp_path, name):
    data = local_add_config(tmp_path)
    Path(data["local_add"][name]).write_bytes(b"changed after config creation")
    with pytest.raises(ValueError, match="SHA256"):
        validate(data)


def test_remote_read_disabled_before_optional_kernel_authentication_or_launch(tmp_path):
    data = config(tmp_path)
    data.update(exchange_mode="remote-read", local_add={"library": "missing kernel"})
    with pytest.raises(ValueError, match="remote-read is disabled"):
        validate(data)


def test_private_frames_roundtrip_and_disconnect():
    a, b = socket.socketpair()
    try:
        send(a, {"phase": "exported", "payload": {"key": "private"}})
        assert receive(b) == {"phase": "exported", "payload": {"key": "private"}}
        a.close()
        with pytest.raises(EOFError):
            receive(b)
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("length", (0, FRAME_BYTES + 1))
def test_invalid_control_length(length):
    a, b = socket.socketpair()
    try:
        a.sendall(struct.pack("!I", length))
        with pytest.raises(ValueError, match="frame size"):
            receive(b)
    finally:
        a.close()
        b.close()


def test_control_budget_and_json_object():
    a, b = socket.socketpair()
    try:
        with pytest.raises(ValueError, match="too large"):
            send(a, {"large": "x" * FRAME_BYTES})
        body = json.dumps([1, 2]).encode()
        a.sendall(struct.pack("!I", len(body)) + body)
        with pytest.raises(ValueError, match="object"):
            receive(b)
    finally:
        a.close()
        b.close()


def test_mismatched_phase_fails():
    a, b = socket.socketpair()
    try:
        send(b, {"phase": "stale", "payload": {}})
        with pytest.raises(RuntimeError, match="phase differs"):
            exchange(a, "expected", {})
    finally:
        a.close()
        b.close()


def test_patterns_change_rank_and_iteration_and_cover_full_payload():
    assert len(pattern(4096, 0, 0)) == 16384
    assert pattern(4096, 0, 0) != pattern(4096, 1, 0)
    assert pattern(4096, 0, 0) != pattern(4096, 0, 1)


def timings():
    return [
        dict(
            rank=rank,
            timings=[
                dict(
                    pair=0,
                    mode="custom",
                    count=30,
                    started_ns=1000 + rank * 300,
                    submitted_ns=2000 + rank * 100,
                    completed_ns=4000 + rank * 100,
                )
            ],
        )
        for rank in range(2)
    ]


def test_summary_matches_both_host_clocks_without_claiming_transit():
    result = summarize(timings())
    assert result["samples"][0]["arrival_spread_us"] == 0.3
    assert result["samples"][0]["completed_batch_us_per_call"] == pytest.approx(3100 / 30 / 1000)
    assert result["samples"][0]["completion_after_latest_start_us_per_call"] == pytest.approx(2800 / 30 / 1000)
    assert not result["pure_transit_measured"] and not result["confidence_interval_computed"]


@pytest.mark.parametrize("change", ("missing-rank", "missing-sample", "duplicate", "bad-clock", "bad-count"))
def test_summary_rejects_unmatched_or_invalid_evidence(change):
    values = timings()
    if change == "missing-rank":
        values.pop()
    elif change == "missing-sample":
        values[1]["timings"] = []
    elif change == "duplicate":
        values[0]["timings"] *= 2
    elif change == "bad-clock":
        values[0]["timings"][0]["completed_ns"] = 0
    else:
        values[1]["timings"][0]["count"] = 31
    with pytest.raises(ValueError):
        summarize(values)
