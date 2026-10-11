# SPDX-License-Identifier: Apache-2.0
"""Resident banks survive orderly reconnects and close on malformed packets."""

import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from unittest.mock import Mock

import pytest

from tools.sidecard.placement import ExpertPlacement
from tools.sidecard.readiness import receive_ready
from tools.sidecard.reference import ExpertRequest, Route
from tools.sidecard.resident_daemon import run
from tools.sidecard.runtime import SocketPeer
from tools.sidecard.tensor_wire import TensorSocketPeer, decode_tensor, encode_tensor
from tools.sidecard.wire import PREFIX, PacketLimits, WireIdentity, decode_packet, encode_packet
from tools.sidecard.worker import DenseExpert, DenseExpertBank, ExpertWorker


@pytest.mark.parametrize("malformed", [False, True])
@pytest.mark.parametrize("tensor_transport", [False, True])
def test_sequential_sessions_prepare_once_and_fail_closed(tmp_path, monkeypatch, malformed, tensor_transport):
    placement = ExpertPlacement(("cuda:0",), (("cuda:0",),))
    identity = WireIdentity.for_placement(placement, "a" * 64)
    limits = PacketLimits(1, 1, 1, 1, 1)
    expert = DenseExpert(((1.0,),), ((1.0,),), ((1.0,),))
    bank = DenseExpertBank(((3, 0, expert),))
    backend = Mock()
    backend.prepare.return_value = bank.prepare()
    backend.compute.side_effect = bank.compute
    backend.cuda.allocated_bytes = 123
    backend.keys = ((3, 0),)
    backend.plan.by_layer = {3: (0,)}
    backend.metrics = {}
    worker = ExpertWorker("cuda:0", placement, limits, backend)

    def compute_tensor(packet):
        request = decode_packet(encode_tensor(packet, identity, limits), identity, limits)
        reply = worker.execute(request)
        return decode_tensor(encode_packet(reply, identity, limits), identity, limits)

    backend.compute_tensor.side_effect = compute_tensor
    monkeypatch.setattr("tools.sidecard.resident_daemon.worker_from_config", lambda value: (worker, identity))
    config, address, ready = (tmp_path / name for name in ("config.json", "worker.sock", "ready.json"))
    config.write_text(json.dumps({"backend": {"kind": "cuda_resident"}}))
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run, config, address, ready, max_sessions=2, tensor_transport=tensor_transport)
        deadline = time.monotonic() + 5
        while not ready.exists():
            if future.done():
                future.result()
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert json.loads(ready.read_text())["max_sessions"] == 2
        for session in range(1 if malformed else 2):
            connection = socket.socket(socket.AF_UNIX)
            connection.connect(str(address))
            receive_ready(connection, "cuda:0", identity, time.monotonic() + 5)
            if malformed:
                connection.sendall(PREFIX.pack(b"BADMAGIC", 1, 0) + b"x")
                connection.settimeout(5)
                # Rejecting an unread malformed payload may reset the stream.
                with suppress(ConnectionResetError):
                    assert connection.recv(1) == b""
                connection.close()
            else:
                peer = (TensorSocketPeer if tensor_transport else SocketPeer)(connection, identity, limits, 5)
                request = ExpertRequest(str(session), 3, "cuda:0", (Route(0, 0, 0),), (0,), ((0.5,),))
                if tensor_transport:
                    packet = decode_tensor(encode_packet(request, identity, limits), identity, limits)
                    reply = peer.exchange(packet)
                    assert decode_packet(
                        encode_tensor(reply, identity, limits), identity, limits
                    ).outputs == bank.compute(request)
                else:
                    assert peer.exchange(request).outputs == bank.compute(request)
                peer.close()
        future.result(timeout=5)
    backend.prepare.assert_called_once()
    backend.close.assert_called_once()
    assert not address.exists()


@pytest.mark.parametrize("sessions", [0, -1, True, 1.5])
def test_session_bounds_reject_before_loading(tmp_path, sessions):
    with pytest.raises(ValueError, match="positive integer"):
        run(tmp_path / "missing", tmp_path / "socket", tmp_path / "ready", max_sessions=sessions)
