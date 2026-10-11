# SPDX-License-Identifier: Apache-2.0
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.sidecard.reference import ExpertReply
from tools.sidecard.service_adapter import SidecardMoEMethod
from tools.sidecard.tensor_wire import TensorPacket


@pytest.mark.parametrize("selected_token", [None, 0, 1])
def test_native_dispatch_runs_once_on_caller_while_host_cuda_rpc_is_pending(monkeypatch, selected_token):
    monkeypatch.setitem(
        sys.modules,
        "vllm.forward_context",
        SimpleNamespace(get_forward_context=lambda: SimpleNamespace(attn_metadata={})),
    )
    caller = threading.get_ident()
    gpu_started, native_started = threading.Event(), threading.Event()
    hidden = torch.zeros((2, 4096), dtype=torch.float16)
    ids = torch.zeros((2, 8), dtype=torch.int64)
    weights = torch.zeros((2, 8), dtype=torch.float32)
    if selected_token is not None:
        ids[selected_token, 7] = 143
        weights[selected_token, 7] = 0.25
    calls = []

    def exchange(packet):
        assert threading.get_ident() != caller
        assert type(packet.tensor) is bytes
        gpu_started.set()
        assert native_started.wait(5), "CUDA reply cannot complete until native dispatch starts"
        c = packet.control
        body = np.full((len(c.token_ids), 4096), 1.5, dtype="<f4").tobytes()
        return TensorPacket(ExpertReply(c.execution_id, c.layer_id, c.owner, c.routes, ()), body, partial_reduce=True)

    def native():
        assert threading.get_ident() == caller
        if selected_token is not None:
            assert gpu_started.wait(5)
        calls.append("native")
        native_started.set()
        return torch.full_like(hidden, 2, dtype=torch.float32)

    with ThreadPoolExecutor(max_workers=1) as executor:
        session = SimpleNamespace(
            one_card=True,
            cuda_experts=144,
            tensor_transport=True,
            partial_reduce=True,
            reply_fp16=False,
            packet_tokens=1,
            layers={},
            executor=executor,
            peer=SimpleNamespace(exchange=exchange),
            flush=lambda: None,
            close=lambda: None,
        )
        method = SidecardMoEMethod(session, 3, None)
        layer = SimpleNamespace(w2_experts=SimpleNamespace(local_expert_offset=144, num_local_experts=72))
        actual = method.apply(layer, hidden, weights, ids, None, None, native_dispatch=native)
    expected = torch.full_like(hidden, 2, dtype=torch.float32)
    if selected_token is not None:
        expected[selected_token] += 1.5
    assert torch.equal(actual, expected)
    assert calls == ["native"]
    assert method.state["native_overlap_calls"] == int(selected_token is not None)


def test_native_failure_closes_session_and_never_returns_cuda_fallback(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "vllm.forward_context",
        SimpleNamespace(get_forward_context=lambda: SimpleNamespace(attn_metadata={})),
    )
    released = threading.Event()
    closed = []

    def exchange(packet):
        assert released.wait(5)
        c = packet.control
        return TensorPacket(
            ExpertReply(c.execution_id, c.layer_id, c.owner, c.routes, ()),
            np.zeros((1, 4096), dtype="<f4").tobytes(),
            partial_reduce=True,
        )

    def native():
        raise RuntimeError("native expert failure")

    def close():
        closed.append(True)
        released.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        session = SimpleNamespace(
            one_card=True,
            cuda_experts=144,
            tensor_transport=True,
            partial_reduce=True,
            reply_fp16=False,
            packet_tokens=512,
            layers={},
            executor=executor,
            peer=SimpleNamespace(exchange=exchange),
            flush=lambda: None,
            close=close,
        )
        method = SidecardMoEMethod(session, 3, None)
        layer = SimpleNamespace(w2_experts=SimpleNamespace(local_expert_offset=144, num_local_experts=72))
        ids = torch.zeros((1, 8), dtype=torch.int64)
        weights = torch.zeros((1, 8), dtype=torch.float32)
        weights[0, 7] = 1
        with pytest.raises(RuntimeError, match="native expert failure"):
            method.apply(
                layer, torch.zeros((1, 4096), dtype=torch.float16), weights, ids, None, None, native_dispatch=native
            )
    assert closed == [True]
