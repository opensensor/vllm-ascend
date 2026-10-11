# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools.sidecard.ordered_tensor import combine_compact
from tools.sidecard.reference import ExpertReply
from tools.sidecard.service_adapter import SidecardMoEMethod
from tools.sidecard.tensor_wire import TensorPacket


@pytest.mark.parametrize("partial,half", [(False, False), (False, True), (True, False)])
def test_model_tensor_boundary_preserves_weighted_outputs_and_zero_tokens(monkeypatch, partial, half):
    import torch

    monkeypatch.setitem(
        sys.modules,
        "vllm.forward_context",
        SimpleNamespace(get_forward_context=lambda: SimpleNamespace(attn_metadata={})),
    )
    hidden = np.random.default_rng(53).normal(size=(4, 4096)).astype("<f2")
    ids = np.zeros((4, 8), dtype=np.int64)
    ids[0, 2] = 3
    ids[0, 6] = 7
    ids[3, 7] = 9
    weights = np.zeros((4, 8), dtype=np.float32)
    weights[0, 2] = 0.25
    weights[0, 6] = 0.5
    weights[3, 7] = 0.125
    expected = np.zeros((4, 4096), dtype=np.float32)
    expected[0] = hidden[0].astype(np.float32) * 4 * 0.25 + hidden[0].astype(np.float32) * 8 * 0.5
    expected[3] = ((hidden[3].astype(np.float32) * 10).astype("<f2").astype(np.float32)) * 0.125
    calls = []

    def exchange(packet):
        c = packet.control
        calls.append(packet)
        rows = np.frombuffer(packet.tensor, dtype="<f2").reshape(len(c.token_ids), 4096)
        mapping = {token: index for index, token in enumerate(c.token_ids)}
        raw = np.array([(rows[mapping[r.token]].astype(np.float32) * (r.expert + 1)).astype("<f2") for r in c.routes])
        control = ExpertReply(c.execution_id, c.layer_id, c.owner, c.routes, ())
        if partial:
            dense = np.zeros((4, 8), dtype=np.float32)
            w = np.frombuffer(packet.route_weights, dtype="<f4")
            for r, weight in zip(c.routes, w, strict=True):
                dense[r.token, r.slot] = weight
            result = combine_compact(c.routes, raw.astype("<f4").tobytes(), dense, 4096)
            return TensorPacket(control, result[list(c.token_ids)].tobytes(), partial_reduce=True)
        return TensorPacket(control, raw.tobytes() if half else raw.astype("<f4").tobytes(), reply_fp16=half)

    session = SimpleNamespace(
        layers={},
        tensor_transport=True,
        packet_tokens=640,
        partial_reduce=partial,
        reply_fp16=half,
        peer=SimpleNamespace(exchange=exchange),
        flush=lambda: None,
    )
    method = SidecardMoEMethod(session, 3, None)
    method.state["shadow_checks"] = 3
    layer = SimpleNamespace(w2_experts=SimpleNamespace(local_expert_offset=0, num_local_experts=72))
    actual = method.apply(layer, torch.from_numpy(hidden), torch.from_numpy(weights), torch.from_numpy(ids), None, None)
    assert actual.numpy().tobytes() == expected.tobytes()
    assert len(calls) == 1 and calls[0].control.token_ids == (0, 3)
    assert method.state["real_cuda_routes"] == 3
