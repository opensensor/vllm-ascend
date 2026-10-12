# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.sidecard.cold_dispatch import HEADER, INPUT_ROW_BYTES, ColdDispatch
from tools.sidecard.cold_placement import cuda_keys, load, lookup, validate


def hot_placement():
    return {str(layer): [1, 31, 76, 125, 144, 210, 223, 284] for layer in range(18, 33)}


class Peer:
    def __init__(self, rank, events):
        self.rank, self.events = rank, events
        self.header = None
        self.body = bytearray()
        self.corrupt = False

    def sendall(self, body):
        if self.header is None:
            self.header = bytes(body)
            self.events.append(("send", self.rank))
            return
        sequence, layer, rows = HEADER.unpack(self.header)
        assert len(body) == rows * INPUT_ROW_BYTES
        raw = bytes(body)
        x = np.frombuffer(raw, dtype=np.float16, count=rows * 4096).reshape(rows, 4096).astype(np.float32)
        ids = np.frombuffer(raw, dtype=np.int32, count=rows * 8, offset=rows * 8192).reshape(rows, 8)
        weights = np.frombuffer(raw, dtype=np.float32, count=rows * 8, offset=rows * (8192 + 32)).reshape(rows, 8)
        scale = np.where((ids >= 144 * self.rank) & (ids < 144 * (self.rank + 1)), weights * (ids + 1), 0).sum(
            -1, dtype=np.float32
        )
        response = HEADER.pack(sequence + int(self.corrupt), layer, rows)
        self.body.extend(response + (x * scale[:, None]).astype(np.float32).tobytes())
        self.header = None

    def recv_into(self, target):
        self.events.append(("receive", self.rank))
        count = min(len(target), len(self.body), 1001)
        target[:count] = self.body[:count]
        del self.body[:count]
        return count


def dispatcher():
    value = object.__new__(ColdDispatch)
    events = []
    hot = hot_placement()
    value.lookups = {int(index): torch.tensor(lookup(ids), dtype=torch.int32) for index, ids in hot.items()}
    value.torch = SimpleNamespace(
        **{name: getattr(torch, name) for name in ("where", "float16", "float32", "int32")},
        cuda=SimpleNamespace(synchronize=lambda: None),
    )
    value.host = torch.empty(32 * INPUT_ROW_BYTES, dtype=torch.uint8)
    value.host_bytes = memoryview(value.host.numpy())
    value.replies = [torch.empty(32 * 4096 * 4, dtype=torch.uint8) for _ in range(2)]
    value.reply_bytes = [memoryview(v.numpy()) for v in value.replies]
    value.device_replies = [torch.full((32, 4096), float("nan")) for _ in range(2)]
    value.connections = [Peer(rank, events) for rank in range(2)]
    value.sequences = [0, 0]

    def cuda(layer, x, ids, weights):
        events.append(("cuda_hot", None))
        global_ids = torch.tensor(hot[str(layer)])[ids.long()]
        return x.float() * ((global_ids.float() + 1) * weights).sum(-1, keepdim=True)

    value.cuda = cuda
    return value, events


@pytest.mark.parametrize("rows", [1, 4, 17, 32])
@pytest.mark.parametrize(
    "selected",
    [
        [1, 31, 76, 125, 144, 210, 223, 284],
        [2, 3, 4, 5, 6, 7, 8, 9],
        [145, 146, 147, 148, 149, 150, 151, 152],
        [1, 2, 3, 4, 144, 145, 146, 210],
    ],
)
def test_exact_owner_partition_and_no_stale_reply(rows, selected):
    actor, events = dispatcher()
    hidden = torch.full((rows, 4096), 0.5, dtype=torch.float16)
    ids = torch.tensor(selected).expand(rows, -1)
    weights = torch.full((rows, 8), 2.5 / 8)
    actual = actor(18, hidden, ids, weights)
    expected = hidden.float() * ((ids.float() + 1) * weights).sum(-1, keepdim=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    assert sum(a == "send" for a, b in events) == sum(actor.sequences)


def test_shared_runs_after_both_dispatches_before_reply_wait():
    actor, events = dispatcher()
    hidden = torch.ones(1, 4096).half()
    ids = torch.tensor([[1, 2, 3, 4, 144, 145, 146, 210]])

    def shared():
        events.append(("shared", None))
        return hidden * 2

    routed, result = actor.compute(18, hidden, ids, torch.full((1, 8), 2.5 / 8), shared)
    assert torch.equal(result, hidden * 2)
    positions = {
        kind: next(i for i, e in enumerate(events) if e[0] == kind) for kind in ("cuda_hot", "shared", "receive")
    }
    assert events[:2] == [("send", 0), ("send", 1)]
    assert positions["cuda_hot"] < positions["shared"] < positions["receive"]
    assert torch.isfinite(routed).all()


def test_wrong_invocation_response_is_rejected():
    actor, _ = dispatcher()
    actor.connections[0].corrupt = True
    with pytest.raises(ValueError, match="another invocation"):
        actor(18, torch.ones(1, 4096).half(), torch.arange(2, 10)[None], torch.full((1, 8), 2.5 / 8))


def test_placement_authentication_and_arbitrary_offsets(tmp_path):
    hot = hot_placement()
    validate(hot)
    record = {"checkpoint_sha256": "a" * 64, "gpu_hot_experts": hot}
    path = tmp_path / "placement.json"
    raw = json.dumps(record).encode()
    path.write_bytes(raw)
    spec = {
        "checkpoint_sha256": "a" * 64,
        "expert_placement": str(path),
        "expert_placement_sha256": hashlib.sha256(raw).hexdigest(),
    }
    assert load(spec) == hot
    assert lookup(hot["18"])[284] == 7
    assert lookup(hot["18"])[2] == -1
    keys = cuda_keys(hot, gate_layers=(18, 19))
    assert keys[:8] == tuple((18, e) for e in hot["18"])
    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="authentication"):
        load(spec)


@pytest.mark.parametrize("kind", ["missing_layer", "duplicate", "unsorted", "negative", "too_large", "boolean"])
def test_placement_rejects_incomplete_or_invalid_ownership(kind):
    hot = hot_placement()
    if kind == "missing_layer":
        del hot["18"]
    elif kind == "duplicate":
        hot["18"] = [1, 1]
    elif kind == "unsorted":
        hot["18"] = [2, 1]
    elif kind == "negative":
        hot["18"] = [-1]
    elif kind == "too_large":
        hot["18"] = [288]
    else:
        hot["18"] = [True]
    with pytest.raises(ValueError):
        validate(hot)
