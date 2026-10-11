# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.sidecard.cuda_bank_cache import MANIFEST_NAME, CachedCudaBanks, build_cache, tensor_sizes
from tools.sidecard.device_backend import CUDA_ROW_MAJOR, CUDA_WORD_PLANES, encode_cuda_weights
from tools.sidecard.native_expert_probe import digest_file


@pytest.fixture
def cache_fixture(tmp_path):
    values = [
        np.arange(64 * 64, dtype=np.int8).reshape(1, 64, 64),
        np.ones((1, 2, 2), dtype=np.float16),
        np.arange(64 * 32, dtype=np.int8).reshape(1, 64, 32),
        np.ones((1, 2, 1), dtype=np.float16),
    ]
    tensors = tuple(SimpleNamespace(numpy=lambda v=v: v) for v in values)
    plan = SimpleNamespace(
        by_layer={3: (5,)},
        authenticate=lambda: None,
        check_unchanged=lambda: None,
        items=lambda: iter([(3, dict(tensors=tensors))]),
    )
    root = tmp_path / "cache"
    report = build_cache(plan, root, "a" * 64, CUDA_WORD_PLANES, hidden=64, intermediate=32, unpacker=lambda t, k: t)

    def load():
        return CachedCudaBanks(
            root, report["manifest_sha256"], "a" * 64, ((3, 5),), CUDA_WORD_PLANES, hidden=64, intermediate=32
        )

    return root, report, values, load, plan


def test_cached_banks_round_trip_without_re_unpacking(cache_fixture):
    _, report, values, load, _ = cache_fixture
    banks = load()
    with pytest.raises(RuntimeError):
        list(banks.items())
    banks.authenticate()
    layer, bank = next(banks.items())
    assert layer == 3
    expected = tuple(
        encode_cuda_weights(v, CUDA_WORD_PLANES) if i in (0, 2) else v.tobytes() for i, v in enumerate(values)
    )
    assert bank["encoded"] == expected
    assert report["required_cuda_bytes"] == sum(tensor_sizes(1, 64, 32))


def test_corrupt_cached_payload_fails_before_upload(cache_fixture):
    root, _, _, load, _ = cache_fixture
    path = root / "layer-3.bin"
    raw = bytearray(path.read_bytes())
    raw[0] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum"):
        load().authenticate()


def test_cached_payload_changed_after_authentication_is_rejected(cache_fixture):
    root, _, _, load, _ = cache_fixture
    banks = load()
    banks.authenticate()
    (root / "layer-3.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        list(banks.items())


def test_cache_manifest_is_pinned(cache_fixture):
    root, _, _, load, _ = cache_fixture
    (root / MANIFEST_NAME).write_text("{}")
    with pytest.raises(ValueError, match="pinned"):
        load()


@pytest.mark.parametrize(
    "field,value", [("weight_layout", "unknown"), ("experts_by_layer", {"3": [6]}), ("required_cuda_bytes", 0)]
)
def test_cache_manifest_semantics_cannot_be_relabelled(cache_fixture, field, value):
    root, _, _, _, _ = cache_fixture
    path = root / MANIFEST_NAME
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        CachedCudaBanks(root, digest_file(path), "a" * 64, ((3, 5),), CUDA_WORD_PLANES, hidden=64, intermediate=32)


def test_cache_builder_preserves_existing_destination(cache_fixture):
    root, _, _, _, plan = cache_fixture
    with pytest.raises(ValueError, match="fresh"):
        build_cache(plan, root, "a" * 64, CUDA_WORD_PLANES, hidden=64, intermediate=32, unpacker=lambda t, k: t)


def test_selected_cached_layers_restore_row_major_bytes(cache_fixture):
    _, _, values, load, _ = cache_fixture
    _, bank = next(load().selected_items((3,), layout=CUDA_ROW_MAJOR))
    assert bank["encoded"] == tuple(v.tobytes() for v in values)
    with pytest.raises(ValueError, match="foreign"):
        list(load().selected_items((4,)))


def test_selected_cached_layer_requires_payload_authentication(cache_fixture):
    root, _, _, load, _ = cache_fixture
    path = root / "layer-3.bin"
    raw = bytearray(path.read_bytes())
    raw[0] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum"):
        list(load().selected_items((3,)))
