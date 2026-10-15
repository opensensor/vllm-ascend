# SPDX-License-Identifier: Apache-2.0
"""File-relative admission tests; no physical I/O or NPU access."""

import json
import os
import struct

import pytest

from tools.qwen4exp.ple_storage import MAX_LOOKUP_ROWS, register


def checkpoint(root, *, rows=40, columns=160, mutation=None):
    header, payload, weights = {}, bytearray(), {}
    for shard in range(3):
        key = f"model.ple.ngram_embedding.shard_{shard}.weight"
        start = len(payload)
        payload.extend(bytes((index + 17 * shard) % 256 for index in range(rows * columns * 2)))
        header[key] = dict(dtype="F16", shape=[rows, columns], data_offsets=[start, len(payload)])
        weights[key] = "shards.safetensors"
    if mutation:
        mutation(header, weights)
    raw = json.dumps(header).encode()
    (root / "shards.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    (root / "index.json").write_text(json.dumps(dict(weight_map=weights)))
    return bytes(payload)


def materialize(storage, plan, count):
    pages = []
    for item in plan.reads:
        with open(storage.files[item.file].path, "rb") as stream:
            stream.seek(item.offset)
            pages.append(stream.read(item.length))
    output = [bytearray(storage.shards[0].row_bytes) for _ in range(count)]
    for item in plan.slices:
        data = pages[item.read][item.source_offset : item.source_offset + item.length]
        output[item.destination_row][item.destination_offset : item.destination_offset + item.length] = data
    return output


@pytest.mark.parametrize("page_bytes", [512, 1024, 4096, 8192])
def test_all_rows_cross_pages_and_shards_exactly(tmp_path, page_bytes):
    source = checkpoint(tmp_path)
    storage = register(tmp_path, index_filename="index.json", shard_count=3)
    ids = [119, 0, 39, 40, 79, 80, *range(120), 3, 3]
    plan = storage.plan(ids, page_bytes=page_bytes)
    assert materialize(storage, plan, len(ids)) == [source[row * 320 : (row + 1) * 320] for row in ids]
    assert len({(read.file, read.offset) for read in plan.reads}) == len(plan.reads)
    assert plan.useful_bytes == len(ids) * 320
    assert plan.fetched_bytes <= storage.files[0].size
    assert all(read.offset + read.length <= storage.files[read.file].size for read in plan.reads)
    assert storage.table_bytes == 120 * 320
    assert storage.manifest()["physical_extents_pinned"] is False


@pytest.mark.parametrize("ids", [[], [-1], [120], [1.5], [True], [0] * (MAX_LOOKUP_ROWS + 1)])
def test_unadmitted_rows_are_rejected_before_io(tmp_path, ids):
    checkpoint(tmp_path)
    storage = register(tmp_path, index_filename="index.json", shard_count=3)
    with pytest.raises(ValueError):
        storage.plan(ids)


@pytest.mark.parametrize("page_bytes", [0, 256, 513, -4096, True, 512.0])
def test_invalid_page_geometry(tmp_path, page_bytes):
    checkpoint(tmp_path)
    storage = register(tmp_path, index_filename="index.json", shard_count=3)
    with pytest.raises(ValueError):
        storage.plan([0], page_bytes=page_bytes)


@pytest.mark.parametrize("mode", ["dtype", "rows", "columns", "length", "overlap", "missing", "escape"])
def test_invalid_manifest(tmp_path, mode):
    def mutate(header, weights):
        names = list(header)
        entry = header[names[1]]
        if mode == "dtype":
            entry["dtype"] = "BF16"
        elif mode == "rows":
            entry["shape"][0] -= 1
        elif mode == "columns":
            entry["shape"][1] -= 1
        elif mode == "length":
            entry["data_offsets"][1] += 2
        elif mode == "overlap":
            entry["data_offsets"] = header[names[0]]["data_offsets"]
        elif mode == "missing":
            weights.pop(names[1])
        else:
            outside = tmp_path.parent / "outside-ple.safetensors"
            outside.write_bytes(b"do not read")
            weights[names[1]] = str(outside)

    checkpoint(tmp_path, mutation=mutate)
    with pytest.raises(ValueError):
        register(tmp_path, index_filename="index.json", shard_count=3)


def test_changed_file_identity_is_not_an_extent_lease(tmp_path):
    checkpoint(tmp_path)
    storage = register(tmp_path, index_filename="index.json", shard_count=3)
    storage.files[0].verify()
    path = tmp_path / "shards.safetensors"
    original = path.read_bytes()
    path.unlink()
    path.write_bytes(original)
    os.utime(path, ns=(storage.files[0].mtime_ns + 1, storage.files[0].mtime_ns + 1))
    with pytest.raises(RuntimeError, match="identity or contents changed"):
        storage.files[0].verify()


@pytest.mark.parametrize("data", [b"", b"123", struct.pack("<Q", 2**63), struct.pack("<Q", 200) + b"{}"])
def test_truncated_or_unbounded_header(tmp_path, data):
    checkpoint(tmp_path)
    (tmp_path / "shards.safetensors").write_bytes(data)
    with pytest.raises(ValueError, match="header"):
        register(tmp_path, index_filename="index.json", shard_count=3)
