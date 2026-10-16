# SPDX-License-Identifier: Apache-2.0
"""Private bounded diagnostic capture for the existing stage/cache objects.

T11 hook contract (gate mode only): create one DiagnosticWriter per owner,
call capture_boundary at each declared handoff and capture_selection at the
actual route/indexer result, then capture_layers after prefill and decode.
The coordinator's execution_diagnostics adapter implements begin_fixture(plan,
directory), finish_phase(phase), end_fixture(). finish_phase waits for both
rank manifests and returns nothing; end_fixture returns {owner: manifest_path}.
Owners must finish all writes before replying, outside production timing.

Capture helpers never replace, resize, clone, or reset a cache. Only active KV,
pool and tail regions are semantic state; uninitialized reserved capacity is
excluded, and its full allocation geometry is recorded separately. Diagnostics
copy at most max_chunk_bytes per transfer and retain no complete CPU cache.
An owner must synchronize its own stream before observation; this explicit
gate-only synchronization is deliberately not part of these storage helpers.
"""

import hashlib
import json
import os
from pathlib import Path

DEFAULT_CHUNK_BYTES = 1 << 20
DEFERRED_FIELDS = ("hidden", "residual", "post", "comb")
TENSOR_KINDS = frozenset({"floating_state", "vocabulary", "selectors", "integer_boolean_state"})
DTYPE_BYTES = {"float16": 2, "float32": 4, "float64": 8, "int64": 8, "int32": 4, "uint8": 1, "bool": 1}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(DEFAULT_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def private_json(path, value):
    path = Path(path)
    with open(path, "w", opener=lambda name, flags: os.open(name, flags, 0o600)) as target:
        json.dump(value, target, sort_keys=True, indent=2, allow_nan=False)
        target.write("\n")


class DiagnosticWriter:
    """Append logical row segments to authenticated, private tensor files.

    All state/selection hooks append along axis zero. Offsets are checked before
    data transfer, detecting skipped, repeated or reordered prompt chunks.
    Contiguous views are required so flattening never allocates a device copy.
    Torch and NumPy are imported only when a tensor is actually observed.
    """

    def __init__(self, directory, owner, *, max_chunk_bytes=DEFAULT_CHUNK_BYTES):
        if type(max_chunk_bytes) is not int or max_chunk_bytes < 8:
            raise ValueError("diagnostic transfer bound must be at least eight bytes")
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.owner, self.max_chunk_bytes = owner, max_chunk_bytes
        self.records, self.closed = {}, False

    def append(self, key, tensor, kind, *, offset=0):
        if self.closed or kind not in TENSOR_KINDS or not key:
            raise ValueError("invalid or closed diagnostic observation")
        dtype = str(tensor.dtype).removeprefix("torch.")
        shape = list(tensor.shape)
        if not shape or dtype not in DTYPE_BYTES or any(type(size) is not int or size < 0 for size in shape):
            raise ValueError("diagnostic tensor needs a supported dtype and explicit row axis")
        integer = dtype in ("int64", "int32", "uint8", "bool")
        if integer != (kind in ("selectors", "integer_boolean_state")):
            raise ValueError("diagnostic semantic kind and dtype disagree")
        if type(offset) is not int or offset < 0:
            raise ValueError("invalid diagnostic row offset")
        previous = self.records.get(key)
        if previous is None:
            if offset != 0:
                raise ValueError("diagnostic tensor starts after a missing chunk")
            record = dict(
                key=key,
                kind=kind,
                dtype=dtype,
                shape=[0, *shape[1:]],
                path=hashlib.sha256(key.encode()).hexdigest() + ".bin",
                segments=[],
            )
        else:
            record = previous
            if record["shape"][1:] != shape[1:] or record["dtype"] != dtype or record["kind"] != kind:
                raise ValueError("diagnostic segment changed tensor geometry")
        if record["shape"][0] != offset:
            raise ValueError("diagnostic chunks overlap or leave a gap")
        path = self.directory / record["path"]
        if type(tensor).__module__.split(".")[0] == "torch":
            if not tensor.is_contiguous():
                raise ValueError("diagnostic capture requires a contiguous view")
            flat = tensor.detach().view(-1)
            count = max(1, self.max_chunk_bytes // tensor.element_size())

            def chunks():
                for start in range(0, flat.numel(), count):
                    yield flat[start : start + count].cpu().numpy().tobytes(order="C")
        else:
            import numpy as np

            array = np.asarray(tensor)
            if not array.flags.c_contiguous or not array.dtype.isnative:
                raise ValueError("diagnostic capture requires native contiguous storage")
            flat = memoryview(array).cast("B") if array.size else memoryview(b"")

            def chunks():
                for start in range(0, len(flat), self.max_chunk_bytes):
                    yield flat[start : start + self.max_chunk_bytes]

        with open(path, "ab", opener=lambda name, flags: os.open(name, flags, 0o600)) as target:
            for chunk in chunks():
                if len(chunk) > self.max_chunk_bytes:
                    raise ValueError("diagnostic transfer exceeded its configured bound")
                target.write(chunk)
        record["segments"].append({"offset": offset, "rows": shape[0]})
        record["shape"][0] += shape[0]
        self.records[key] = record

    def scalar(self, key, value):
        import numpy as np

        if type(value) is not int or value < 0:
            raise ValueError("owner-local diagnostic lengths must be nonnegative Python integers")
        self.append(key, np.array([value], dtype=np.int64), "integer_boolean_state")

    def close(self, *, allocation_geometry=None):
        if self.closed:
            raise ValueError("diagnostic writer was already finalized")
        self.closed = True
        for record in self.records.values():
            path = self.directory / record["path"]
            record.update(sha256=sha256_file(path), bytes=path.stat().st_size)
        path = self.directory / "manifest.json"
        private_json(
            path,
            dict(
                schema_version=1,
                owner=self.owner,
                complete=True,
                max_chunk_bytes=self.max_chunk_bytes,
                tensors=list(self.records.values()),
                allocation_geometry=allocation_geometry or {},
                unused_capacity="excluded: only initialized active KV/pool/tail regions are semantic state",
            ),
        )
        return path


def capture_boundary(writer, prefix, state, *, offset):
    if len(state) != len(DEFERRED_FIELDS) or any(value is None for value in state):
        raise ValueError("complete deferred hidden/residual/post/comb required at stage handoff")
    for name, value in zip(DEFERRED_FIELDS, state, strict=True):
        writer.append(f"{prefix}/{name}", value, "floating_state", offset=offset)


def capture_chunk(writer, position, rows, *, iteration):
    """Record the actual owner iteration, after its completion fence."""
    import numpy as np

    if any(type(value) is not int or value < 0 for value in (position, iteration)) or type(rows) is not int or rows < 1:
        raise ValueError("invalid actual owner iteration")
    writer.append("chunks", np.array([[position, rows]], dtype=np.int64), "integer_boolean_state", offset=iteration)


def capture_vocabulary(writer, logits, *, step=None, prompt_position=None, prompt_chunk=None):
    """Retain the full vector already produced by the real owner head path.

    Final prefill vector is both the last chunk vector and step zero. Replayed
    teacher steps are 1..255. No additional head calculation is introduced.
    """
    if len(logits.shape) == 1:
        logits = logits.reshape(1, -1)
    if len(logits.shape) != 2 or logits.shape[0] != 1:
        raise ValueError("one full output vocabulary vector required")
    if step is not None:
        writer.append("vocabulary", logits, "vocabulary", offset=step)
    if prompt_position is not None:
        import numpy as np

        if type(prompt_position) is not int or prompt_position < 0 or type(prompt_chunk) is not int:
            raise ValueError("actual prompt chunk position/index required")
        writer.append("chunk_vocabulary", logits, "vocabulary", offset=prompt_chunk)
        writer.append(
            "chunk_positions",
            np.array([[prompt_position]], dtype=np.int64),
            "integer_boolean_state",
            offset=prompt_chunk,
        )


def capture_selection(writer, prefix, value, *, offset, rows=None, width=None):
    """Observe actual expert IDs or sparse indices, never recompute a selector.

    DSA hooks pass width=index_topk+3, rows=chunk_rows. Missing/unused columns
    are -1, including a None dense selection; these are a diagnostic encoding,
    not a replacement for the actual selector. Padding is bounded CPU work.
    """
    if width is None:
        writer.append(prefix, value, "selectors", offset=offset)
        return
    import numpy as np

    if type(rows) is not int or rows < 1 or type(width) is not int or width < 1:
        raise ValueError("selection padding requires positive declared rows/width")
    if width * 8 > writer.max_chunk_bytes:
        raise ValueError("selection row exceeds diagnostic transfer bound")
    if value is not None and (list(value.shape)[0] != rows or value.shape[1] > width):
        raise ValueError("actual selection exceeds declared diagnostic geometry")
    row_chunk = max(1, writer.max_chunk_bytes // (width * 8))
    for start in range(0, rows, row_chunk):
        count = min(row_chunk, rows - start)
        padded = np.full((count, width), -1, dtype=np.int64)
        if value is not None:
            observed = value[start : start + count]
            if type(observed).__module__.split(".")[0] == "torch":
                observed = observed.detach().cpu().numpy()
            if str(observed.dtype) != "int64":
                raise ValueError("actual selection must retain its int64 dtype")
            padded[:, : observed.shape[1]] = observed
        writer.append(prefix, padded, "selectors", offset=offset + start)


def capture_route(writer, prefix, ids, weights, *, offset):
    """Capture both the actual route choice and FP32 weighting operands."""
    writer.append(f"{prefix}/route", ids, "selectors", offset=offset)
    writer.append(f"{prefix}/route_weights", weights, "floating_state", offset=offset)


def capture_layers(writer, prefix, layers):
    """Traverse actual TextLayer / PoolIndexer fields without device clones."""
    geometry = {}
    values = layers.values() if hasattr(layers, "values") else layers
    for layer in values:
        key = f"{prefix}/layer{layer.index}"
        writer.scalar(f"{key}/length", layer.length)
        if layer.linear:
            writer.append(f"{key}/recurrent", layer.state, "floating_state")
            writer.append(f"{key}/conv", layer.conv, "floating_state")
            geometry[str(layer.index)] = dict(recurrent=list(layer.state.shape), conv=list(layer.conv.shape))
        else:
            indexer = layer.indexer
            if indexer.pool_count != layer.length // 4 or indexer.tail_length != layer.length % 4:
                raise ValueError("indexer pool/tail coverage does not match initialized KV length")
            writer.append(f"{key}/kv", layer.cache[: layer.length], "floating_state")
            writer.append(f"{key}/pools", indexer.pools[: indexer.pool_count], "floating_state")
            writer.append(f"{key}/tail_keys", indexer.tail_keys[: indexer.tail_length], "floating_state")
            writer.append(f"{key}/tail_gates", indexer.tail_gates[: indexer.tail_length], "floating_state")
            writer.scalar(f"{key}/pool_count", indexer.pool_count)
            writer.scalar(f"{key}/tail_length", indexer.tail_length)
            geometry[str(layer.index)] = dict(
                kv=list(layer.cache.shape),
                pools=list(indexer.pools.shape),
                tail_keys=list(indexer.tail_keys.shape),
                tail_gates=list(indexer.tail_gates.shape),
            )
    return geometry
