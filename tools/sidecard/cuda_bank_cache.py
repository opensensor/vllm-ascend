# SPDX-License-Identifier: Apache-2.0
"""Build and stream pinned, authenticated CUDA bank bytes without re-unpacking."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from .device_backend import CUDA_ROW_MAJOR, CUDA_WORD_PLANES, encode_cuda_weights
from .native_expert_probe import contained, digest_file
from .wire import unique_metadata

MANIFEST_NAME = "cuda-bank-cache.json"
HOST_DISK_RESERVE_BYTES = 8 * 1024**3


def tensor_sizes(experts, hidden, intermediate):
    if any(type(n) is not int or n < 1 for n in (experts, hidden, intermediate)) or hidden % 32 or intermediate % 32:
        raise ValueError("invalid CUDA cache geometry")
    return (
        experts * 2 * intermediate * hidden,
        experts * (2 * intermediate // 32) * (hidden // 32) * 2,
        experts * hidden * intermediate,
        experts * (hidden // 32) * (intermediate // 32) * 2,
    )


def build_cache(plan, output, checkpoint_sha, layout, *, hidden=4096, intermediate=2048, unpacker=None):
    if layout not in (CUDA_ROW_MAJOR, CUDA_WORD_PLANES):
        raise ValueError("unsupported CUDA cache layout")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("CUDA cache requires a fresh directory")
    required = sum(sum(tensor_sizes(len(experts), hidden, intermediate)) for experts in plan.by_layer.values())
    if shutil.disk_usage(output.parent).free < required + HOST_DISK_RESERVE_BYTES:
        raise ValueError("CUDA cache would exhaust the host disk reserve")
    if unpacker is None:
        # CPU-only unpacking inside the preparation process.
        from tools.glm_perf.fused_weight_layout import unpack_cube

        unpacker = unpack_cube
    started = time.monotonic()
    plan.authenticate()
    output.mkdir(mode=0o700)
    records = []
    for layer, bank in plan.items():
        path = output / f"layer-{layer}.bin"
        expected_sizes = tensor_sizes(len(plan.by_layer[layer]), hidden, intermediate)
        offsets, offset, checksum = [], 0, hashlib.sha256()
        with path.open("xb") as handle:
            for index, tensor in enumerate(bank["tensors"]):
                values = (
                    unpacker(tensor, hidden if index == 0 else intermediate).numpy()
                    if index in (0, 2)
                    else tensor.numpy()
                )
                data = encode_cuda_weights(values, layout) if index in (0, 2) else values.tobytes()
                if len(data) != expected_sizes[index] or (index in (1, 3) and values.dtype.name != "float16"):
                    raise ValueError("CUDA cache tensor geometry differs")
                handle.write(data)
                checksum.update(data)
                offsets.append([offset, offset + len(data)])
                offset += len(data)
                del data, values
        plan.check_unchanged()
        records.append(dict(layer=layer, file=path.name, size=offset, sha256=checksum.hexdigest(), offsets=offsets))
        print(json.dumps(dict(kind="cuda_cache_layer", layer=layer, bytes=offset)), flush=True)
    plan.check_unchanged()
    if [record["layer"] for record in records] != list(plan.by_layer):
        raise ValueError("CUDA cache builder did not stream every planned layer")
    manifest = dict(
        schema_version=1,
        checkpoint_sha256=checkpoint_sha,
        weight_layout=layout,
        hidden_width=hidden,
        intermediate_width=intermediate,
        experts_by_layer={str(layer): list(experts) for layer, experts in plan.by_layer.items()},
        layers=records,
        required_cuda_bytes=required,
        build_seconds=time.monotonic() - started,
    )
    path = output / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return dict(root=str(output), manifest_sha256=digest_file(path), **manifest)


class CachedCudaBanks:
    def __init__(self, root, manifest_sha, checkpoint_sha, keys, layout, *, hidden=4096, intermediate=2048):
        self.root = Path(root).resolve()
        path = self.root / MANIFEST_NAME
        if digest_file(path) != manifest_sha:
            raise ValueError("CUDA cache manifest differs from pinned identity")
        value = json.loads(path.read_bytes(), object_pairs_hook=unique_metadata)
        by_layer = {}
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("CUDA cache ownership is empty or duplicated")
        for layer, expert in keys:
            by_layer.setdefault(str(layer), []).append(expert)
        if (
            value["schema_version"] != 1
            or value["checkpoint_sha256"] != checkpoint_sha
            or value["weight_layout"] != layout
            or value["experts_by_layer"] != by_layer
            or value["hidden_width"] != hidden
            or value["intermediate_width"] != intermediate
        ):
            raise ValueError("CUDA cache checkpoint, layout or ownership differs")
        self.records = value["layers"]
        self.layout = layout
        self.by_layer = by_layer
        self.hidden, self.intermediate = hidden, intermediate
        if [str(row["layer"]) for row in self.records] != list(by_layer):
            raise ValueError("CUDA cache layer coverage differs")
        for row in self.records:
            sizes = tensor_sizes(len(by_layer[str(row["layer"])]), hidden, intermediate)
            offsets = []
            offset = 0
            for size in sizes:
                offsets.append([offset, offset + size])
                offset += size
            if row["offsets"] != offsets or row["size"] != offset or row["file"] != f"layer-{row['layer']}.bin":
                raise ValueError("CUDA cache tensor byte ranges differ")
            if contained(self.root, row["file"]).stat().st_size != offset:
                raise ValueError("CUDA cache payload size differs")
        if value["required_cuda_bytes"] != sum(row["size"] for row in self.records):
            raise ValueError("CUDA cache allocation ledger differs")
        self.snapshots = {
            name: self.snapshot(contained(self.root, name))
            for name in [MANIFEST_NAME] + [r["file"] for r in self.records]
        }
        self.authenticated = False

    @staticmethod
    def snapshot(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def check_unchanged(self):
        if any(self.snapshot(contained(self.root, name)) != state for name, state in self.snapshots.items()):
            raise ValueError("CUDA cache changed during loading")

    def authenticate(self):
        for row in self.records:
            if digest_file(contained(self.root, row["file"])) != row["sha256"]:
                raise ValueError("CUDA cache payload checksum differs")
        self.check_unchanged()
        self.authenticated = True

    def items(self):
        if not self.authenticated:
            raise RuntimeError("CUDA cache is not authenticated")
        for row in self.records:
            yield row["layer"], dict(encoded=self.read_row(row))

    def read_row(self, row):
        self.check_unchanged()
        with contained(self.root, row["file"]).open("rb") as handle:
            values = tuple(handle.read(stop - start) for start, stop in row["offsets"])
        if tuple(map(len, values)) != tuple(stop - start for start, stop in row["offsets"]):
            raise ValueError("truncated CUDA cache payload")
        self.check_unchanged()
        return values

    def selected_items(self, layers, *, layout=None):
        # Hardware microgates authenticate only selected layers; they do not
        # claim that the full resident partition has been admitted.
        import numpy as np

        layout = layout or self.layout
        if layout not in (CUDA_ROW_MAJOR, CUDA_WORD_PLANES) or not layers or len(set(layers)) != len(layers):
            raise ValueError("invalid CUDA cache subset or layout")
        if any(str(layer) not in self.by_layer for layer in layers):
            raise ValueError("CUDA cache subset has a foreign layer")
        for row in self.records:
            if row["layer"] not in layers:
                continue
            if digest_file(contained(self.root, row["file"])) != row["sha256"]:
                raise ValueError("CUDA cache subset checksum differs")
            values = list(self.read_row(row))
            if layout != self.layout:
                for index, k in ((0, self.hidden), (2, self.intermediate)):
                    n = 2 * self.intermediate if index == 0 else self.hidden
                    e = len(self.by_layer[str(row["layer"])])
                    raw = np.frombuffer(values[index], dtype=np.int8)
                    if self.layout == CUDA_WORD_PLANES:
                        values[index] = raw.reshape(e, k // 32, 8, n, 4).transpose(0, 3, 1, 2, 4).tobytes()
                    else:
                        values[index] = encode_cuda_weights(raw.reshape(e, n, k), layout)
            yield row["layer"], dict(encoded=tuple(values))


def main():
    from .device_backend import owned_keys
    from .resident import NativeResidentPlan
    from .service_adapter import service_placement

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=(CUDA_ROW_MAJOR, CUDA_WORD_PLANES), default=CUDA_WORD_PLANES)
    args = parser.parse_args()
    plan = NativeResidentPlan(args.checkpoint, owned_keys(service_placement(), "cuda:0"), args.checkpoint_sha256)
    print(json.dumps(build_cache(plan, args.output, args.checkpoint_sha256, args.layout)), flush=True)


if __name__ == "__main__":
    main()
