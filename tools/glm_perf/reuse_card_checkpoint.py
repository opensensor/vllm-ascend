# SPDX-License-Identifier: Apache-2.0
"""Rebuild exported shard bytes from host-local tensors, without quantization.

The template contains original safetensors headers and full shard SHA256 values.
Every replacement is checked against that SHA before atomic publication; sources
remain read-only. Copy config and kernel assets separately, then run the complete
card-checkpoint verifier before loading the model.
"""

import argparse
import base64
import hashlib
import json
import os
import struct
import tempfile
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path

COPY_BYTES = 8 * 1024**2
MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024**2


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_BYTES), b""):
            result.update(chunk)
    return result.hexdigest()


def header(handle):
    prefix = handle.read(8)
    if len(prefix) != 8:
        raise ValueError("truncated safetensors prefix")
    count = struct.unpack("<Q", prefix)[0]
    if not 0 < count <= MAX_SAFETENSORS_HEADER_BYTES:
        raise ValueError("invalid safetensors header size")
    raw = handle.read(count)
    if len(raw) != count:
        raise ValueError("truncated safetensors header")
    return prefix + raw, json.loads(raw)


def within(folder, filename):
    path = folder / filename
    if not path.resolve().is_relative_to(folder.resolve()):
        raise ValueError("shard path escapes checkpoint")
    return path


def make_template(checkpoint):
    """Read headers only; caller must first verify the original export's SHA."""
    checkpoint = Path(checkpoint).resolve()
    manifest = json.loads((checkpoint / "native-layout.json").read_text())
    if manifest.get("complete") is not True:
        raise ValueError("template requires a complete exported checkpoint")
    result = {}
    for field in ("native_shards", "scale_shards", "dense_shards"):
        for record in manifest[field]:
            filename = record["file"]
            path = within(checkpoint, filename)
            if path.stat().st_size != record["bytes"]:
                raise ValueError(f"template shard byte count differs: {filename}")
            with path.open("rb") as handle:
                raw, _ = header(handle)
            value = {"header": base64.b64encode(raw).decode(), "sha256": record["sha256"], "bytes": record["bytes"]}
            if filename in result and result[filename] != value:
                raise ValueError("conflicting exported shard records")
            result[filename] = value
    return result


def rebuild(template, sources, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    indices = []
    for source in sources:
        source = Path(source).resolve()
        if source == destination:
            raise ValueError("source and destination must differ")
        index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
        indices.append((source, index))
    report = {"reused_shards": 0, "rebuilt_shards": 0, "rebuilt_bytes": 0}
    for filename, record in template.items():
        target = within(destination, filename)
        if target.exists() and target.stat().st_size == record["bytes"] and digest(target) == record["sha256"]:
            report["reused_shards"] += 1
            continue
        encoded = base64.b64decode(record["header"], validate=True)
        raw, desired = header(BytesIO(encoded))
        if raw != encoded:
            raise ValueError("template has trailing header bytes")
        tensors = sorted(
            ((name, entry) for name, entry in desired.items() if name != "__metadata__"),
            key=lambda item: item[1]["data_offsets"][0],
        )
        descriptor, name = tempfile.mkstemp(prefix=target.name + ".reuse-", dir=target.parent)
        os.close(descriptor)
        temporary = Path(name)
        try:
            with ExitStack() as stack:
                handles = {}
                output = stack.enter_context(temporary.open("wb"))
                output.write(raw)
                position = 0
                for name, wanted in tensors:
                    start, end = wanted["data_offsets"]
                    if start != position or end < start:
                        raise ValueError("template has invalid tensor offsets")
                    selected = None
                    for source, index in indices:
                        if name not in index:
                            continue
                        path = within(source, index[name])
                        if path not in handles:
                            stream = stack.enter_context(path.open("rb"))
                            source_raw, source_header = header(stream)
                            handles[path] = stream, len(source_raw), source_header
                        stream, offset, source_header = handles[path]
                        entry = source_header.get(name, {})
                        if entry.get("shape") == wanted["shape"] and entry.get("dtype") == wanted["dtype"]:
                            first, last = entry["data_offsets"]
                            if last - first == end - start:
                                selected = stream, offset + first
                                break
                    if selected is None:
                        raise ValueError(f"no host-local source for {name}")
                    stream, offset = selected
                    stream.seek(offset)
                    remaining = end - start
                    while remaining:
                        chunk = stream.read(min(COPY_BYTES, remaining))
                        if not chunk:
                            raise ValueError(f"truncated source tensor {name}")
                        output.write(chunk)
                        remaining -= len(chunk)
                    position = end
                output.flush()
            if temporary.stat().st_size != record["bytes"] or digest(temporary) != record["sha256"]:
                raise ValueError(f"host-local tensors differ from exported shard: {filename}")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        report["rebuilt_shards"] += 1
        report["rebuilt_bytes"] += record["bytes"]
        print(json.dumps({"file": filename, **report}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-checkpoint", help="Write a header plan from an already SHA-verified export")
    parser.add_argument("--template")
    parser.add_argument("--source", action="append", help="Ordered source checkpoints")
    parser.add_argument("--destination")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.template_checkpoint:
        if args.template or args.source or args.destination:
            parser.error("template creation cannot also rebuild shards")
        report = make_template(args.template_checkpoint)
    else:
        if not args.template or not args.source or not args.destination:
            parser.error("rebuild requires --template, --source, and --destination")
        report = rebuild(json.loads(Path(args.template).read_text()), args.source, args.destination)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
