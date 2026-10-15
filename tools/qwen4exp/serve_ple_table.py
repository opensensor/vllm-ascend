# SPDX-License-Identifier: Apache-2.0
"""Own one sealed PLE RAM table for local resident serving workers."""

import argparse
import json
import os
import signal
from pathlib import Path

from tools.qwen4exp.ple_sources import SharedRamSource
from tools.qwen4exp.ple_storage import register


def serve(model, ready):
    # Imported only in this cold owner process, without initializing an NPU.
    from safetensors import safe_open

    model = model.resolve(strict=True)
    if ready.exists():
        raise ValueError("use a new readiness file")
    storage = register(model, index_filename="model.safetensors.index.json")
    weights = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    cfg = json.loads((model / "config.json").read_text())["text_config"]
    geometry = dict(eos=cfg["eos_token_id"], vocabulary=cfg["vocab_size"], table_rows=storage.table_bytes // 320)
    for name, suffix in (
        ("multipliers", "layer_multipliers"),
        ("sizes", "ngram_heads_vocab_sizes"),
        ("offsets", "ngram_heads_offsets"),
    ):
        key = next(key for key in weights if key.endswith("ple_embedding." + suffix))
        with safe_open(model / weights[key], framework="numpy") as file:
            geometry[name] = file.get_tensor(key).reshape(-1).tolist()
    print(f"Loading {storage.table_bytes} PLE bytes into one sealed memfd", flush=True)
    with SharedRamSource.load(storage, budget_bytes=storage.table_bytes, reserve_bytes=48 * 1024**3) as source:
        fd = source.duplicate_fd()
        try:
            info = os.fstat(fd)
            metadata = dict(
                model=str(model),
                fd_path=f"/proc/{os.getpid()}/fd/{fd}",
                inode=info.st_ino,
                device=info.st_dev,
                bytes=info.st_size,
                geometry=geometry,
            )
            with ready.open("x") as stream:
                json.dump(metadata, stream, indent=2)
            print(f"Shared table ready: {ready}", flush=True)
            signal.pause()
        finally:
            os.close(fd)
            ready.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
    serve(args.model, args.ready)


if __name__ == "__main__":
    main()
