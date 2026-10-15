"""Small real-weight shared-RAM + filesystem gate; no serving mutation."""

import json
import os
import random
import subprocess
from dataclasses import replace
from pathlib import Path

from safetensors import safe_open

from tools.qwen4exp.ple_sources import FileSource, SharedRamSource
from tools.qwen4exp.ple_storage import register

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
MODEL = Path("/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i")
SAMPLE_ROWS_PER_SHARD = 64
OUTPUT = ROOT / "ple-shared-source-real-weights-v1.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    index = "model.safetensors.index.json"
    if not (MODEL / index).exists():
        index = "quant_model_weights.safetensors.index.json"
    full = register(MODEL, index_filename=index)
    sample = replace(full, shards=tuple(replace(shard, rows=SAMPLE_ROWS_PER_SHARD) for shard in full.shards))
    weights = json.loads((MODEL / index).read_text())["weight_map"]
    rng = random.Random(14102026)
    checked_rows = 0
    rank_results = []
    with (
        FileSource(sample) as files,
        SharedRamSource.load(sample, budget_bytes=4 * 1024 * 1024, reserve_bytes=48 * 1024**3) as ram,
    ):
        for shard in range(len(sample.shards)):
            local_ids = [0, 12, 63, rng.randrange(SAMPLE_ROWS_PER_SHARD), 12]
            ids = [shard * SAMPLE_ROWS_PER_SHARD + row for row in local_ids]
            key = next(key for key in weights if key.endswith(f"ngram_embedding.shard_{shard}.weight"))
            with safe_open(MODEL / weights[key], framework="numpy") as source:
                tensor = source.get_slice(key)
                expected = b"".join(tensor[row : row + 1].tobytes() for row in local_ids)
            assert files.read_rows(ids) == ram.read_rows(ids) == expected
            checked_rows += len(ids)
        fd = ram.duplicate_fd()
        try:
            identity = os.fstat(fd)
            for rank in range(4):
                result = subprocess.run(
                    [str(ROOT / "ple_host_dma_probe"), str(rank), str(fd)],
                    pass_fds=(fd,), capture_output=True, text=True, check=True, timeout=30,
                )
                records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
                copies = [record for record in records if "source" in record]
                assert len(copies) == 5 and all(record.get("exact") for record in copies), records
                rank_results.append(dict(rank=rank, shared_inode=identity.st_ino, records=records))
        finally:
            os.close(fd)
    report = dict(
        passed=True, checked_rows=checked_rows, sample_bytes=sample.table_bytes,
        full_table_bytes=full.table_bytes, ranks=rank_results,
        full_table_loaded=False, device_initiated=False, host_submitted=True,
        ssd_device_io=False, server_modified=False,
    )
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
