"""Private six-device transport gate against the real shared PLE table."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from tools.qwen4exp.ple_serving import Lookup, ServingPLE, checked

ROOT = Path("/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016")


def expected_rows(tokens, history, starts, geometry):
    result = []
    for request, (start, end) in enumerate(zip(starts, starts[1:])):
        for position in range(start, end):
            local = position - start
            current = tokens[position]
            recent = tokens[position - 1] if local else history[request][1]
            older = tokens[position - 2] if local >= 2 else history[request][local]
            if recent == geometry.eos:
                older = geometry.eos
            first = current * geometry.multipliers[0] ^ recent * geometry.multipliers[1]
            result.extend(
                (first ^ (older * geometry.multipliers[2] if head >= 8 else 0)) % size + offset
                for head, (size, offset) in enumerate(zip(geometry.sizes, geometry.offsets))
            )
    return result


@torch.inference_mode()
def main(rank):
    output = ROOT / f"ple-transport-rank{rank}.json"
    assert not output.exists()
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)
    library = ROOT / "ple-host-six-v1/libple_host_gather.so"
    binary = ROOT / "ple-selector-six-v2/native_ple_request.bin"
    manager = ServingPLE(
        dict(
            max_requests=6,
            table=str(ROOT / "ple-table.json"),
            library=str(library),
            library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            binary=str(binary),
            binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        )
    )
    cases = []
    for count in (1, 3, 6, 9, 12, 15, 18):
        for requests in (4, 6):
            owner = Lookup(manager, count, requests, device)
            for revision in range(3):
                tokens = [(71 + i * 23 + revision * 431) % owner.geometry.vocabulary for i in range(count)]
                histories = [[owner.geometry.eos, 113 + i + revision] for i in range(requests)]
                starts = [i * count // requests for i in range(requests + 1)]
                if revision:
                    tokens[revision % count] = owner.geometry.eos
                inputs = (
                    torch.tensor(tokens, dtype=torch.int64, device=device),
                    torch.tensor(starts, dtype=torch.int32, device=device),
                    torch.tensor(histories, dtype=torch.int64, device=device),
                )
                rows = expected_rows(tokens, histories, starts, owner.geometry)
                expected = b"".join(os.pread(manager.fd, 320, row * 320) for row in rows)
                actual = owner.fetch(*inputs)
                assert owner.rows.cpu().reshape(-1).tolist() == rows
                assert actual.cpu().contiguous().view(torch.uint8).numpy().tobytes() == expected
            cases.append(dict(tokens=count, requests=requests, revisions=3, exact=True))
    torch.npu.synchronize()
    for owner in reversed(manager.retained):
        checked(manager.lib.ple_host_close(owner.handle), "private owner close")
    os.close(manager.fd)
    report = dict(
        passed=True,
        rank=rank,
        cases=cases,
        owners_closed=len(manager.retained),
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        full_table=True,
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    main(parser.parse_args().rank)
