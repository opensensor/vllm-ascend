# SPDX-License-Identifier: Apache-2.0
"""Exact signed, strided and changed-replay gates for fused QSA metadata."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


def load(build):
    provenance = json.loads((build / "provenance.json").read_text())
    options = provenance["_build"]
    package = options["helper_package"]
    root = build / package
    for name, digest in provenance["_helpers"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("frozen QSA helper changed: " + name)
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package, root / "__init__.py", submodule_search_locations=[str(root)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    if Path(sys.modules[package].__file__).resolve() != (root / "__init__.py").resolve():
        raise ValueError("QSA package belongs to another build")
    binaries = (build / f"glm_reconstruction_bridge_v{options['version']}.so", build / "glm_qsa_metadata.bin")
    for binary in binaries:
        key = "reconstruction_bridge.cpp" if binary.suffix == ".so" else binary.name
        if hashlib.sha256(binary.read_bytes()).hexdigest() != provenance[key]["binary_sha256"]:
            raise ValueError("QSA binary differs from provenance")
    torch.ops.load_library(str(binaries[0]))
    helper = importlib.import_module(package + ".qsa_metadata")
    return helper.NativeQsaMetadata(build, options["namespace"]), provenance


def reference(ids, positions, table, rows, token_start, budget, split):
    groups = torch.div(ids[token_start : token_start + rows, : budget * 4 : 4], 4, rounding_mode="floor").int()
    lengths = positions[:rows].int() + 1
    pools = torch.div(lengths, 4, rounding_mode="floor")
    dense = lengths <= budget * 4
    counts = torch.where(dense, lengths, pools.clamp(max=budget))
    starts = pools * 4
    tails = torch.where(dense, -1, lengths - starts)
    logical = torch.div(table[:, ::split], split, rounding_mode="floor").int().contiguous()
    return (groups, counts, starts, tails), logical


def case(native, rows=2, requests=4, budget=512, position_dtype=torch.int64, split=20, token_start=3):
    helper = importlib.import_module(type(native).__module__)
    prefix, suffix, gap = 8, 24, 17
    ids_width = budget * 4 + 128
    ids_stride = ids_width * 2 + gap
    table_width = 9720 if budget == 512 else 41
    table_stride = table_width * 2 + gap
    generator = torch.Generator().manual_seed(941 + rows + requests + budget)
    cpu_backings = [
        torch.randint(-10000, 10000, (length,), generator=generator, dtype=torch.int32)
        for length in (prefix + (rows + token_start) * ids_stride + suffix, prefix + requests * table_stride + suffix)
    ]
    cpu_positions = torch.empty(prefix + rows * 2 + suffix, dtype=position_dtype)
    cpu_positions.fill_(999)
    backings = [tensor.to(native.device) for tensor in (*cpu_backings, cpu_positions)]
    ids = backings[0].as_strided((rows + token_start, ids_width), (ids_stride, 2), prefix)
    table = backings[1].as_strided((requests, table_width), (table_stride, 2), prefix)
    positions = backings[2].as_strided((rows,), (2,), prefix)
    cpu_ids = cpu_backings[0].as_strided(ids.shape, ids.stride(), prefix)
    cpu_table = cpu_backings[1].as_strided(table.shape, table.stride(), prefix)
    cpu_pos = cpu_positions.as_strided(positions.shape, positions.stride(), prefix)
    cpu_pos.copy_(
        torch.tensor([-1, 0, budget * 4 - 1, budget * 4, 311039, 2049] * ((rows + 5) // 6), dtype=position_dtype)[:rows]
    )
    cpu_ids.reshape(-1)[:1] = -1  # Random signed IDs already cover negative floor semantics.
    backings[2].copy_(cpu_positions)
    layout = helper.geometry(ids, positions, table, rows, token_start, budget, split * 32)
    native.prepare([layout])
    kwargs = dict(rows=rows, token_start=token_start, budget=budget, block_size=split * 32)
    native.plan(ids, positions, table, **kwargs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        plan, logical = native.plan(ids, positions, table, **kwargs)
    for replay in range(2):
        if replay:
            for backing in cpu_backings:
                backing.random_(-10000, 10000, generator=generator)
            cpu_pos.copy_(
                torch.tensor([2047, 2048, 2049, 311038, 0, -1] * ((rows + 5) // 6), dtype=position_dtype)[:rows]
            )
            for device, cpu in zip(backings, (*cpu_backings, cpu_positions)):
                device.copy_(cpu)
        expected_plan, expected_table = reference(cpu_ids, cpu_pos, cpu_table, rows, token_start, budget, split)
        graph.replay()
        torch.npu.synchronize()
        assert all(torch.equal(actual.cpu(), expected) for actual, expected in zip(plan, expected_plan))
        assert torch.equal(logical.cpu(), expected_table)
        for actual, expected in zip(backings, (*cpu_backings, cpu_positions)):
            assert torch.equal(actual.cpu(), expected), "metadata kernel modified its inputs or guards"
        # All owned DMA padding is initialized to zero, including each of
        # three tiny metadata vectors; odd table rows share one linear span.
        for tensor in (plan[0], logical):
            flat = tensor.as_strided((helper.aligned(tensor.numel()),), (1,))
            assert not torch.any(flat[tensor.numel() :].cpu())
        for tensor in plan[1:]:
            flat = tensor.as_strided((helper.aligned(rows),), (1,))
            assert not torch.any(flat[rows:].cpu())
    return dict(
        passed=True,
        changed_replay=True,
        signed=True,
        guards_checked=True,
        owned_padding_checked=True,
        rows=rows,
        requests=requests,
        budget=budget,
        position_bytes=position_dtype.itemsize,
        split=split,
        token_start=token_start,
        strided_inputs=True,
    )


def run(build, output):
    native, provenance = load(build)
    records = [
        case(native, rows=rows, requests=requests, budget=budget, position_dtype=dtype, split=split)
        for rows in (2, 8, 640)
        for requests in (1, 4)
        for budget in (4, 512)
        for dtype in (torch.int32, torch.int64)
        for split in (1, 20)
    ]
    paths = (build / f"glm_reconstruction_bridge_v{provenance['_build']['version']}.so", build / "glm_qsa_metadata.bin")
    report = dict(
        complete=True,
        quality_evaluated=False,
        build_options=provenance["_build"],
        records=records,
        binaries={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(complete=True, cases=len(records))), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.build_dir, args.output)


if __name__ == "__main__":
    main()
