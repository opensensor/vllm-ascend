# SPDX-License-Identifier: Apache-2.0
"""Bitwise selected-state-copy gates, including gapped backing and changed replay."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


def load_helper(build):
    provenance = json.loads((build / "provenance.json").read_text())
    options = provenance["_build"]
    package = options["helper_package"]
    root = build / package
    for name, digest in provenance["_helpers"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("frozen state-copy helper changed: " + name)
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package, root / "__init__.py", submodule_search_locations=[str(root)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    if Path(sys.modules[package].__file__).resolve() != (root / "__init__.py").resolve():
        raise ValueError("state-copy helper package belongs to another build")
    binaries = [
        build / f"glm_reconstruction_bridge_v{options['version']}.so",
        build / "glm_state_rows_gather.bin",
        build / "glm_state_rows_scatter.bin",
    ]
    for binary in binaries:
        key = "reconstruction_bridge.cpp" if binary.suffix == ".so" else binary.name
        if hashlib.sha256(binary.read_bytes()).hexdigest() != provenance[key]["binary_sha256"]:
            raise ValueError("state-copy binary differs from provenance")
    torch.ops.load_library(str(binaries[0]))
    helper = importlib.import_module(package + ".state_rows")
    return helper.NativeStateRows(build, options["namespace"]), provenance


def case(native, payload_shape=(2, 16, 16), gap=192, selected=4, index_dtype=torch.int64):
    rows, prefix, suffix = 7, 16, 16
    payload = 1
    for size in payload_shape:
        payload *= size
    stride = payload + gap
    generator = torch.Generator().manual_seed(922 + payload + gap + selected + index_dtype.itemsize)
    initial_bits = torch.randint(
        -32768, 32768, (prefix + rows * stride + suffix,), generator=generator, dtype=torch.int16
    )
    initial = initial_bits.view(torch.float16)
    backing = initial.to(native.device)
    cache = backing.as_strided(
        (rows, *payload_shape), (stride, payload_shape[1] * payload_shape[2], payload_shape[2], 1), prefix
    )
    native.prepare(cache)
    slots_cpu = torch.tensor([6, -1, 2, 0][:selected], dtype=index_dtype)
    writes_cpu = torch.tensor([0, 2, 4, 6][:selected], dtype=index_dtype)
    flags_cpu = torch.tensor([True, False, True, False][:selected])
    values_cpu = torch.randint(-32768, 32768, (selected, *payload_shape), generator=generator, dtype=torch.int16).view(
        torch.float16
    )
    slots, writes, flags, values = (t.to(native.device) for t in (slots_cpu, writes_cpu, flags_cpu, values_cpu))
    # Allocate/prepare outside capture; the graph subsequently reads current
    # slot ids, flags and bank payload, rather than capture-time content.
    native.gather(cache, slots, flags)
    native.scatter(cache, writes, values)
    torch.npu.synchronize()
    backing.copy_(initial)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = native.gather(cache, slots, flags)
        native.scatter(cache, writes, values)
    for replay in range(2):
        if replay:
            initial = torch.randint(-32768, 32768, initial.shape, generator=generator, dtype=torch.int16).view(
                torch.float16
            )
            slots_cpu = torch.tensor([1, 5, 3, -1][:selected], dtype=index_dtype)
            writes_cpu = torch.tensor([1, 3, 5, -1][:selected], dtype=index_dtype)
            flags_cpu = ~flags_cpu
            values_cpu = torch.randint(-32768, 32768, values_cpu.shape, generator=generator, dtype=torch.int16).view(
                torch.float16
            )
            for device, cpu in zip((slots, writes, flags, values), (slots_cpu, writes_cpu, flags_cpu, values_cpu)):
                device.copy_(cpu)
        backing.copy_(initial)
        expected_backing = initial.clone()
        expected_cache = expected_backing.as_strided(cache.shape, cache.stride(), prefix)
        expected = expected_cache[slots_cpu.long()].clone()
        expected[~flags_cpu] = 0
        expected_cache[writes_cpu.long()] = values_cpu
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(output.cpu().view(torch.int16), expected.view(torch.int16)), "gather changed FP16 bits"
        assert torch.equal(backing.cpu().view(torch.int16), expected_backing.view(torch.int16)), (
            "scatter changed unrelated payload or padding"
        )
    return {
        "passed": True,
        "changed_replay": True,
        "payload_shape": list(payload_shape),
        "gap": gap,
        "selected": selected,
        "index_bytes": index_dtype.itemsize,
        "storage_offset": prefix,
        "all_backing_bytes_checked": True,
        "negative_slots": True,
        "fresh_flags": True,
    }


def run(build, output):
    native, provenance = load_helper(build)
    records = [
        case(native, shape, gap, selected, dtype)
        for shape in ((2, 16, 16), (16, 128, 128))
        for gap in (0, 192)
        for selected in (1, 4)
        for dtype in (torch.int32, torch.int64)
    ]
    options = provenance["_build"]
    paths = [
        build / f"glm_reconstruction_bridge_v{options['version']}.so",
        build / "glm_state_rows_gather.bin",
        build / "glm_state_rows_scatter.bin",
    ]
    report = {
        "complete": True,
        "quality_evaluated": False,
        "build_options": options,
        "records": records,
        "binaries": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": True, "cases": len(records)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.build_dir, args.output)


if __name__ == "__main__":
    main()
