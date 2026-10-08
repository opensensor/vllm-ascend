# SPDX-License-Identifier: Apache-2.0
"""Preflight and paired W2/W3/W4 grouped-kernel checks.

``preflight`` and ``compare`` never open an NPU. ``run`` is the only command
that does; run control and candidate in separate processes with one isolated
``ASCEND_CUSTOM_OPP_PATH`` each.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch

GROUPED_OPERATOR = "npu_w2_grouped_blocked_dequant_matmul_310"
SINGLE_OPERATOR = "npu_w2_blocked_dequant_matmul_310"
GROUPED_OPP_DIR = "w2_grouped_blocked_dequant_matmul_v310"
NZ_OUTPUT_TILE = 16
NZ_INPUT_TILE = 256


@dataclass(frozen=True)
class Case:
    name: str
    bits: int
    n: int
    k: int
    rows: int
    group_ends: tuple[int, ...]


def cases() -> tuple[Case, ...]:
    small = tuple(
        Case(f"w{bits}_small_{pattern}", bits, 256, 256, 4, ends)
        for bits in (2, 3, 4)
        for pattern, ends in (
            ("first_empty", (0, 1, 1, 3)),
            ("middle_empty", (1, 1, 2, 3)),
        )
    )
    return (
        *small,
        Case("w3_glm_gate_wide", 3, 2048, 4096, 67, (33, 33, 66)),
        Case("w3_sixteen_active_experts", 3, 2048, 4096, 17, tuple(range(1, 17))),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def pack_codes(signed: torch.Tensor, bits: int, *, nz_packed: bool) -> torch.Tensor:
    """Pack signed codes independently of the model loader's fast repacker."""
    if signed.dtype != torch.int8 or signed.ndim != 3 or bits not in (2, 3, 4):
        raise ValueError("expected signed int8 [experts,N,K] W2/W3/W4 codes")
    experts, n, k = signed.shape
    if n % NZ_OUTPUT_TILE or k % NZ_INPUT_TILE:
        raise ValueError("N and K must fit 16x256 NZ tiles")
    fields = signed.to(torch.int32) & ((1 << bits) - 1)
    if nz_packed:
        fields = (
            fields.view(experts, n // NZ_OUTPUT_TILE, NZ_OUTPUT_TILE, k // NZ_INPUT_TILE, NZ_INPUT_TILE)
            .permute(0, 1, 3, 4, 2)
            .reshape(experts, n // NZ_OUTPUT_TILE, k // NZ_INPUT_TILE, NZ_INPUT_TILE * NZ_OUTPUT_TILE)
        )
    if bits == 3:
        # Canonical packing groups adjacent K codes. NZ packing is field-major
        # over one entire 16x256 tile, matching the device vector decoder.
        if nz_packed:
            groups = fields.reshape(*fields.shape[:-1], 8, NZ_OUTPUT_TILE * NZ_INPUT_TILE // 8)
            field_axis = -2
        else:
            groups = fields.reshape(*fields.shape[:-1], -1, 8)
            field_axis = -1
        words = torch.zeros_like(groups.select(field_axis, 0), dtype=torch.int32)
        for field in range(8):
            words |= groups.select(field_axis, field) << (3 * field)
        # NZ tiles store three 512-byte planes; canonical stores interleaved
        # three-byte groups. Their stack axes differ accordingly.
        packed = torch.stack(tuple((words >> (8 * byte)) & 255 for byte in range(3)), dim=field_axis).to(torch.uint8)
    else:
        codes_per_byte = 8 // bits
        if nz_packed:
            groups = fields.reshape(
                *fields.shape[:-1], codes_per_byte, NZ_OUTPUT_TILE * NZ_INPUT_TILE // codes_per_byte
            )
            field_axis = -2
        else:
            groups = fields.reshape(*fields.shape[:-1], -1, codes_per_byte)
            field_axis = -1
        packed = torch.zeros_like(groups.select(field_axis, 0), dtype=torch.uint8)
        for field in range(codes_per_byte):
            packed |= groups.select(field_axis, field).to(torch.uint8) << (bits * field)
    packed = packed.reshape(experts, n, k * bits // 8).contiguous()
    return packed.view(torch.int8) if nz_packed else packed


def describe_package(opp_root: Path, binding_lib: Path) -> dict:
    root = opp_root.resolve(strict=True)
    binding = binding_lib.resolve(strict=True)
    op_dir = root / "op_impl/ai_core/tbe/kernel/ascend310p" / GROUPED_OPP_DIR
    binaries: dict[str, dict[str, str]] = {}
    for binary in op_dir.glob("*.o"):
        metadata_path = binary.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        inputs = [entry for entry in metadata["supportInfo"]["inputs"] if entry["name"] == "codes"]
        if len(inputs) != 1:
            raise ValueError(f"{metadata_path}: expected one codes input")
        dtype = inputs[0]["dtype"]
        if dtype in ("uint8", "int8"):
            if dtype in binaries:
                raise ValueError(f"duplicate {dtype} grouped binary in {op_dir}")
            binaries[dtype] = {
                "path": str(binary),
                "sha256": sha256_file(binary),
                "metadata_sha256": sha256_file(metadata_path),
            }
    if set(binaries) != {"uint8", "int8"}:
        raise ValueError(f"expected canonical uint8 and NZ int8 grouped binaries, got {sorted(binaries)}")
    grouped_source = (
        root / "op_impl/ai_core/tbe/custom_transformer_impl/ascendc" / GROUPED_OPP_DIR / f"{GROUPED_OPP_DIR}.cpp"
    )
    standalone_dir = root / "op_impl/ai_core/tbe/kernel/ascend310p/w2_blocked_dequant_matmul_v310"
    standalone_binaries = {binary.name: sha256_file(binary) for binary in standalone_dir.glob("*.o")}
    if not standalone_binaries:
        raise ValueError(f"no standalone W2/W3/W4 binary in {standalone_dir}")
    torch.ops.load_library(str(binding))
    if not all(hasattr(torch.ops._C_ascend, name) for name in (GROUPED_OPERATOR, SINGLE_OPERATOR)):
        raise RuntimeError("binding library lacks the grouped or standalone 310P operator")
    return {
        "opp_root": str(root),
        "binding_lib": str(binding),
        "binding_sha256": sha256_file(binding),
        "grouped_source_sha256": sha256_file(grouped_source),
        "binaries": binaries,
        "standalone_binaries": standalone_binaries,
    }


def input_tensors(case: Case, *, nz_packed: bool) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20261004 + case.bits + case.rows)
    signed = torch.randint(
        -(1 << (case.bits - 1)),
        1 << (case.bits - 1),
        (len(case.group_ends), case.n, case.k),
        dtype=torch.int8,
        generator=generator,
    )
    codes = pack_codes(signed, case.bits, nz_packed=nz_packed)
    reference_codes = pack_codes(signed, case.bits, nz_packed=False) if nz_packed else codes
    scales = (torch.rand(len(case.group_ends), case.n // 32, case.k // 32, generator=generator) * 0.02 + 0.005).float()
    inputs = torch.randn(case.rows, case.k, generator=generator).half()
    ends = torch.tensor(case.group_ends, dtype=torch.int64)
    return inputs, codes, reference_codes, scales, ends


def run_case(case: Case, *, nz_packed: bool, warmup: int, repeats: int) -> dict:
    inputs, codes, reference_codes, scales, ends = input_tensors(case, nz_packed=nz_packed)
    input_hashes = {
        name: sha256_tensor(value)
        for name, value in zip(
            ("inputs", "codes", "reference_codes", "scales", "ends"),
            (inputs, codes, reference_codes, scales, ends),
        )
    }
    inputs, codes, reference_codes, scales, ends = (
        value.npu() for value in (inputs, codes, reference_codes, scales, ends)
    )
    grouped = getattr(torch.ops._C_ascend, GROUPED_OPERATOR)
    standalone = getattr(torch.ops._C_ascend, SINGLE_OPERATOR)
    for _ in range(warmup):
        grouped(inputs, codes, scales, ends)
    torch.npu.synchronize()
    samples_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        actual = grouped(inputs, codes, scales, ends)
        torch.npu.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1000)
    actual = actual.cpu()
    if not torch.isfinite(actual).all():
        raise RuntimeError(f"{case.name}: non-finite grouped output")
    start_row = 0
    for expert, end_row in enumerate(case.group_ends):
        # The standalone OpDef only admits uint8 canonical codes; for NZ
        # layouts use the same signed weights packed in canonical order.
        if end_row > start_row:
            expected = standalone(inputs[start_row:end_row], reference_codes[expert], scales[expert]).cpu()
            if not torch.equal(actual[start_row:end_row].view(torch.uint8), expected.view(torch.uint8)):
                raise RuntimeError(f"{case.name}: expert {expert} differs bitwise from standalone")
        start_row = end_row
    if torch.count_nonzero(actual[start_row:]):
        raise RuntimeError(f"{case.name}: peer-owned rows are not zero")
    return {
        "case": case.name,
        "bits": case.bits,
        "layout": "nzpacked" if nz_packed else "canonical",
        "shape": [case.rows, case.n, case.k],
        "group_ends": list(case.group_ends),
        "inputs_sha256": input_hashes,
        "output_sha256": sha256_tensor(actual),
        "median_synchronized_call_ms": statistics.median(samples_ms),
        "samples_ms": samples_ms,
    }


def preflight(args: argparse.Namespace) -> None:
    package = describe_package(args.opp_root, args.binding_lib)
    if package["grouped_source_sha256"] != args.expected_source_sha:
        raise ValueError("packaged grouped source does not match --expected-source-sha")
    for case in cases():
        for nz_packed in (False, True):
            inputs, codes, reference_codes, scales, ends = input_tensors(case, nz_packed=nz_packed)
            assert inputs.shape == (case.rows, case.k)
            assert codes.shape == (len(case.group_ends), case.n, case.k * case.bits // 8)
            assert reference_codes.shape == codes.shape
            assert reference_codes.dtype == torch.uint8
            assert scales.shape == (len(case.group_ends), case.n // 32, case.k // 32)
            assert ends[-1] < case.rows
    print(json.dumps({"preflight": "passed", "package": package}, indent=2))


def run(args: argparse.Namespace) -> None:
    import torch_npu

    if args.output.exists():
        raise FileExistsError(args.output)
    if args.warmup < 1 or args.repeats < 2:
        raise ValueError("warmup must be >= 1 and repeats >= 2")
    package = describe_package(args.opp_root, args.binding_lib)
    if package["grouped_source_sha256"] != args.expected_source_sha:
        raise ValueError("packaged grouped source does not match --expected-source-sha")
    configured = [str(Path(path).resolve()) for path in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if path]
    if configured != [package["opp_root"]]:
        raise ValueError("ASCEND_CUSTOM_OPP_PATH must contain only the selected --opp-root")
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(args.device):
        raise RuntimeError("run requires Ascend 310P")
    torch.npu.set_device(args.device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    records = [
        run_case(case, nz_packed=nz_packed, warmup=args.warmup, repeats=args.repeats)
        for case in cases()
        for nz_packed in (False, True)
    ]
    for info in package["binaries"].values():
        if sha256_file(Path(info["path"])) != info["sha256"]:
            raise RuntimeError("OPP binary changed during measurement")
    standalone_dir = Path(package["opp_root"]) / "op_impl/ai_core/tbe/kernel/ascend310p/w2_blocked_dequant_matmul_v310"
    if {binary.name: sha256_file(binary) for binary in standalone_dir.glob("*.o")} != package["standalone_binaries"]:
        raise RuntimeError("standalone OPP binary changed during measurement")
    payload = {"schema_version": 1, "package": package, "device": args.device, "records": records}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"passed_cases": len(records), "output": str(args.output)}))


def compare(args: argparse.Namespace) -> None:
    before = json.loads(args.control.read_text())
    after = json.loads(args.candidate.read_text())
    if before["schema_version"] != 1 or after["schema_version"] != 1:
        raise ValueError("measurement schema mismatch")
    if before["package"]["binding_sha256"] != after["package"]["binding_sha256"]:
        raise ValueError("control and candidate bindings differ")
    if before["package"]["standalone_binaries"] != after["package"]["standalone_binaries"]:
        raise ValueError("control and candidate standalone binaries differ")
    if before["package"]["grouped_source_sha256"] == after["package"]["grouped_source_sha256"]:
        raise ValueError("control and candidate grouped source hashes are identical")
    if all(
        before["package"]["binaries"][dtype]["sha256"] == after["package"]["binaries"][dtype]["sha256"]
        for dtype in ("uint8", "int8")
    ):
        raise ValueError("control and candidate binaries are identical")
    reference = {record["case"] + "/" + record["layout"]: record for record in before["records"]}
    candidate = {record["case"] + "/" + record["layout"]: record for record in after["records"]}
    if reference.keys() != candidate.keys():
        raise ValueError("case sets differ")
    results = []
    for name in sorted(reference):
        left, right = reference[name], candidate[name]
        if any(left[field] != right[field] for field in ("bits", "layout", "shape", "group_ends", "inputs_sha256")):
            raise ValueError(f"{name}: case inputs differ")
        if left["output_sha256"] != right["output_sha256"]:
            raise ValueError(f"{name}: grouped outputs differ bitwise")
        results.append(
            {
                "case": name,
                "control_ms": left["median_synchronized_call_ms"],
                "candidate_ms": right["median_synchronized_call_ms"],
            }
        )
    print(
        json.dumps({"bitwise_parity": "passed", "cases": len(results), "synchronized_call_medians": results}, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("preflight", preflight), ("run", run)):
        command = commands.add_parser(name)
        command.add_argument("--opp-root", type=Path, required=True)
        command.add_argument("--binding-lib", type=Path, required=True)
        command.add_argument("--expected-source-sha", required=True)
        if name == "run":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--device", type=int, default=0)
            command.add_argument("--warmup", type=int, default=1)
            command.add_argument("--repeats", type=int, default=3)
        command.set_defaults(func=function)
    compare_command = commands.add_parser("compare")
    compare_command.add_argument("--control", type=Path, required=True)
    compare_command.add_argument("--candidate", type=Path, required=True)
    compare_command.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
