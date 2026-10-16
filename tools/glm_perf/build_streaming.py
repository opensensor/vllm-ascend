# SPDX-License-Identifier: Apache-2.0
"""Build immutable streaming candidates and verify probe/live admission offline."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path
from types import MappingProxyType

from tools.glm_perf.streaming_memory import memory_contract
from tools.glm_perf.streaming_protocol import (
    BASELINE_VERSION,
    DECODE_VERSION,
    SCALE_LAYOUT,
    canonical_digest,
    evidence_schema,
    file_digest,
    validate_evidence,
    verify_reference,
)

STREAMING_HEADERS = (
    "glm_streaming_contract.h",
    "glm_streaming_operands.h",
    "glm_streaming_products.h",
    "glm_streaming_epilogue.h",
)
RUNTIME_BINARIES = (
    "glm_w4a8_pack.bin",
    "glm_w4a8_matmul.bin",
    "glm_fused_gate_up.bin",
    "glm_fused_down.bin",
    "glm_fused_gate_up_w4.bin",
    "glm_fused_down_w4.bin",
    "glm_fused_route_input.bin",
    "glm_fused_pack.bin",
    "glm_fused_reduce.bin",
)
MEMORY_COMPONENTS = frozenset(
    ("old_resources", "new_resources", "reference_scratch", "candidate_scratch", "outputs", "verification")
)
DISPATCH = MappingProxyType(
    {
        "bulk_a4_gate": "streaming",
        "bulk_a4_down": "streaming",
        "sparse": "qualified_reference",
        "a8": "qualified_reference",
        "decode_version": DECODE_VERSION,
    }
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("missing or escaping frozen artifact")
    return path


def _verify_map(root: Path, hashes: dict) -> None:
    if not hashes or any(file_digest(_path(root, name)) != digest for name, digest in hashes.items()):
        raise ValueError("frozen sources or binaries changed")


def _contract(path: Path, *, header_path: Path | None = None) -> dict:
    contract = _read(path)
    if contract.get("sdk_verified") is not True or contract != memory_contract(
        contract.get("sdk_evidence"), header_path=header_path
    ):
        raise ValueError("stale or incompatible streaming memory contract")
    return contract


def _reference(path: Path) -> dict:
    reference = verify_reference(_read(path), path.parent)
    if (
        reference.get("version") != BASELINE_VERSION
        or reference.get("decode_version") != DECODE_VERSION
        or reference.get("checkpoint", {}).get("scale_layout") != SCALE_LAYOUT
    ):
        raise ValueError("reference is not the qualified v1021 FP16-scale/decode984 contract")
    return reference


def _compatible_options(options: dict, reference: dict, version: int) -> None:
    expected = dict(reference["build_options"])
    expected.update(
        version=version,
        namespace=f"glm_reconstruction_v{version}",
        helper_package=f"glm_reconstruction_v{version}_helpers",
        streaming_pipeline=True,
    )
    if options != expected:
        raise ValueError("streaming build options differ from frozen reference")


def candidate_identity(
    build_dir: Path, *, contract_path: Path | None = None, reference_path: Path | None = None
) -> dict:
    """Bind actual frozen sources/binaries; does not admit unqualified live use."""
    root = Path(build_dir).resolve()
    manifest = _read(root / "streaming-manifest.json")
    contract_file, reference_file = root / "contract.json", root / "reference" / "reference.json"
    contract, reference, provenance = (
        _contract(contract_file, header_path=root / "native_sources" / STREAMING_HEADERS[0]),
        _reference(reference_file),
        _read(root / "provenance.json"),
    )
    # Immutable admission uses the frozen header, rather than today's workspace.
    if contract.get("header_sha256") != file_digest(root / "native_sources" / STREAMING_HEADERS[0]):
        raise ValueError("frozen contract header does not match memory proof")
    if contract_path is not None and file_digest(Path(contract_path)) != file_digest(contract_file):
        raise ValueError("supplied contract differs from frozen proof")
    if (
        reference_path is not None
        and _reference(Path(reference_path)).get("reference_sha256") != reference["reference_sha256"]
    ):
        raise ValueError("supplied reference differs from frozen identity")
    if manifest.get("schema_version") != 1 or manifest.get("dispatch") != DISPATCH:
        raise ValueError("incomplete streaming manifest/dispatch")
    version = provenance["_build"]["version"]
    if type(version) is not int or version <= BASELINE_VERSION:
        raise ValueError("candidate version must be newer than reference")
    _compatible_options(provenance["_build"], reference, version)
    sources = dict(provenance.get("_streaming_sources", {}))
    for name, digest in provenance.get("_helpers", {}).items():
        sources[f"glm_reconstruction_v{version}_helpers/{name}"] = digest
    for name in STREAMING_HEADERS:
        if sources.get(f"native_sources/{name}") != provenance.get(name, {}).get("source_sha256"):
            raise ValueError("missing or inconsistent streaming helper provenance")
    _verify_map(root, sources)
    binaries = {path.name: file_digest(path) for path in (*root.glob("*.bin"), *root.glob("*.so"))}
    expected_names = {*RUNTIME_BINARIES, f"glm_reconstruction_bridge_v{version}.so"}
    if set(binaries) != expected_names or any((root / name).stat().st_size == 0 for name in binaries):
        raise ValueError("complete pipeline binaries are missing or empty")
    for name in binaries:
        record = provenance.get("reconstruction_bridge.cpp" if name.endswith(".so") else name)
        if record is None and name == "glm_w4a8_pack.bin":
            record = provenance.get("glm_w4a8_pack.cpp")
        if record is None and name == "glm_w4a8_matmul.bin":
            record = provenance.get("glm_lowbit_a8_pipeline.cpp")
        if record is None or record.get("binary_sha256") != binaries[name]:
            raise ValueError("binary is absent from provenance or changed")
    identity = {
        "contract_sha256": file_digest(contract_file),
        "reference_sha256": reference["reference_sha256"],
        "source_sha256": canonical_digest(sources),
        "binaries": binaries,
        "version": version,
        "namespace": f"glm_reconstruction_v{version}",
    }
    if manifest.get("identity") != identity or manifest.get("sources") != sources:
        raise ValueError("candidate identity does not match frozen manifest")
    return identity


def admit_candidate(build_dir: Path, *, native_evidence_path: Path | None = None, live: bool = False) -> dict:
    """Permit standalone probes after build proof; live admission needs native gates."""
    root = Path(build_dir).resolve()
    identity = candidate_identity(root)
    validate_evidence(_read(root / "build.json"), kind="build", bindings=identity, artifact_root=root)
    if live:
        if native_evidence_path is None:
            raise ValueError("live admission requires complete native gates")
        gates_file = Path(native_evidence_path).resolve()
        gates = validate_evidence(
            _read(gates_file), kind="native_gates", bindings=identity, artifact_root=gates_file.parent
        )
        admission = gates["result"].get("resident_memory_admission", {})
        if (
            admission.get("passed") is not True
            or type(admission.get("required_free_bytes")) is not int
            or admission["required_free_bytes"] <= 0
            or not isinstance(admission.get("includes"), list)
            or not MEMORY_COMPONENTS.issubset(admission["includes"])
        ):
            raise ValueError("native gates lack complete resident memory admission")
    return identity


def _reserve_version(root: Path, version: int) -> None:
    if type(version) is not int or version <= BASELINE_VERSION:
        raise ValueError("explicit unused candidate version must be newer than reference")
    if root.exists():
        raise FileExistsError("build directories are append-only")
    root.parent.mkdir(parents=True, exist_ok=True)
    for path in root.parent.glob("*/provenance.json"):
        if _read(path).get("_build", {}).get("version") == version:
            raise ValueError("candidate namespace already exists in sibling build")
    reservation = root.parent / f".glm-streaming-v{version}.reserve"
    with reservation.open("x") as handle:
        json.dump({"version": version, "build_dir": str(root)}, handle)
    # Keep the reservation on failure. Failed versions are never recycled.


def build(
    contract_path: Path,
    reference_path: Path,
    build_dir: Path,
    *,
    version: int,
    cann_root: Path,
    source_root: Path,
) -> Path:
    """Compile only. Never contact serving, load binaries or allocate NPU memory."""
    contract_path, reference_path = Path(contract_path).resolve(), Path(reference_path).resolve()
    contract, reference = _contract(contract_path), _reference(reference_path)
    contract_hash, reference_hash = file_digest(contract_path), file_digest(reference_path)
    for record in contract["sdk_evidence"].get("files", []):
        if file_digest(Path(record["path"])) != record["sha256"]:
            raise ValueError("installed SDK headers changed since memory proof")
    from tools.glm_perf import build_reconstruction  # Compiler dependencies loaded only for offline build.

    root = Path(build_dir).resolve()
    parameters = inspect.signature(build_reconstruction.build).parameters
    options = {
        name: value
        for name, value in reference["build_options"].items()
        if name in parameters and name not in ("version", "streaming_pipeline")
    }
    _reserve_version(root, version)
    try:
        build_reconstruction.build(
            root, Path(cann_root), Path(source_root), version=version, streaming_pipeline=True, **options
        )
        shutil.copy2(contract_path, root / "contract.json")
        reference_root = root / "reference"
        reference_root.mkdir()
        shutil.copy2(reference_path, reference_root / "reference.json")
        if (
            file_digest(root / "contract.json") != contract_hash
            or file_digest(reference_root / "reference.json") != reference_hash
        ):
            raise ValueError("contract/reference changed during compilation")
        for record in reference["frozen_files"]:
            target = reference_root / record["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(reference_path.parent / record["path"], target)
        provenance = _read(root / "provenance.json")
        _compatible_options(provenance["_build"], reference, version)
        sources = dict(provenance["_streaming_sources"])
        sources.update(
            {f"glm_reconstruction_v{version}_helpers/{name}": digest for name, digest in provenance["_helpers"].items()}
        )
        identity = {
            "contract_sha256": file_digest(root / "contract.json"),
            "reference_sha256": reference["reference_sha256"],
            "source_sha256": canonical_digest(sources),
            "binaries": {path.name: file_digest(path) for path in (*root.glob("*.bin"), *root.glob("*.so"))},
            "version": version,
            "namespace": f"glm_reconstruction_v{version}",
        }
        manifest = {"schema_version": 1, "identity": identity, "sources": sources, "dispatch": dict(DISPATCH)}
        (root / "streaming-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        candidate_identity(root)
        artifact_paths = ("streaming-manifest.json", "provenance.json", "contract.json", "reference/reference.json")
        report = {
            "schema_version": 1,
            "kind": "build",
            "state": "complete",
            "identity": identity,
            "dispatch": dict(DISPATCH),
            "coverage": evidence_schema()["x-required-coverage"]["build"],
            "artifacts": [{"path": name, "sha256": file_digest(root / name)} for name in artifact_paths],
            "errors": [],
            "result": {
                "memory_proof_passed": True,
                "complete_pipeline": True,
                "hardware_validated": False,
                "native_gates_required_for_live": True,
            },
        }
        validate_evidence(report, kind="build", bindings=identity, artifact_root=root)
        (root / "build.json").write_text(json.dumps(report, indent=2) + "\n")
        return root
    except Exception as error:
        failure = root.parent / f"build-failure-v{version}.json"
        failure.write_text(
            json.dumps({"state": "failed", "version": version, "build_dir": str(root), "error": repr(error)}, indent=2)
            + "\n"
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument(
        "--version", required=True, type=int, help="unused namespace, reserved append-only; check resident history"
    )
    parser.add_argument("--cann-root", type=Path, default=Path("/usr/local/Ascend/ascend-toolkit/latest"))
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(
        build(
            args.contract,
            args.reference,
            args.build_dir,
            version=args.version,
            cann_root=args.cann_root,
            source_root=args.source_root,
        )
    )


if __name__ == "__main__":
    main()
