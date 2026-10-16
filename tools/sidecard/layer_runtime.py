# SPDX-License-Identifier: Apache-2.0
"""Freeze a reproducible source snapshot and launch only assigned stage actors."""

import argparse
import hashlib
import json
import os
import shutil
import signal
from pathlib import Path

from .cold_placement import load as load_placement
from .execution_options import validate_options

DEFAULT_MAX_CONTEXT = 65536
GLM_DEPENDENCIES = ("__init__.py", "deployment_profiles.py", "fused_moe_profile.py", "glm_int4.py")
ACCEPTANCE_FIXTURE = Path("tests/e2e/nightly/sidecard/fixtures/acceptance-quality.json")
UNIX_SOCKET_PATH_BYTES = 108


def native_inputs(reference, name):
    """Retain the authenticated native sources alongside each runtime freeze."""
    original = reference.root / name
    raw = original.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference.manifest_sha256:
        raise ValueError("candidate receipt changed during snapshot")
    yield Path(name), raw, original.stat().st_mode & 0o777
    manifest = json.loads(raw)
    if name == "execution-build.json":
        root = reference.root / "source"
        records = manifest["source_files"]
        prefix = "execution-source"
        companions = {
            "source-manifest.json": manifest["source_manifest_sha256"],
            "abi-manifest.json": manifest["abi_manifest_sha256"],
        }
    else:
        root = reference.root
        records = manifest["source_sha256"]
        prefix = "multi-route-source"
        companions = {"candidate.patch": manifest["patch_sha256"], "provenance.json": manifest["provenance_sha256"]}
    for inventory, base, destination in (
        (records, root, Path(prefix)),
        (companions, reference.root, Path(prefix) / "receipts"),
    ):
        for relative, checksum in inventory.items():
            source = base / relative
            if Path(relative).is_absolute() or not source.resolve().is_relative_to(base.resolve()):
                raise ValueError("candidate source escapes its authenticated build")
            content = source.read_bytes()
            if hashlib.sha256(content).hexdigest() != checksum:
                raise ValueError("candidate source changed during snapshot")
            yield destination / relative, content, source.stat().st_mode & 0o777


def prepare(output, specification, source):
    """Create a new snapshot; never overwrite a previous run or copy its secrets."""
    output, source = Path(output).resolve(), Path(source).resolve()
    if output == source or source in output.parents:
        raise ValueError("runtime output must be outside the source tree")
    execution = validate_options(specification)
    suffix = None
    if specification.get("stage_transport") == "direct-dma":
        suffix = "stage-dma-control/rank0.sock"
    elif specification.get("expert_transport") == "direct-dma":
        suffix = "dma-control/rank0.sock"
    if suffix is not None and len(os.fsencode(output / suffix)) >= UNIX_SOCKET_PATH_BYTES:
        raise ValueError("runtime output path is too long for its Unix control socket")
    placement = None
    if specification.get("architecture") == "cuda-backbone-ascend-experts":
        load_placement(specification)
        placement = Path(specification["expert_placement"]).read_bytes()
        if hashlib.sha256(placement).hexdigest() != specification["expert_placement_sha256"]:
            raise ValueError("expert ownership changed during snapshot")
    files = []
    for original in sorted((source / "tools" / "sidecard").rglob("*")):
        if original.suffix in (".py", ".cu"):
            files.append(original)
    files.extend(source / "tools" / "glm_perf" / name for name in GLM_DEPENDENCIES)
    fixture = source / ACCEPTANCE_FIXTURE
    if fixture.is_file():
        files.append(fixture)
    elif (source / "tools/sidecard/execution_acceptance.py").exists():
        raise ValueError("source snapshot is missing the public acceptance fixture")
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "source" / "tools"
    (destination / "sidecard").mkdir(parents=True)
    (destination / "glm_perf").mkdir()
    hashes = {}
    for original in files:
        relative = original.relative_to(source)
        (output / "source" / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, output / "source" / relative)
        hashes[str(relative)] = hashlib.sha256(original.read_bytes()).hexdigest()
    for reference, name in (
        (execution.native_build, "execution-build.json"),
        (execution.multi_route, "multi-route-build.json"),
    ):
        if reference is not None:
            for relative_native, raw, mode in native_inputs(reference, name):
                relative = Path("native-inputs") / relative_native
                target = output / "source" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                target.chmod(mode)
                hashes[str(relative)] = hashlib.sha256(raw).hexdigest()
    (output / "source-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    specification = dict(specification, output=str(output))
    if placement is not None:
        (output / "placement.json").write_bytes(placement)
        specification["expert_placement"] = str(output / "placement.json")
    (output / "runtime.json").write_text(json.dumps(specification, indent=2) + "\n")
    return output / "runtime.json"


def stop(output):
    """Stop the exact recorded parent; its cleanup owns the child processes."""
    output = Path(output).resolve()
    receipt = json.loads((output / "process.json").read_bytes())
    pid = receipt["pid"]
    if type(pid) is not int or pid <= 1:
        raise ValueError("invalid recorded coordinator PID")
    arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    if b"tools.sidecard.layer_server" not in arguments or os.fsencode(output / "runtime.json") not in arguments:
        raise ValueError("recorded PID no longer belongs to this runtime")
    os.kill(pid, signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--specification", type=Path)
    parser.add_argument("--cuda-python", type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    if args.stop:
        stop(args.output)
        return
    if args.specification is None or args.cuda_python is None:
        parser.error("launch requires --specification and --cuda-python")
    specification = json.loads(args.specification.read_bytes())
    specification.setdefault("max_context", DEFAULT_MAX_CONTEXT)
    configuration = prepare(args.output, specification, args.source)
    if args.prepare_only:
        print(configuration)
        return
    command = [
        # Resolving a venv's python symlink selects the base interpreter and
        # loses the CUDA-only package environment.
        str(args.cuda_python.absolute()),
        "-m",
        "tools.sidecard.layer_server",
        "--config",
        str(configuration),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    (configuration.parent / "process.json").write_text(
        json.dumps({"pid": os.getpid(), "command": command}, indent=2) + "\n"
    )
    environment = dict(os.environ, PYTHONPATH=str(configuration.parent / "source"))
    if specification.get("cuda_environment"):
        environment.update(json.loads(Path(specification["cuda_environment"]).read_bytes()))
    os.chdir(configuration.parent / "source")
    os.execve(command[0], command, environment)


if __name__ == "__main__":
    main()
