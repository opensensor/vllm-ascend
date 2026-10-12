# SPDX-License-Identifier: Apache-2.0
"""Freeze a reproducible source snapshot and launch only assigned stage actors."""

import argparse
import hashlib
import json
import os
import shutil
import signal
from pathlib import Path

DEFAULT_MAX_CONTEXT = 65536
GLM_DEPENDENCIES = ("__init__.py", "deployment_profiles.py", "fused_moe_profile.py", "glm_int4.py")


def prepare(output, specification, source):
    """Create a new snapshot; never overwrite a previous run or copy its secrets."""
    output, source = Path(output).resolve(), Path(source).resolve()
    if output == source or source in output.parents:
        raise ValueError("runtime output must be outside the source tree")
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "source" / "tools"
    (destination / "sidecard").mkdir(parents=True)
    (destination / "glm_perf").mkdir()
    files = []
    for original in sorted((source / "tools" / "sidecard").glob("*.py")):
        files.append(original)
    files.extend(source / "tools" / "glm_perf" / name for name in GLM_DEPENDENCIES)
    hashes = {}
    for original in files:
        relative = original.relative_to(source)
        shutil.copyfile(original, output / "source" / relative)
        hashes[str(relative)] = hashlib.sha256(original.read_bytes()).hexdigest()
    (output / "source-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    specification = dict(specification, output=str(output))
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
        str(args.cuda_python.resolve()),
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
