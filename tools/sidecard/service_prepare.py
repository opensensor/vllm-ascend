# SPDX-License-Identifier: Apache-2.0
"""Create an isolated sidecard deployment using a qualified native GLM server."""

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
from pathlib import Path

from .native_expert_probe import digest_file
from .service_adapter import service_placement
from .wire import PacketLimits, WireIdentity

GIB = 1024**3
CUDA_BUDGET_BYTES = 82 * GIB


def prepare(
    qualified_source,
    previous_process,
    runtime_environment,
    checkpoint,
    library,
    cuda_python,
    output,
    *,
    max_sessions=1,
    tensor_transport=False,
):
    if type(max_sessions) is not int or max_sessions < 1:
        raise ValueError("max_sessions must be a positive integer")
    output = Path(output).resolve()
    qualified_source = Path(qualified_source).resolve()
    if output.is_relative_to(qualified_source):
        raise ValueError("deployment output must be outside the qualified source tree")
    if output.exists():
        raise ValueError("sidecard deployment requires a fresh output directory")
    previous = json.loads(Path(previous_process).read_bytes())
    runtime = json.loads(Path(runtime_environment).read_bytes())
    checkpoint, library = (Path(path).resolve() for path in (checkpoint, library))
    # Keep the venv interpreter spelling: resolving its symlink would select
    # the system interpreter and lose that environment's installed packages.
    cuda_python = Path(cuda_python).absolute()
    if not checkpoint.is_dir() or not library.is_file() or not cuda_python.is_file():
        raise ValueError("checkpoint, compiled CUDA library and CUDA interpreter must exist")
    placement = service_placement()
    identity = WireIdentity.for_placement(placement, digest_file(checkpoint / "native-layout.json"))
    output.mkdir(mode=0o700)
    source = output / "source"
    shutil.copytree(qualified_source, source, ignore=shutil.ignore_patterns("__pycache__", "kernel_meta"))
    tools = source / "tools" / "sidecard"
    if tools.exists():
        shutil.rmtree(tools)
    shutil.copytree(Path(__file__).parent, tools, ignore=shutil.ignore_patterns("__pycache__"))
    backend = dict(
        kind="cuda_resident",
        checkpoint_root=str(checkpoint),
        library=str(library),
        library_sha256=digest_file(library),
        cuda_runtime="/usr/local/cuda/lib64/libcudart.so",
        cuda_memory_budget_bytes=CUDA_BUDGET_BYTES,
    )
    worker = dict(
        owner="cuda:0",
        placement=placement.to_dict(),
        identity=dataclasses.asdict(identity),
        limits=dataclasses.asdict(PacketLimits(4096, 32, 256, 288, 8)),
        backend=backend,
    )
    worker_path = output / "cuda-worker.json"
    worker_path.write_text(json.dumps(worker, indent=2) + "\n")
    config = dict(
        checkpoint_root=str(checkpoint),
        checkpoint_sha256=identity.checkpoint_sha256,
        worker_config_sha256=digest_file(worker_path),
        worker_ready_path=str(output / "cuda-ready.json"),
        cuda_socket=str(output / "cuda.sock"),
        report_path=str(output / "model-boundary-metrics.json"),
        tensor_transport=tensor_transport,
    )
    (output / "service.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "runtime-environment.json").write_text(json.dumps(runtime, indent=2) + "\n")
    (output / "previous-service.json").write_text(json.dumps(previous, indent=2) + "\n")
    extension = (
        "from pathlib import Path\n"
        "from card_worker_extension import CardWorkerExtension\n"
        "from tools.sidecard.service_adapter import SidecardStatusMixin, install_loader_hook\n"
        'install_loader_hook(Path(__file__).resolve().parent.parent / "service.json")\n'
        "class SidecardWorkerExtension(SidecardStatusMixin, CardWorkerExtension):\n"
        "    pass\n"
    )
    (source / "sidecard_worker.py").write_text(extension)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source)
    command = [
        str(cuda_python),
        "-m",
        "tools.sidecard.resident_daemon",
        "--config",
        str(worker_path),
        "--socket",
        str(output / "cuda.sock"),
        "--ready",
        str(output / "cuda-ready.json"),
        "--max-sessions",
        str(max_sessions),
    ]
    if tensor_transport:
        command.append("--tensor-transport")
    with (output / "cuda-worker.log").open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=source,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    result = dict(pid=process.pid, command=command, cwd=str(source))
    (output / "cuda-process.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-sessions", type=int, default=1)
    parser.add_argument("--tensor-transport", action="store_true")
    for flag in (
        "qualified-source",
        "previous-process",
        "runtime-environment",
        "checkpoint",
        "library",
        "cuda-python",
        "output",
    ):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.qualified_source,
                args.previous_process,
                args.runtime_environment,
                args.checkpoint,
                args.library,
                args.cuda_python,
                args.output,
                max_sessions=args.max_sessions,
                tensor_transport=args.tensor_transport,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
