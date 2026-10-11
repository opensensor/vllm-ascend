# SPDX-License-Identifier: Apache-2.0
"""Create an isolated sidecard deployment using a qualified native GLM server."""

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
from pathlib import Path

from .device_backend import CUDA_NIBBLE_PLANES, CUDA_ROW_MAJOR, CUDA_WORD_PLANES
from .native_expert_probe import digest_file
from .service_adapter import CHUNK_ROWS, MAX_PACKET_TOKENS, service_placement
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
    packet_tokens=CHUNK_ROWS,
    weight_layout=CUDA_ROW_MAJOR,
    bank_cache=None,
    reply_fp16=False,
    partial_reduce=False,
    stage_only=False,
    one_card=False,
    cuda_gate=None,
    npu_devices=None,
):
    if type(max_sessions) is not int or max_sessions < 1:
        raise ValueError("max_sessions must be a positive integer")
    if type(packet_tokens) is not int or not 1 <= packet_tokens <= MAX_PACKET_TOKENS:
        raise ValueError("packet_tokens must be within 1–640")
    if not tensor_transport and packet_tokens != CHUNK_ROWS:
        raise ValueError("large service packets require tensor transport")
    if weight_layout not in (CUDA_ROW_MAJOR, CUDA_WORD_PLANES, CUDA_NIBBLE_PLANES):
        raise ValueError("unsupported CUDA weight layout")
    if type(one_card) is not bool or (
        one_card
        and (
            weight_layout != CUDA_NIBBLE_PLANES or not tensor_transport or bank_cache is not None or not partial_reduce
        )
    ):
        raise ValueError("one-card service requires packed tensor transport without an INT8 bank cache")
    if one_card and cuda_gate is None:
        raise ValueError("one-card service requires a real-weight packed CUDA gate receipt")
    if npu_devices is not None and not one_card:
        raise ValueError("physical NPU pair selection requires one-card mode")
    if type(reply_fp16) is not bool or (reply_fp16 and not tensor_transport):
        raise ValueError("FP16 replies require tensor transport")
    if type(partial_reduce) is not bool or (partial_reduce and (reply_fp16 or not tensor_transport)):
        raise ValueError("partial combination requires tensor transport and FP32 replies")
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
    placement = service_placement(one_card=one_card)
    identity = WireIdentity.for_placement(placement, digest_file(checkpoint / "native-layout.json"))
    packed_gate = None
    if one_card:
        from .one_card import validate_device_pair, validate_packed_gate

        npu_devices = validate_device_pair((0, 1) if npu_devices is None else npu_devices)
        manifest = json.loads((checkpoint / "native-layout.json").read_bytes())
        packed_gate = validate_packed_gate(cuda_gate, digest_file(library), manifest["sidecard_source_manifest_sha256"])
    output.mkdir(mode=0o700)
    source = output / "source"
    shutil.copytree(qualified_source, source, ignore=shutil.ignore_patterns("__pycache__", "kernel_meta"))
    tools = source / "tools" / "sidecard"
    if tools.exists():
        shutil.rmtree(tools)
    shutil.copytree(Path(__file__).parent, tools, ignore=shutil.ignore_patterns("__pycache__"))
    if one_card:
        repository = Path(__file__).resolve().parents[2]
        for relative in ("tools/glm_perf/deployment_profiles.py", "vllm_ascend/models/glm5next_w2/model.py"):
            shutil.copy2(repository / relative, source / relative)
        runtime["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, npu_devices))
        command = list(previous["command"])
        command[command.index("serve") + 1] = str(checkpoint)
        command[command.index("--tensor-parallel-size") + 1] = "2"
        command[command.index("--model-loader-extra-config") + 1] = json.dumps(
            {"router_owner_rank": 1, "router_exchange_budget_bytes": 393216, "router_overlap_prefill": True}
        )
        previous["command"] = command
    backend = dict(
        kind="cuda_resident",
        checkpoint_root=str(checkpoint),
        library=str(library),
        library_sha256=digest_file(library),
        cuda_runtime="/usr/local/cuda/lib64/libcudart.so",
        cuda_memory_budget_bytes=CUDA_BUDGET_BYTES,
        weight_layout=weight_layout,
        reply_fp16=reply_fp16,
        partial_reduce=partial_reduce,
    )
    if bank_cache is not None:
        from .cuda_bank_cache import MANIFEST_NAME, CachedCudaBanks
        from .device_backend import owned_keys

        bank_cache = Path(bank_cache).resolve()
        cache_sha = digest_file(bank_cache / MANIFEST_NAME)
        CachedCudaBanks(
            bank_cache, cache_sha, identity.checkpoint_sha256, owned_keys(placement, "cuda:0"), weight_layout
        )
        backend.update(cache_root=str(bank_cache), cache_manifest_sha256=cache_sha)
    worker = dict(
        owner="cuda:0",
        placement=placement.to_dict(),
        identity=dataclasses.asdict(identity),
        limits=dataclasses.asdict(PacketLimits(4096, packet_tokens, packet_tokens * 8, 288, 8)),
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
        one_card=one_card,
        tensor_transport=tensor_transport,
        packet_tokens=packet_tokens,
        reply_fp16=reply_fp16,
        partial_reduce=partial_reduce,
    )
    if packed_gate is not None:
        config["packed_cuda_gate"] = packed_gate
        config["npu_devices"] = list(npu_devices)
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
    if stage_only:
        result = dict(command=command, cwd=str(source), staged=True, worker_config_sha256=digest_file(worker_path))
        (output / "cuda-staged.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
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
    parser.add_argument("--packet-tokens", type=int, default=CHUNK_ROWS)
    parser.add_argument(
        "--weight-layout", choices=(CUDA_ROW_MAJOR, CUDA_WORD_PLANES, CUDA_NIBBLE_PLANES), default=CUDA_ROW_MAJOR
    )
    parser.add_argument("--one-card", action="store_true")
    parser.add_argument("--cuda-gate", type=Path)
    parser.add_argument("--npu-devices", choices=("0,1", "2,3"))
    parser.add_argument("--bank-cache", type=Path)
    parser.add_argument("--reply-fp16", action="store_true")
    parser.add_argument("--partial-reduce", action="store_true")
    parser.add_argument("--stage-only", action="store_true")
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
                packet_tokens=args.packet_tokens,
                weight_layout=args.weight_layout,
                bank_cache=args.bank_cache,
                reply_fp16=args.reply_fp16,
                partial_reduce=args.partial_reduce,
                stage_only=args.stage_only,
                one_card=args.one_card,
                cuda_gate=args.cuda_gate,
                npu_devices=tuple(map(int, args.npu_devices.split(","))) if args.npu_devices else None,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
