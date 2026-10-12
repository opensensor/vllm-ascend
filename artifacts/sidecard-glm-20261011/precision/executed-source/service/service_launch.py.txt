# SPDX-License-Identifier: Apache-2.0
"""Start the joined service from an existing qualified host deployment receipt.

CUDA preparation is a separate process, completed before replacing the NPU
server. This launcher sends SIGTERM only to the explicitly named old server
after checking its saved command. No device reset or SIGKILL is used.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from .native_expert_probe import digest_file
from .one_card_partition import SEED_PARTITION, one_card_partition
from .placement import validate_glm_config

SERVICE_NAME = "glm-sidecard"
SHUTDOWN_SECONDS = 180
MAX_QUALIFIED_PREFILL_TOKENS = 640


def argument(command, flag, value):
    result = list(command)
    if flag in result:
        result[result.index(flag) + 1] = value
    else:
        result.extend((flag, value))
    return result


def serving_command(
    previous,
    *,
    host="127.0.0.1",
    port=8001,
    max_model_len=None,
    kv_cache_memory_bytes=None,
    capacity_first=False,
    prefill_tokens=128,
    prefix_caching=None,
    max_num_seqs=None,
    fused_sinkhorn=None,
):
    if type(prefill_tokens) is not int or not 1 <= prefill_tokens <= MAX_QUALIFIED_PREFILL_TOKENS:
        raise ValueError("prefill_tokens must be within the qualified 1–640 token budget")
    if max_num_seqs is not None and (type(max_num_seqs) is not int or max_num_seqs < 1):
        raise ValueError("max_num_seqs must be positive")
    if capacity_first and prefix_caching is True:
        raise ValueError("capacity-first cannot retain prefix snapshots")
    if max_model_len is not None and (type(max_model_len) is not int or (max_model_len <= 0 and max_model_len != -1)):
        raise ValueError("max_model_len must be positive or -1 for automatic capacity fitting")
    if kv_cache_memory_bytes is not None and (type(kv_cache_memory_bytes) is not int or kv_cache_memory_bytes <= 0):
        raise ValueError("kv_cache_memory_bytes must be positive")
    command = list(previous)
    if fused_sinkhorn is not None:
        if type(fused_sinkhorn) is not bool:
            raise ValueError("fused Sinkhorn selection must be boolean")
        overrides = json.loads(command[command.index("--hf-overrides") + 1]) if "--hf-overrides" in command else {}
        if not isinstance(overrides, dict):
            raise ValueError("HF overrides must be an object")
        overrides["ascend_glm_fused_sinkhorn"] = fused_sinkhorn
        command = argument(command, "--hf-overrides", json.dumps(overrides))
    if capacity_first:
        prefix_caching = False
        max_num_seqs = 1
    if prefix_caching is not None:
        command = [value for value in command if value not in ("--enable-prefix-caching", "--no-enable-prefix-caching")]
        command.append("--enable-prefix-caching" if prefix_caching else "--no-enable-prefix-caching")
        command = argument(command, "--mamba-cache-mode", "align" if prefix_caching else "none")
    if max_num_seqs is not None:
        command = argument(command, "--max-num-seqs", str(max_num_seqs))
    if max_model_len is not None:
        command = argument(command, "--max-model-len", str(max_model_len))
    if kv_cache_memory_bytes is not None:
        command = argument(command, "--kv-cache-memory-bytes", str(kv_cache_memory_bytes))
    for flag, value in (
        ("--served-model-name", SERVICE_NAME),
        ("--host", host),
        ("--port", str(port)),
        ("--max-num-batched-tokens", str(prefill_tokens)),
        ("--worker-extension-cls", "sidecard_worker.SidecardWorkerExtension"),
        ("--compilation-config", json.dumps({"cudagraph_mode": "NONE", "mode": 0})),
    ):
        command = argument(command, flag, value)
    if "--enforce-eager" not in command:
        command.append("--enforce-eager")
    if "--speculative-config" in command:
        raise ValueError("the first sidecard service excludes MTP")
    return command


def stop_previous(pid, saved):
    if pid != saved["pid"]:
        raise ValueError("replacement PID differs from qualified service receipt")
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return
    actual = proc.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
    if actual != saved["command"]:
        raise ValueError("replacement PID command differs; refusing to signal it")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + SHUTDOWN_SECONDS
    while proc.exists():
        try:
            state = proc.joinpath("stat").read_text().split(") ", 1)[1].split()[0]
        except FileNotFoundError:
            break
        if state == "Z":
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("previous server has not stopped; inspect it before retrying")
        time.sleep(1)


def launch(
    root,
    *,
    replace_pid=None,
    host="127.0.0.1",
    port=8001,
    max_model_len=None,
    kv_cache_memory_bytes=None,
    capacity_first=False,
    prefill_tokens=128,
    prefix_caching=None,
    max_num_seqs=None,
    fused_sinkhorn=None,
):
    root = Path(root).resolve()
    if (root / "service.log").exists() or (root / "service-process.json").exists():
        raise ValueError("service launcher requires a fresh deployment; existing service receipts are preserved")
    source = root / "source"
    config = json.loads((root / "service.json").read_bytes())
    receipt = json.loads(Path(config["worker_ready_path"]).read_bytes())
    if (
        receipt["config_sha256"] != digest_file(root / "cuda-worker.json")
        or receipt["checkpoint_sha256"] != config["checkpoint_sha256"]
        or receipt["resident_layers"] != 42
        or receipt["resident_experts"]
        != (
            one_card_partition(config.get("cuda_partition", SEED_PARTITION)).resident_experts
            if config.get("one_card", False)
            else 3024
        )
        or receipt.get("tp_rank_inputs", False) != (config.get("cuda_partition", SEED_PARTITION) != SEED_PARTITION)
    ):
        raise ValueError("CUDA preparation does not match service admission")
    validate_glm_config(json.loads((Path(config["checkpoint_root"]) / "config.json").read_bytes()))
    if digest_file(Path(config["checkpoint_root"]) / "native-layout.json") != config["checkpoint_sha256"]:
        raise ValueError("service checkpoint changed after CUDA admission")
    os.kill(receipt["pid"], 0)
    saved = json.loads((root / "previous-service.json").read_bytes())
    environment = dict(os.environ)
    environment.update(json.loads((root / "runtime-environment.json").read_bytes()))
    if config.get("one_card", False):
        from .one_card import validate_device_pair

        devices = validate_device_pair(tuple(config.get("npu_devices", (0, 1))))
        if environment.get("ASCEND_RT_VISIBLE_DEVICES") != ",".join(map(str, devices)):
            raise ValueError("one-card runtime visibility differs from the admitted physical NPU pair")
    environment["PYTHONPATH"] = str(source) + ":" + environment.get("PYTHONPATH", "")
    command = serving_command(
        saved["command"],
        host=host,
        port=port,
        max_model_len=max_model_len,
        kv_cache_memory_bytes=kv_cache_memory_bytes,
        capacity_first=capacity_first,
        prefill_tokens=prefill_tokens,
        prefix_caching=prefix_caching,
        max_num_seqs=max_num_seqs,
        fused_sinkhorn=fused_sinkhorn,
    )
    # Compile/import preflight before stopping a working service. The worker
    # extension loads the same auxiliary bindings as the qualified baseline.
    subprocess.run(
        [command[0], "-c", "import sidecard_worker; print('SIDECARD_IMPORT_PREFLIGHT_PASS')"],
        cwd=source,
        env=environment,
        check=True,
    )
    if replace_pid is not None:
        stop_previous(replace_pid, saved)
    with socket.socket() as connection:
        if connection.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError("service port is occupied")
    log = root / "service.log"
    with log.open("xb") as output:
        process = subprocess.Popen(
            command,
            cwd=source,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    result = {
        "pid": process.pid,
        "command": command,
        "cwd": str(source),
        "log": str(log),
        "host": host,
        "port": port,
        "model": SERVICE_NAME,
        "cuda_pid": receipt["pid"],
    }
    (root / "service-process.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--replace-pid", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fused-sinkhorn", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--capacity-first",
        action="store_true",
        help="Use one live sequence and compact KDA state without prefix snapshots",
    )
    args = parser.parse_args()
    launch(
        args.root,
        replace_pid=args.replace_pid,
        host=args.host,
        port=args.port,
        max_model_len=args.max_model_len,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        capacity_first=args.capacity_first,
        prefill_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        fused_sinkhorn=args.fused_sinkhorn,
        prefix_caching=args.enable_prefix_caching,
    )


if __name__ == "__main__":
    main()
