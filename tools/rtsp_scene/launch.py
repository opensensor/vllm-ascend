"""Launch a scene model with explicit devices, parallelism and optional native decode graphs."""

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

PR_SET_THP_DISABLE = 41
VISION_PARALLEL_MODES = ("weights", "data")
KV_BLOCK_SIZES = (64, 128)
MAX_MODEL_LENGTH = 8192


def build_command(
    model: str,
    port: int,
    max_sequences: int,
    int8: bool,
    tensor_parallel_size: int = 1,
    max_batched_tokens: int = 1024,
    profiler_directory: str | None = None,
    decode_graph_sizes: list[int] | None = None,
    dense: bool = False,
    vision_parallel_mode: str = "weights",
    local_frame_root: str | None = None,
    kv_block_size: int | None = None,
    async_scheduling: bool = False,
    chunked_prefill: bool = True,
) -> list[str]:
    if not model or not 1 <= port <= 65535 or min(max_sequences, tensor_parallel_size, max_batched_tokens) <= 0:
        raise ValueError("Model, valid port and positive sequence/parallel/token counts are required")
    if vision_parallel_mode not in VISION_PARALLEL_MODES:
        raise ValueError("Use weights or data parallelism for the vision encoder")
    if local_frame_root is not None and not Path(local_frame_root).is_absolute():
        raise ValueError("The shared frame root must be an absolute path")
    if kv_block_size is not None and kv_block_size not in KV_BLOCK_SIZES:
        raise ValueError("Use a supported 64- or 128-token KV block size")
    if not chunked_prefill and max_batched_tokens < MAX_MODEL_LENGTH:
        raise ValueError("Unchunked prefill requires a token budget at least as large as the model context")
    if not chunked_prefill and kv_block_size is None:
        raise ValueError("Unchunked prefill requires an explicit supported KV block size on 310P")
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model,
        "--served-model-name",
        "qwen3-vl-scene",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "float16",
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-model-len",
        str(MAX_MODEL_LENGTH),
        "--max-num-batched-tokens",
        str(max_batched_tokens),
        "--max-num-seqs",
        str(max_sequences),
        "--gpu-memory-utilization",
        "0.65",
        "--limit-mm-per-prompt",
        '{"image":8,"video":0}',
        "--mm-processor-kwargs",
        '{"max_pixels":262144}',
        "--async-scheduling" if async_scheduling else "--no-async-scheduling",
        "--disable-custom-all-reduce",
    ]
    if vision_parallel_mode == "data":
        command.extend(["--mm-encoder-tp-mode", vision_parallel_mode])
    if not chunked_prefill:
        command.append("--no-enable-chunked-prefill")
    if local_frame_root is not None:
        command.extend(["--allowed-local-media-path", local_frame_root])
    if kv_block_size is not None:
        command.extend(["--block-size", str(kv_block_size)])
    if decode_graph_sizes:
        if len(decode_graph_sizes) > 2 or any(value <= 0 or value > max_sequences for value in decode_graph_sizes):
            raise ValueError("Use at most two positive decode graph sizes within the sequence budget on 310P")
        command.extend(
            [
                "--compilation-config",
                json.dumps(
                    {
                        "mode": 0,
                        "cudagraph_mode": "FULL_DECODE_ONLY",
                        "cudagraph_capture_sizes": sorted(set(decode_graph_sizes)),
                        "cudagraph_num_of_warmups": 1,
                    }
                ),
            ]
        )
    else:
        command.append("--enforce-eager")
    if int8:
        command.extend(
            [
                "--quantization",
                "ascend",
                "--safetensors-load-strategy",
                "eager",
                "--no-enable-prefix-caching",
                "--mm-processor-cache-gb",
                "0",
            ]
        )
    if profiler_directory:
        command.extend(
            [
                "--profiler-config",
                json.dumps(
                    {
                        "profiler": "torch",
                        "torch_profiler_dir": profiler_directory,
                        "torch_profiler_with_stack": False,
                        "torch_profiler_record_shapes": True,
                        "ignore_frontend": True,
                        "delay_iterations": 0,
                        "max_iterations": 0,
                    }
                ),
            ]
        )
    if dense:
        if not int8 or tensor_parallel_size != 2:
            raise ValueError("Dense scene kernels require the INT8 TP2 model overlay")
        command.extend(["--worker-cls", "tools.rtsp_scene.dense_worker.SceneDenseWorker"])
    return command


def select_devices(devices: str | None, device: int, parallel_size: int) -> list[int]:
    selected = [int(value) for value in devices.split(",")] if devices is not None else [device]
    if min(selected) < 0 or len(set(selected)) != len(selected) or len(selected) != parallel_size:
        raise ValueError("Select distinct nonnegative devices matching tensor parallel size")
    return selected


def disable_process_thp(libc) -> None:
    # Applies to this process and its descendants, without changing host settings.
    if libc.prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, "Could not disable transparent huge pages for the loader")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--devices", help="Comma-separated logical devices for local workers, e.g. 0,1")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--vision-parallel-mode",
        choices=VISION_PARALLEL_MODES,
        default="weights",
        help="Shard vision weights or distribute images across the selected TP devices",
    )
    parser.add_argument("--max-batched-tokens", type=int, default=1024)
    parser.add_argument("--local-frame-root", help="Allow image files under this explicit private shared directory")
    parser.add_argument(
        "--kv-block-size", type=int, choices=KV_BLOCK_SIZES, help="Explicit 310P attention cache geometry"
    )
    parser.add_argument("--async-scheduling", action="store_true", help="Opt-in asynchronous scheduling experiment")
    parser.add_argument("--disable-chunked-prefill", action="store_true", help="Opt-in full-prefill attention trial")
    parser.add_argument("--profiler-directory", help="Configure the native profiler; recording starts only on request")
    parser.add_argument("--decode-graph-sizes", type=int, nargs="+", help="Capture up to two native decode batch sizes")
    parser.add_argument("--max-sequences", type=int, default=2)
    parser.add_argument("--dense", action="store_true", help="Use a prepared dense model overlay and gated worker")
    parser.add_argument(
        "--int8", action="store_true", help="Ascend quantization, eager loading and per-process THP disable"
    )
    args = parser.parse_args()
    try:
        devices = select_devices(args.devices, args.device, args.tensor_parallel_size)
        command = build_command(
            args.model,
            args.port,
            args.max_sequences,
            args.int8,
            args.tensor_parallel_size,
            args.max_batched_tokens,
            args.profiler_directory,
            args.decode_graph_sizes,
            args.dense,
            args.vision_parallel_mode,
            args.local_frame_root,
            args.kv_block_size,
            args.async_scheduling,
            not args.disable_chunked_prefill,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.int8:
        disable_process_thp(ctypes.CDLL(None, use_errno=True))
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
