"""Profile a new long prefill while a resident request is already decoding."""

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import preserved_identity
from tools.qwen4exp.thermal_guard import admit, query
from tools.sidecard.qfn_resident_server import QfnResidentClient

ROOT = Path("/srv/ai/storage/artifacts/qfn-local-pack-20261016")
RUN = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v16")
SOURCE = """
def operation_inner(worker):
    worker = getattr(worker, "worker", worker)
    from pathlib import Path
    import json
    from tools.qwen4exp.mixed_prefill_profile import install_worker, stop_worker
    from tools.sidecard.qfn_collective_profile import QfnProfileWorkerExtension
    if START:
        result = install_worker(worker)
        if MODE == 'cpu':
            result['profiler'] = QfnProfileWorkerExtension.qfn_cpu_profile_start(worker)
        elif MODE == 'native':
            payload = json.dumps(dict(output_dir=OUTPUT, tag='native', delay=DELAY, max=STEPS))
            result['profiler'] = QfnProfileWorkerExtension.qfn_native_profile_start(worker, payload)
            if 'error' in result['profiler']: raise RuntimeError(result['profiler']['error'])
        return result
    result = stop_worker(worker)
    Path(OUTPUT, f'rank{worker.rank}-spans.json').write_text(json.dumps(result))
    result = dict(rank=worker.rank, passed=True, total=result['total'], dropped=result['dropped'])
    if MODE == 'cpu':
        result['profiler'] = QfnProfileWorkerExtension.qfn_cpu_profile_stop(worker, OUTPUT)
    elif MODE == 'native':
        result['profiler'] = QfnProfileWorkerExtension.qfn_native_profile_stop(worker)
        if not result['profiler'].get('ownership_released'): raise RuntimeError(str(result['profiler']))
        if not result['profiler'].get('recorded'): raise RuntimeError('native profiler recorded no iterations')
    return result
def operation(wrapper):
    try:
        return operation_inner(wrapper)
    except Exception as error:
        import traceback
        return dict(rank=getattr(wrapper, 'rank', None), passed=False,
                    error=repr(error), traceback=traceback.format_exc())
class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        return dict(passed=True)
    def apply(self, engine):
        import json
        from pathlib import Path
        from tools.qwen4exp.mixed_prefill_profile import Recorder
        self.validate(engine)
        if START:
            assert getattr(engine, '_qfn_mixed_profile', None) is None
            engine._qfn_mixed_original_limit = engine.vllm_config.scheduler_config.long_prefill_token_threshold
            if LIMIT is not None:
                engine.vllm_config.scheduler_config.long_prefill_token_threshold = LIMIT
            recorder = engine._qfn_mixed_profile = Recorder()
            def metadata(*args, **kwargs):
                return dict(requests={r.request_id:dict(computed=r.num_computed_tokens,
                    prompt=r.num_prompt_tokens, output=r.num_output_tokens) for r in engine.scheduler.running})
            recorder.wrap(engine.scheduler, 'schedule', 'schedule', metadata)
        workers = engine.model_executor.collective_rpc(operation)
        if not START:
            recorder = engine._qfn_mixed_profile
            recorder.restore()
            Path(OUTPUT, 'engine-spans.json').write_text(json.dumps(recorder.snapshot()))
            engine._qfn_mixed_profile = None
            engine.vllm_config.scheduler_config.long_prefill_token_threshold = engine._qfn_mixed_original_limit
        config = engine.vllm_config.scheduler_config
        return dict(passed=all(row.get("passed", "error" not in row) for row in workers),
            workers=workers, scheduler=type(engine.scheduler).__name__,
            async_scheduling=engine.async_scheduling, batch_queue_size=engine.batch_queue_size,
            max_num_scheduled_tokens=engine.scheduler.max_num_scheduled_tokens,
            max_num_batched_tokens=config.max_num_batched_tokens,
            long_prefill_token_threshold=config.long_prefill_token_threshold)
def prepare(engine): return Candidate()
"""


def metrics(client):
    req = urllib.request.Request(client.base_url + "/metrics", headers={"x-qfn-resident-token": client.token})
    with urllib.request.urlopen(req, timeout=15) as response:
        return response.read().decode()


def main():
    """Drain to install hooks, overlap two streams, then drain and restore owners."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--profile-delay", type=int, default=10)
    parser.add_argument("--profile-iterations", type=int, default=2)
    parser.add_argument("--mode", choices=("spans", "cpu", "native"), default="spans")
    parser.add_argument("--paragraphs", type=int, default=500)
    workload = parser.add_mutually_exclusive_group()
    workload.add_argument("--baseline", action="store_true")
    workload.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--prefill-limit", type=int, choices=(128, 256, 512))
    parser.add_argument("--run-root", type=Path, default=RUN)
    args = parser.parse_args()
    assert Path(args.label).name == args.label
    assert 0 <= args.profile_delay <= 128 and 1 <= args.profile_iterations <= 8
    output = ROOT / args.label
    output.mkdir(parents=True, exist_ok=False)
    client = QfnResidentClient("http://127.0.0.1:8001", args.run_root / "control.token", timeout=900)
    record = dict(mode=args.mode, before=client.rpc("resident_status"), metrics_before=metrics(client))
    for line in record["metrics_before"].splitlines():
        if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{")):
            assert float(line.rsplit(" ", 1)[1]) == 0, "defer experiment while server is busy"
    record["sensors_before"] = admit(query(), max_temperature_c=85)
    assert record["sensors_before"]["passed"], record["sensors_before"]

    def save():
        (output / "receipt.json").write_text(json.dumps(record, indent=2))

    def control(start):
        client.request("/pause?mode=wait&clear_cache=false")
        source = (
            f"START={start!r}\nMODE={args.mode!r}\nOUTPUT={str(output)!r}\nLIMIT={args.prefill_limit!r}\n"
            f"DELAY={args.profile_delay!r}\nSTEPS={args.profile_iterations!r}\n" + SOURCE
        )
        revision, digest = "mixed_" + uuid.uuid4().hex, hashlib.sha256(source.encode()).hexdigest()
        prepared = client.request(
            "/qfn_engine_control", dict(operation="prepare", revision=revision, sha256=digest, source=source)
        )
        assert prepared.get("passed"), prepared
        result = client.request("/qfn_engine_control", dict(operation="apply", revision=revision, sha256=digest))
        record["start" if start else "stop"] = result
        save()
        assert result.get("passed"), result
        client.resume()

    trigger = threading.Event()

    def stream(name, content, tokens):
        body = dict(
            model="qwen38-flash-next",
            messages=[dict(role="user", content=content)],
            temperature=0,
            seed=42,
            max_tokens=tokens,
            min_tokens=tokens,
            ignore_eos=True,
            chat_template_kwargs=dict(enable_thinking=False),
            stream=True,
            stream_options=dict(include_usage=True),
        )
        request = urllib.request.Request(
            client.base_url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        result = dict(
            name=name,
            start_ns=time.monotonic_ns(),
            events=[],
            prompt_sha256=hashlib.sha256(content.encode()).hexdigest(),
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                raw = line[6:].strip()
                if raw == b"[DONE]":
                    break
                event = json.loads(raw)
                if "error" in event:
                    raise RuntimeError(event["error"])
                result["request_id"] = event.get("id", result.get("request_id"))
                if event.get("usage"):
                    result["usage"] = event["usage"]
                for choice in event.get("choices", []):
                    text = choice.get("delta", {}).get("content")
                    if text:
                        result["events"].append(dict(time_ns=time.monotonic_ns(), characters=len(text)))
                        if name == "decode" and len(result["events"]) >= 8:
                            trigger.set()
        result["end_ns"] = time.monotonic_ns()
        events = [e["time_ns"] for e in result["events"]]
        gaps = [(b - a) / 1e9 for a, b in zip(events, events[1:])]
        result["ttft_s"] = (events[0] - result["start_ns"]) / 1e9
        result["max_content_gap_s"] = max(gaps, default=0)
        result["median_content_gap_s"] = statistics.median(gaps) if gaps else None
        assert result["usage"]["completion_tokens"] == tokens
        (output / f"{name}-stream.json").write_text(json.dumps(result, indent=2))
        return result

    nonce = uuid.uuid4().hex
    long_prompt = (
        f"Experiment {nonce}. Read these engineering notes.\n"
        + (
            "The cache stores immutable entries indexed by a stable key. Readers verify ownership before use. "
            "Writers allocate a new version and retire older entries only after pending readers finish. "
            "Requests may arrive concurrently, so the scheduler tracks their progress independently.\n"
            * args.paragraphs
        )
        + "\nSummarize the design in one sentence."
    )
    control(True)
    try:
        if args.prefill_only:
            record["prefill"] = stream("prefill", long_prompt, 16)
            save()
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                decode = pool.submit(
                    stream,
                    "decode",
                    f"Experiment {nonce}. Write a detailed numbered guide to "
                    "implementing a hash table, with Python examples and tests.",
                    128,
                )
                if not args.baseline:
                    assert trigger.wait(30), "decoder did not begin streaming"
                    prefill = pool.submit(stream, "prefill", long_prompt, 16)
                    record["prefill"] = prefill.result()
                    save()
                record["decode"] = decode.result()
                save()
    finally:
        control(False)
    record["after"] = client.rpc("resident_status")
    assert preserved_identity(record["before"]) == preserved_identity(record["after"])
    record["metrics_after"] = metrics(client)
    record["sensors_after"] = admit(query(), max_temperature_c=85)
    record["unpaused"] = not client.request("/is_paused", method="GET")["is_paused"]
    record["profiler_status"] = client.rpc("qfn_native_profile_status")
    save()
    print(
        json.dumps(
            {
                name: {k: v for k, v in record[name].items() if k not in ("events",)}
                for name in ("decode", "prefill")
                if name in record
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
