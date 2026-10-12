# SPDX-License-Identifier: Apache-2.0
"""OpenAI text endpoint for the serialized complete-layer sidecard runtime."""

import argparse
import atexit
import json
import os
import threading
import time
import uuid
from pathlib import Path

from .layer_actor import Coordinator
from .layer_reasoning import ReasoningStream, chat_message

MODEL_NAME = "glm-sidecard"
DEFAULT_MAX_TOKENS = 128
VOCAB_SIZE = 154880
WARMUP_PROMPT = "Reply with the word hello."


async def managed_deltas(iterator):
    """Close the CUDA server's synchronous generator on every ASGI exit."""
    import anyio
    from starlette.concurrency import iterate_in_threadpool

    try:
        async for delta in iterate_in_threadpool(iterator):
            yield delta
    finally:
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(iterator.close)


def request_tokens(body, tokenizer, max_context, *, chat):
    if not isinstance(body, dict) or body.get("model", MODEL_NAME) != MODEL_NAME:
        raise ValueError("unknown model or malformed request")
    if body.get("n", 1) != 1 or body.get("tools") or body.get("logprobs") or body.get("response_format"):
        raise ValueError("initial serialized runtime supports one plain text completion per request")
    if chat:
        messages = body.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(message, dict)
                or not isinstance(message.get("content"), str)
                or message.get("role") not in ("system", "user", "assistant")
                for message in messages
            )
        ):
            raise ValueError("text messages with system/user/assistant roles are required")
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, reasoning_effort=body.get("reasoning_effort", "max")
        )
        if hasattr(tokens, "keys"):
            tokens = tokens["input_ids"]
    else:
        prompt = body.get("prompt")
        if isinstance(prompt, str) and prompt:
            tokens = tokenizer.encode(prompt)
        elif (
            isinstance(prompt, list)
            and prompt
            and all(type(value) is int and 0 <= value < VOCAB_SIZE for value in prompt)
        ):
            tokens = prompt
        else:
            raise ValueError("prompt must be text or one list of valid token IDs")
    limit = body.get("max_completion_tokens", body.get("max_tokens", DEFAULT_MAX_TOKENS))
    if type(limit) is not int or limit < 1 or not tokens or len(tokens) + limit > max_context:
        raise ValueError(f"prompt and requested output must fit the qualified {max_context}-token context")
    temperature = body.get("temperature", 0)
    if not isinstance(temperature, (int, float)) or temperature != 0:
        raise ValueError("initial correctness gate supports greedy temperature=0")
    if body.get("stop"):
        raise ValueError("custom stop sequences are not yet supported")
    if type(body.get("ignore_eos", False)) is not bool:
        raise ValueError("ignore_eos must be boolean")
    if type(body.get("stream", False)) is not bool or not isinstance(body.get("stream_options", {}), dict):
        raise ValueError("stream must be boolean and stream_options must be an object")
    if chat and body.get("reasoning_effort", "max") not in ("low", "high", "max"):
        raise ValueError("reasoning_effort must be low, high, or max")
    return tokens, limit


class Generation:
    def __init__(self, coordinator, tokenizer):
        self.coordinator, self.tokenizer = coordinator, tokenizer
        self.lock = threading.Lock()
        self.eos = set(coordinator.config["eos_token_id"])
        self.last_result = None

    def generate(self, tokens, limit, result=None, *, ignore_eos=False):
        if result is None:
            result = {}
        with self.lock:
            engine = self.coordinator
            engine.reset(context_limit=len(tokens) + limit)
            engine.metrics = {name: [] for name in engine.metrics}
            metric_start = {name: len(values) for name, values in engine.metrics.items()}
            begin = time.perf_counter()
            logits = engine.prefill(tokens, overlap=engine.spec.get("prefill_overlap", True))
            output, text, first = [], "", None
            for _ in range(limit):
                token = int(logits.argmax().cpu())
                output.append(token)
                now = time.perf_counter()
                if first is None:
                    first = now
                if token in self.eos and not ignore_eos:
                    break
                next_text = self.tokenizer.decode(output, skip_special_tokens=True)
                if next_text.startswith(text):
                    delta = next_text[len(text) :]
                else:
                    # Full decoding can change a trailing replacement character;
                    # withhold incomplete Unicode instead of emitting corruption.
                    delta = ""
                if next_text.endswith("\ufffd"):
                    next_text, delta = text, ""
                text = next_text
                yield delta
                if len(output) < limit:
                    logits = engine.forward([token])
            end = time.perf_counter()
            stages = {name: values[metric_start[name] :] for name, values in engine.metrics.items()}
            self.last_result = {
                "prompt_tokens": len(tokens),
                "output_tokens": len(output),
                "output_ids": output,
                "text": self.tokenizer.decode(output, skip_special_tokens=True),
                "ttft_seconds": first - begin,
                "total_seconds": end - begin,
                "decode_tokens_per_second": (len(output) - 1) / (end - first)
                if len(output) > 1 and end > first
                else None,
                "stages": stages,
                "finish_reason": "stop" if output[-1] in self.eos and not ignore_eos else "length",
            }
            result.update(self.last_result)
            (engine.output / "last-generation.json").write_text(json.dumps(self.last_result, indent=2) + "\n")
            print(
                json.dumps(
                    dict(
                        kind="generation_completed",
                        **{
                            key: value
                            for key, value in self.last_result.items()
                            if key not in ("stages", "output_ids", "text")
                        },
                    )
                ),
                flush=True,
            )


def main():
    """Expose readiness only after real-weight smoke and pipeline parity gates."""
    import anyio
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    specification = json.loads(args.config.read_bytes())
    if specification.get("cuda_environment"):
        os.environ.update(json.loads(Path(specification["cuda_environment"]).read_bytes()))
    tokenizer = AutoTokenizer.from_pretrained(
        specification["checkpoint"], local_files_only=True, trust_remote_code=False
    )
    tokens, _ = request_tokens(
        {"messages": [{"role": "user", "content": WARMUP_PROMPT}], "max_tokens": 64, "reasoning_effort": "low"},
        tokenizer,
        specification.get("max_context", 2048),
        chat=True,
    )
    engine = Coordinator(args.config)
    atexit.register(engine.close)
    probe = (tokenizer.encode("Describe the purpose of a Python dictionary. ") * 20)[:96]
    engine.reset(context_limit=len(probe) + 1)
    start = time.perf_counter()
    serial = engine.prefill(probe, overlap=False).cpu()
    serial_seconds = time.perf_counter() - start
    engine.reset(context_limit=len(probe) + 1)
    start = time.perf_counter()
    pipelined = engine.prefill(probe, overlap=True).cpu()
    pipeline_seconds = time.perf_counter() - start
    error = (serial - pipelined).abs()
    parity = {
        "passed": bool(engine.torch.isfinite(pipelined).all()) and bool((error <= 0.02 + 0.002 * serial.abs()).all()),
        "greedy_next_token_equal": int(serial.argmax()) == int(pipelined.argmax()),
        "bitwise_equal": engine.torch.equal(serial, pipelined),
        "max_abs_error": float(error.max()),
        "serial_seconds": serial_seconds,
        "pipeline_seconds": pipeline_seconds,
        "prompt_tokens": len(probe),
    }
    (engine.output / "pipeline-parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    if not parity["passed"] or not parity["greedy_next_token_equal"]:
        raise RuntimeError("full-model serialized/pipeline parity failed")
    # Fixed teacher-forced tokens isolate arithmetic repeatability from an
    # early argmax divergence. Hash complete handoff bodies and full logits.
    traces, forced = [], []
    repeat_prompt = tokenizer.encode('def factorial(n):\n    """Compute factorial recursively."""\n')
    for trial in range(3):
        engine.reset(context_limit=len(repeat_prompt) + 64)
        engine.fingerprints = []
        logits = engine.prefill(repeat_prompt)
        for step in range(64):
            token = int(logits.argmax().cpu()) if trial == 0 else forced[step]
            if trial == 0:
                forced.append(token)
            if step < 63:
                logits = engine.forward([token])
        traces.append(engine.fingerprints)
    engine.fingerprints = None
    repeat = {"passed": all(trace == traces[0] for trace in traces), "teacher_forced_tokens": forced, "traces": traces}
    (engine.output / "decode-repeat-gate.json").write_text(json.dumps(repeat, indent=2) + "\n")
    if not repeat["passed"]:
        raise RuntimeError("complete-model decode boundary/logit repeatability failed")
    generation = Generation(engine, tokenizer)
    # Complete a real-weight request before exposing readiness.
    startup_result = {}
    list(generation.generate(tokens, 64, startup_result))
    warmup = dict(startup_result, complete_model_layers=45, dummy_weights=False, physical_npu_devices=[2, 3])
    (engine.output / "startup-inference.json").write_text(json.dumps(warmup, indent=2) + "\n")
    if not warmup["text"].strip():
        raise RuntimeError("real-weight startup inference produced no text")
    if args.gate_only:
        engine.close()
        return
    app = FastAPI()

    class ClosingStreamingResponse(StreamingResponse):
        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                # A disconnected client can leave a suspended synchronous
                # generator alive. Explicitly close both async wrappers, so
                # its request lock never depends on garbage collection.
                with anyio.CancelScope(shield=True):
                    await self.body_iterator.aclose()

    @app.get("/health")
    def health():
        if engine.failed or any(process.poll() is not None for process in engine.processes):
            raise HTTPException(503, "stage worker unavailable")
        return {
            "status": "ok",
            "architecture": "cuda-prefix-ascend-middle-cuda-suffix",
            "max_model_len": engine.max_context,
        }

    @app.get("/v1/models")
    def models():
        health()
        return {
            "object": "list",
            "data": [
                {"id": MODEL_NAME, "object": "model", "owned_by": "opensensor", "max_model_len": engine.max_context}
            ],
        }

    @app.post("/tokenize")
    def tokenize(body: dict):
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise HTTPException(400, "tokenize requires a text prompt")
        tokens = tokenizer.encode(prompt, add_special_tokens=body.get("add_special_tokens", False))
        return {"tokens": tokens, "count": len(tokens), "max_model_len": engine.max_context}

    def completion(body, chat):
        health()
        try:
            tokens, limit = request_tokens(body, tokenizer, engine.max_context, chat=chat)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        request_id = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
        created = int(time.time())
        result = {}
        if body.get("stream", False):

            async def events():
                reasoning = ReasoningStream() if chat else None
                deltas = managed_deltas(
                    generation.generate(tokens, limit, result, ignore_eos=body.get("ignore_eos", False))
                )
                try:
                    async for delta in deltas:
                        choice = {"index": 0, "finish_reason": None}
                        choice["delta" if chat else "text"] = reasoning.feed(delta) if chat else delta
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "id": request_id,
                                    "object": "chat.completion.chunk" if chat else "text_completion",
                                    "created": created,
                                    "model": MODEL_NAME,
                                    "choices": [choice],
                                }
                            )
                            + "\n\n"
                        )
                finally:
                    with anyio.CancelScope(shield=True):
                        await deltas.aclose()
                choice = {"index": 0, "finish_reason": result["finish_reason"]}
                choice["delta" if chat else "text"] = reasoning.feed("", final=True) if chat else ""
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": request_id,
                            "object": "chat.completion.chunk" if chat else "text_completion",
                            "created": created,
                            "model": MODEL_NAME,
                            "choices": [choice],
                        }
                    )
                    + "\n\n"
                )
                if body.get("stream_options", {}).get("include_usage", False):
                    usage = {
                        "prompt_tokens": len(tokens),
                        "completion_tokens": result["output_tokens"],
                        "total_tokens": len(tokens) + result["output_tokens"],
                    }
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": request_id,
                                "object": "chat.completion.chunk" if chat else "text_completion",
                                "created": created,
                                "model": MODEL_NAME,
                                "choices": [],
                                "usage": usage,
                            }
                        )
                        + "\n\n"
                    )
                yield "data: [DONE]\n\n"

            return ClosingStreamingResponse(events(), media_type="text/event-stream")
        list(generation.generate(tokens, limit, result, ignore_eos=body.get("ignore_eos", False)))
        choice = {"index": 0, "finish_reason": result["finish_reason"]}
        choice["message" if chat else "text"] = chat_message(result["text"]) if chat else result["text"]
        return {
            "id": request_id,
            "object": "chat.completion" if chat else "text_completion",
            "created": created,
            "model": MODEL_NAME,
            "choices": [choice],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": result["output_tokens"],
                "total_tokens": len(tokens) + result["output_tokens"],
            },
        }

    @app.post("/v1/chat/completions")
    def chat_completions(body: dict):
        return completion(body, True)

    @app.post("/v1/completions")
    def completions(body: dict):
        return completion(body, False)

    class Server(uvicorn.Server):
        def handle_exit(self, sig, frame):
            engine.stopping = True
            super().handle_exit(sig, frame)

    try:
        Server(uvicorn.Config(app, host=args.host, port=args.port, timeout_graceful_shutdown=5)).run()
    finally:
        engine.close()


if __name__ == "__main__":
    main()
