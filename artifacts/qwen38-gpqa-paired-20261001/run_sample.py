"""Replay fixed GPQA Diamond questions against an OpenAI-compatible server."""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import regex as re

QUESTION_IDS = (0, 1, 2)
ANSWER_PATTERN = re.compile(r"(?i)ANSWER\s*:\s*([A-D])")


def read_questions(path):
    questions = {}
    for line in path.open():
        row = json.loads(line)
        if row["id"] in QUESTION_IDS:
            questions[row["id"]] = row
    if set(questions) != set(QUESTION_IDS):
        raise ValueError("Reference run is missing a selected question")
    return questions


def stream_completion(base_url, body):
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = last = None
    content = []
    reasoning = []
    usage = None
    finish_reason = None
    with urllib.request.urlopen(request, timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            if "error" in event:
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                part = delta.get("content") or ""
                thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if part or thought:
                    last = time.perf_counter()
                    first = first or last
                    content.append(part)
                    reasoning.append(thought)
                finish_reason = choice.get("finish_reason") or finish_reason
    if usage is None or first is None or last is None:
        raise RuntimeError("Stream did not return text and token usage")
    text = "".join(content)
    matches = ANSWER_PATTERN.findall(text)
    return {
        "content": text,
        "reasoning": "".join(reasoning),
        "prediction": matches[-1] if matches else None,
        "usage": usage,
        "finish_reason": finish_reason,
        "ttft_s": first - started,
        "decode_s": last - first,
        "decode_tok_s": (usage["completion_tokens"] - 1) / (last - first),
        "elapsed_s": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    questions = read_questions(args.reference_predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        for question_id in QUESTION_IDS:
            question = questions[question_id]
            prompt = question["origin_prompt"][0]["prompt"]
            body = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8192,
                "temperature": 0,
                "seed": 1024,
                "top_k": 20,
                "top_p": 1.0,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0,
                "ignore_eos": False,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            result = stream_completion(args.base_url, body)
            result.update(
                {
                    "id": question_id,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "gold": question["gold"],
                    "correct": result["prediction"] == question["gold"],
                    "model": args.model,
                }
            )
            output.write(json.dumps(result) + "\n")
            output.flush()
            print(
                json.dumps(
                    {
                        key: result[key]
                        for key in (
                            "id",
                            "gold",
                            "prediction",
                            "correct",
                            "usage",
                            "finish_reason",
                            "ttft_s",
                            "decode_tok_s",
                            "elapsed_s",
                        )
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
