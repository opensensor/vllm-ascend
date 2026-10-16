"""Send identical scene prompts and frames to baseline and candidate replicas."""

import argparse
import concurrent.futures
import json
import os
import uuid
from pathlib import Path

from tools.rtsp_scene.runner import SCENE_PROMPT, describe, safe_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--prompt-file", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for index, directory in enumerate(args.captures):
        source = json.loads((directory / "capture.json").read_text())
        sample = args.output / f"sample-{index:02d}"
        sample.mkdir()
        prompt = sample / "request-prompt.txt"
        text = args.prompt_file.read_text() if args.prompt_file else SCENE_PROMPT
        prompt.write_text("Request identifier: " + uuid.uuid4().hex + "\n" + text)
        record = {"capture": str(directory), "camera": source["camera"], "responses": {}}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {}
            for label, endpoint in (("baseline", args.baseline), ("candidate", args.candidate)):
                output = sample / label
                output.mkdir()
                request = argparse.Namespace(model="qwen3-vl-scene", endpoint=endpoint, prompt_file=prompt)
                jobs[label] = pool.submit(describe, directory, source, request, output)
            for label, job in jobs.items():
                try:
                    result = job.result()
                    record["responses"][label] = {
                        "latency_seconds": result["latency_seconds"],
                        "usage": result["usage"],
                        "description": result["description"],
                        "prompt_sha256": result["prompt_sha256"],
                    }
                except Exception as error:
                    record["responses"][label] = {"error": safe_error(str(error))}
        records.append(record)
        (args.output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    if any("error" in response for record in records for response in record["responses"].values()):
        raise RuntimeError("Comparison includes failed descriptions; inspect the retained responses")


if __name__ == "__main__":
    main()
