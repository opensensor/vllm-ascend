# SPDX-License-Identifier: Apache-2.0
import concurrent.futures
import json
import subprocess
import uuid

import normal_workload as w

root = w.root
results = []


def control(mode, label):
    subprocess.run(
        [
            "python3",
            str(root / "control.py"),
            "--source",
            str(root / f"{mode}.py"),
            "--output",
            str(root / f"control-{label}.json"),
            "--apply",
        ],
        check=True,
    )


context = (root / "review-context.txt").read_text()
decode_prompt = context + (
    "\nGive a detailed code review covering scheduler fairness, cache lifetime, "
    "tensor parallel shapes, synchronization, and failure recovery. For each issue explain "
    "the precise code behavior, a proposed improvement, and a concrete regression test. "
    "Include at least twelve detailed observations."
)
prefill_context = (root / "prefill-context.txt").read_text()
try:
    w.stream("warmup", decode_prompt, 1)
    for index, mode in enumerate(["d8", "c256d8", "c256d8", "d8"]):
        label = f"realistic-{index}-{mode}"
        control(mode, label)
        w.output = root / label
        w.output.mkdir(exist_ok=False)
        w.trigger.clear()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            decoder = pool.submit(w.stream, "decode", decode_prompt, 768)
            assert w.trigger.wait(180)
            prefill = pool.submit(
                w.stream,
                "prefill",
                f"Review {uuid.uuid4().hex}.\n"
                + prefill_context
                + "\nIdentify the most important scheduler fairness concern in two sentences.",
                64,
            )
            rows = {kind: task.result() for kind, task in [("prefill", prefill), ("decode", decoder)]}
        start = rows["prefill"]["start_ns"]
        end = rows["prefill"]["events"][0]["time_ns"]
        overlap = [e for e in rows["decode"]["events"] if start <= e["time_ns"] <= end]
        record = dict(
            mode=mode,
            label=label,
            **rows,
            decoder_covers_prefill=rows["decode"]["end_ns"] > end,
            overlap_updates=len(overlap),
        )
        results.append(record)
        (root / "realistic-results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    mode=mode,
                    ttft=rows["prefill"]["ttft_s"],
                    max_gap=rows["decode"]["max_content_gap_s"],
                    decode_usage=rows["decode"]["usage"],
                    prefill_usage=rows["prefill"]["usage"],
                    covers=record["decoder_covers_prefill"],
                    overlap_updates=len(overlap),
                )
            ),
            flush=True,
        )
finally:
    control("restore", "realistic-restore")
