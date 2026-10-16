# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import uuid

import normal_workload as w

root = w.root
results = []
context = (root / "review-context.txt").read_text()
question = (
    "\nReview scheduler fairness and cache lifetime in this code. "
    "Explain two concrete risks and the regression tests you would add."
)


def control(name, label):
    subprocess.run(
        [
            "python3",
            str(root / "control.py"),
            "--source",
            str(root / f"{name}.py"),
            "--output",
            str(root / f"context-quality-control-{label}.json"),
            "--apply",
        ],
        check=True,
    )


try:
    control("restore", "baseline")
    for mode in ["graph", "eager"]:
        control("eager-on" if mode == "eager" else "eager-off", mode)
        prompt = f"Inspection {uuid.uuid4().hex}.\n" + context + question
        for iteration in range(2):
            label = f"context-quality-{mode}-{iteration}"
            w.output = root / label
            w.output.mkdir(exist_ok=False)
            row = w.stream("isolated", prompt, 128)
            record = dict(mode=mode, iteration=iteration, **row)
            results.append(record)
            (root / "context-quality-results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(
                json.dumps(
                    dict(
                        mode=mode,
                        iteration=iteration,
                        ttft=row["ttft_s"],
                        usage=row["usage"],
                        text="".join(row["content"]),
                    )
                ),
                flush=True,
            )
finally:
    control("eager-off", "restore")
