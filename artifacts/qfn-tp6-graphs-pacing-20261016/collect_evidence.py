# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path

root = Path("/srv/ai/storage/artifacts/qfn-tp6-graphs-pacing-20261016")
output = root / "export"
output.mkdir(exist_ok=True)


def read(name):
    return json.loads((root / name).read_text())


def write(name, value):
    (output / name).write_text(json.dumps(value, indent=2) + "\n")


def request(row):
    result = {k: v for k, v in row.items() if k not in ("events", "content")}
    if "content" in row:
        result["content_sha256"] = hashlib.sha256("".join(row["content"]).encode()).hexdigest()
    result["content_updates"] = len(row.get("events", []))
    return result


for name in ("sweep", "confirm", "realistic"):
    if not (root / f"{name}-results.json").exists():
        continue
    result = []
    for row in read(f"{name}-results.json"):
        item = {k: v for k, v in row.items() if k not in ("prefill", "decode")}
        item.update(prefill=request(row["prefill"]), decode=request(row["decode"]))
        result.append(item)
    write(f"{name}-summary.json", result)
for name in ("hc-graph-gate", "hc-injection-gate", "hc-graph-retire", "hc-injection-retire"):
    receipt = read(name + ".json")
    validation = receipt["prepare"]["validation"]
    result = dict(passed=validation["passed"], source_sha256=receipt["prepare"]["sha256"], workers=[])
    for row in validation["workers"]:
        item = {k: v for k, v in row.items() if k not in ("before", "after")}
        if "before" in row:
            item["snapshot_unchanged"] = row["before"] == row["after"]
        result["workers"].append(item)
    write(name + ".json", result)
quality = []
for row in read("quality-results.json"):
    response = row["response"]
    quality.append(
        dict(
            mode=row["mode"],
            prompt_index=row["prompt_index"],
            text_sha256=row["text_sha256"],
            elapsed_s=row["elapsed_s"],
            usage=response["usage"],
            request_id=response["id"],
        )
    )
write("quality-summary.json", quality)
if (root / "context-quality-results.json").exists():
    write("context-quality-summary.json", [request(r) for r in read("context-quality-results.json")])
write(
    "raw-manifest.json",
    dict(
        remote_root=str(root),
        files=[
            dict(
                path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()
            )
            for p in sorted(root.rglob("*"))
            if p.is_file()
            and "export" not in p.relative_to(root).parts
            and "__pycache__" not in p.relative_to(root).parts
        ],
    ),
)

if (root / "final-status.json").exists():
    receipt = read("final-status.json")
    write(
        "final-status.json",
        dict(
            validation=receipt["prepare"]["validation"],
            resumed=receipt["resume"],
            workers=[
                {
                    k: row[k]
                    for k in (
                        "rank",
                        "pid",
                        "generation",
                        "mode",
                        "graphs_dirty",
                        "native_failed",
                        "shared_prefill_ple",
                    )
                }
                for row in receipt["after"]["results"]
            ],
        ),
    )
