# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
from pathlib import Path

from tools.qwen38_decode_study.benchmark import PROMPTS, stream_completion
from tools.sidecard.qfn_resident_server import QfnResidentClient

root = Path("/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016")
run = root / "resident-v7"
client = QfnResidentClient("http://127.0.0.1:8001", run / "control.token", expected_workers=6)
for label, action in [("a3", "off"), ("b3", "on"), ("a4", "off"), ("b4", "on")]:
    subprocess.run(
        [
            "bash",
            str(root / "run.sh"),
            "-m",
            "tools.qwen4exp.apply_tp6_state",
            "--run-root",
            str(run),
            "--action",
            action,
            "--output",
            str(run / ("state-ab-" + label + "-control.json")),
        ],
        check=True,
    )
    client.request("/pause?mode=wait&clear_cache=true")
    client.resume()
    with (run / ("state-ab-" + label + ".jsonl")).open("x") as log:
        for index, prompt in enumerate([PROMPTS[0], *PROMPTS]):
            tokens = 32 if index == 0 else 256
            result = stream_completion(
                client.base_url,
                dict(
                    model="qwen38-flash-next",
                    messages=[dict(role="user", content=prompt)],
                    max_tokens=tokens,
                    temperature=0,
                    seed=42,
                    ignore_eos=True,
                    chat_template_kwargs=dict(enable_thinking=False),
                    stream=True,
                    stream_options=dict(include_usage=True),
                ),
            )
            if result["usage"]["completion_tokens"] != tokens:
                raise RuntimeError("wrong completion length")
            result.update(warmup=index == 0, case=index, label=label)
            log.write(json.dumps(result) + "\n")
            log.flush()
            print(label, index, result["client_decode_tok_s"], flush=True)
