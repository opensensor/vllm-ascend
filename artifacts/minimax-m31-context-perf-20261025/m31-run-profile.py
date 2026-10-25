import json
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.minimax_m3.resident import candidate_source

c = ResidentClient("http://127.0.0.1:8000", 6)
c.request("/resident/maintenance")
try:
    print(
        json.dumps(
            c.switch(
                Control(
                    uuid.uuid4().hex, "direct-target", "profile", Path("/workspace/m31-profile-source.py").read_text()
                )
            )
        ),
        flush=True,
    )
    r = c.request(
        "/v1/chat/completions",
        {
            "model": "minimax-m31-flash",
            "messages": [{"role": "user", "content": "Explain why the sky is blue in two sentences."}],
            "temperature": 0,
            "max_tokens": 64,
        },
    )
    Path("/workspace/m31-profile-response.json").write_text(json.dumps(r, indent=2) + "\n")
finally:
    print(
        json.dumps(
            c.switch(Control(uuid.uuid4().hex, "direct-target", "v6", candidate_source(native_name="minimax_w4_v6")))
        ),
        flush=True,
    )
    c.request("/resident/maintenance", method="DELETE")
