import argparse
import hashlib
import json
import urllib.request
import uuid
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--apply", action="store_true")
a = p.parse_args()
token = Path("/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016/resident-v8/control.token").read_text().strip()


def request(path, payload=None, method="POST"):
    req = urllib.request.Request(
        "http://127.0.0.1:8001" + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json", "x-qfn-resident-token": token},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        data = response.read()
    return json.loads(data) if data else None


result = {}


def save():
    a.output.write_text(json.dumps(result, indent=2) + "\n")


with a.output.open("x") as output:
    output.write("{}\n")
result["initial_pause"] = request("/is_paused", method="GET")
assert result["initial_pause"]["is_paused"] is False, "server already paused"
result["before"] = request("/collective_rpc", dict(method="resident_status", args=[], timeout=600))
save()
print("Pausing with wait and clear_cache=false", flush=True)
result["pause"] = request("/pause?mode=wait&clear_cache=false")
save()
source = a.source.read_text()
revision = "preserve_" + uuid.uuid4().hex
identity = dict(revision=revision, sha256=hashlib.sha256(source.encode()).hexdigest())
result["prepare"] = request("/qfn_engine_control", dict(operation="prepare", source=source, **identity))
save()
if not result["prepare"].get("passed"):
    print(json.dumps(result["prepare"]), flush=True)
    raise RuntimeError("preparation failed; inspect before resuming")
if a.apply:
    result["apply"] = request("/qfn_engine_control", dict(operation="apply", **identity))
    save()
    if not result["apply"].get("passed"):
        raise RuntimeError("application failed; inspect before resuming")
else:
    result["discard"] = request("/qfn_engine_control", dict(operation="discard", revision=revision))
    save()
    assert result["discard"].get("passed") and not result["discard"].get("failed")
result["after"] = request("/collective_rpc", dict(method="resident_status", args=[], timeout=600))
save()
before = sorted(result["before"]["results"], key=lambda row: row["rank"])
after = sorted(result["after"]["results"], key=lambda row: row["rank"])
assert len(before) == len(after) == 6
for old, new in zip(before, after):
    assert not new["graphs_dirty"] and not new["native_failed"]
    for key in ("rank", "pid", "weight_storage_digest", "generation", "digest", "mode", "candidate", "native_loaded"):
        assert old[key] == new[key], key
result["resume"] = request("/resume")
save()
print("Resumed; retained worker, weight, native-owner and graph-generation identities.", flush=True)
