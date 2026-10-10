"""Hash all variant shards concurrently, then run complete storage admission."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.glm_perf import verify_card_checkpoint as verifier
from tools.glm_perf.native_checkpoint import file_digest

root = Path("/srv/ai/storage/artifacts/glm-card-profiles-20261010")
folder = Path("/srv/ai/storage/models/GLM-5.3-Flash-three-card-W3-in-W4-native-candidate")
manifest = json.loads((folder / "native-layout.json").read_text())
assert manifest["complete"] and manifest["self_contained_storage"]
paths = sorted(
    {folder / row["file"] for field in ("native_shards", "scale_shards", "dense_shards") for row in manifest[field]}
)


def hash_one(path):
    before = path.stat()
    digest = file_digest(path)
    after = path.stat()
    identity = lambda value: (value.st_ino, value.st_size, value.st_mtime_ns)
    assert identity(before) == identity(after), path
    return path, (digest, identity(after))


with ThreadPoolExecutor(max_workers=6) as pool:
    hashes = dict(pool.map(hash_one, paths))
(root / "w4-storage-measured-shard-hashes.json").write_text(
    json.dumps({path.name: value[0] for path, value in hashes.items()}, indent=2) + "\n"
)
original = verifier.file_digest


def measured_digest(path):
    path = Path(path)
    if path not in hashes:
        return original(path)
    digest, identity = hashes[path]
    value = path.stat()
    assert (value.st_ino, value.st_size, value.st_mtime_ns) == identity, path
    return digest


verifier.file_digest = measured_digest
result = verifier.verify(folder, verify_sha=True)
result["model_folder"] = str(folder)
(root / "w4-storage-full-check.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
