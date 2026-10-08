# SPDX-License-Identifier: Apache-2.0
"""Stage a build flag for reuse of immutable KDA score columns in UB."""

import difflib
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[3]
study = Path(__file__).resolve().parent
relative = "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd_prepare.h"
original = (root / relative).read_text()
metadata_file = study / "kda-score-cache-base.json"
if metadata_file.exists():
    metadata = json.loads(metadata_file.read_text())
    if hashlib.sha256(original.encode()).hexdigest() == metadata["candidate_sha256"]:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / relative
            target.parent.mkdir(parents=True)
            target.write_text(original)
            subprocess.run(
                ["git", "apply", "--reverse", str(study / "kda-score-cache-columns.patch")], cwd=directory, check=True
            )
            original = target.read_text()
    assert hashlib.sha256(original.encode()).hexdigest() == metadata["source_sha256"]
method_start = original.index("    __aicore__ inline void ComputeRawAqkAkkVector310P(")
method_end = original.index("\n#endif", method_start)
method = original[method_start:method_end]
anchor = "        // Run K*K and Q*K in separate passes."
assert method.count(anchor) == 1
prefetch = """#ifdef GLM_KDA_SCORE_CACHE_COLUMNS
        // Keep both original score passes and their FP16 operation order.
        // Only immutable key/gate columns are cached; no score arithmetic
        // or per-dot-product synchronization is fused by this experiment.
        // Reserve the subsequent triangular solve's UB scratch as well;
        // its highest live offset is below 72 KiB.
        constexpr uint64_t scoreCacheByteOffset = 80 * 1024;
        constexpr uint64_t scoreCacheMaxRows = 64;
        constexpr uint64_t scoreCacheMaxChannels = 256;
        const uint64_t cacheElements = curT * K_;
        const bool cacheColumns = BT_ == scoreCacheMaxRows &&
            curT == scoreCacheMaxRows &&
            K_ >= 16 && K_ <= scoreCacheMaxChannels && K_ % 16 == 0 &&
            scoreCacheByteOffset + 2 * cacheElements * sizeof(T) <=
                KDA_VEC_ARENA_ELEMENTS * sizeof(float);
        LocalTensor<T> cachedKeys = typedArena;
        LocalTensor<T> cachedGates = typedArena;
        if (cacheColumns) {
            cachedKeys = typedArena[scoreCacheByteOffset / sizeof(T)];
            cachedGates = cachedKeys[cacheElements];
            if (inputSequenceMajor_) {
                for (uint64_t col = 0; col < curT; ++col) {
                    LocalTensor<T> keyColumn = cachedKeys[col * K_];
                    CopyVectorIn(keyColumn, k_, QOffset(b, h, start + col, 0), K_);
                }
            } else {
                CopyVectorIn(cachedKeys, k_, QOffset(b, h, start, 0), cacheElements);
            }
            // Gates are head-major for both public query/key layouts.
            CopyVectorIn(cachedGates, gk_, KVOffset(b, hv, start, 0, K_), cacheElements);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        }
#endif

"""
method = method.replace(anchor, prefetch + anchor)
loads = """                CopyVectorIn(
                    kCol, k_, QOffset(b, h, start + col, 0), K_);
                SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                CopyVectorIn(
                    gCol, gk_, KVOffset(b, hv, start + col, 0, K_), K_);
                SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
"""
assert method.count(loads) == 2
replacement = (
    """#ifdef GLM_KDA_SCORE_CACHE_COLUMNS
                if (cacheColumns) {
                    kCol = cachedKeys[col * K_];
                    gCol = cachedGates[col * K_];
                } else {
#endif
"""
    + loads
    + """#ifdef GLM_KDA_SCORE_CACHE_COLUMNS
                }
#endif
"""
)
method = method.replace(loads, replacement)
candidate = original[:method_start] + method + original[method_end:]
patch = "".join(
    difflib.unified_diff(
        original.splitlines(True), candidate.splitlines(True), fromfile="a/" + relative, tofile="b/" + relative
    )
)
(study / "kda-score-cache-columns.patch").write_text(patch)
(study / "kda-score-cache-base.json").write_text(
    json.dumps(
        {
            "path": relative,
            "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
            "compile_flag": "GLM_KDA_SCORE_CACHE_COLUMNS",
            "deployed_header_sha256": "36707989c3126c62dff360c80b3d189b85d490a5fc61a21c7a84b4e531e97fb0",
        },
        indent=2,
    )
    + "\n"
)
print("Staged", study / "kda-score-cache-columns.patch")
