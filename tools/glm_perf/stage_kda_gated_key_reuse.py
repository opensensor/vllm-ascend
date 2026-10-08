# SPDX-License-Identifier: Apache-2.0
"""Stage an isolated full-chunk candidate reusing gated keys between score passes."""

import argparse
import hashlib
import json
from pathlib import Path

FLAG = "GLM_KDA_GATED_KEY_REUSE"
EXTRA_UB_BYTES = 64 * 128 * 2


def transform(source):
    if FLAG in source:
        raise ValueError("KDA gated-key reuse already staged")
    begin = source.index("    __aicore__ inline void ComputeRawAqkAkkVector310P(")
    end = source.index("\n    }\n#endif", begin)
    method = source[begin:end]
    anchor = "        for (uint64_t row = 0; row < curT; ++row) {"
    starts = [index for index in range(len(method)) if method.startswith(anchor, index)]
    if len(starts) != 2:
        raise ValueError("authoritative separate score passes changed")
    first_loop = method[starts[0] : starts[1]]
    second_loop = method[starts[1] :]
    first = first_loop[first_loop.index("\n") + 1 : first_loop.rfind("\n        }")]
    second = second_loop[second_loop.index("\n") + 1 : second_loop.rfind("\n        }")]
    copied = "                Mul(product, rowVector, kGated, static_cast<uint32_t>(K_));"
    if first.count(copied) != 1 or "DataCopy(akk_" not in first or "DataCopy(aqk_" not in second:
        raise ValueError("authoritative score product or writeback changed")
    # Preserve two distinct column passes and their GM-output fences. Copy the
    # rounded FP16 gated key with multiplication by +1, preserving finite bits
    # and signed zero. No FP32 promotion or reassociation of products is added.
    first = first.replace(
        copied,
        "                Muls(gatedKeys[col * K_], kGated, static_cast<T>(1.0f), "
        "static_cast<uint32_t>(K_));\n                PipeBarrier<PIPE_V>();\n" + copied,
    )
    # The Q*K pass rebinds kGated to a cached column. Reset the writable scratch
    # at each new K*K row so later columns cannot overwrite earlier cached keys.
    first = "            kGated = typedArena[5 * K_];\n" + first
    arithmetic = second.index("                Sub(gate, gRow, gCol,")
    product = second.index("                Mul(product, rowVector, kGated,", arithmetic)
    second = second[:arithmetic] + "                kGated = gatedKeys[col * K_];\n" + second[product:]
    branch = (
        "#if defined(GLM_KDA_GATED_KEY_REUSE) && defined(GLM_KDA_SCORE_CACHE_COLUMNS)\n"
        "        if (SAFE_GATE && IsSameType<T, half>::value && K_ == 128 && "
        "BT_ == 64 && curT == 64 && cacheColumns) {\n"
        "            LocalTensor<T> gatedKeys = gatedKeyReuseBuf_.Get<T>();\n"
        + anchor
        + "\n"
        + first
        + "\n"
        + second
        + "\n        }\n            return;\n        }\n#endif\n\n"
    )
    source = source[:begin] + method[: starts[0]] + branch + method[starts[0] :] + source[end:]
    replacements = (
        (
            "            AllocVectorEvents();",
            "#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200) && "
            "defined(GLM_KDA_GATED_KEY_REUSE)\n"
            "            pipe_->InitBuffer(gatedKeyReuseBuf_, 64 * 128 * sizeof(half));\n"
            "#endif\n            AllocVectorEvents();",
        ),
        (
            "    TBuf<TPosition::VECCALC> gateWritebackBuf_;",
            "    TBuf<TPosition::VECCALC> gateWritebackBuf_;\n"
            "#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200) && "
            "defined(GLM_KDA_GATED_KEY_REUSE)\n"
            "    TBuf<TPosition::VECCALC> gatedKeyReuseBuf_;\n#endif",
        ),
    )
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError("gated-key reuse source anchor changed")
        source = source.replace(before, after)
    return source


def stage(source, destination, expected_sha256):
    original = source.read_bytes()
    if hashlib.sha256(original).hexdigest() != expected_sha256:
        raise ValueError("gated-key reuse source differs from qualified parent")
    candidate = transform(original.decode()).encode()
    with destination.open("xb") as output:
        output.write(candidate)
    return dict(
        source=str(source),
        source_sha256=expected_sha256,
        destination=str(destination),
        candidate_sha256=hashlib.sha256(candidate).hexdigest(),
        compiler_define=FLAG,
        extra_ub_bytes=EXTRA_UB_BYTES,
        full_kda_evaluated=False,
        serving_evaluated=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = stage(args.source, args.destination, args.expected_sha256)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
