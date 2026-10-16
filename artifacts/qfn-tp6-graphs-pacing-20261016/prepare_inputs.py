# SPDX-License-Identifier: Apache-2.0
"""Prepare code-review inputs and controls in this evidence directory."""

from pathlib import Path

from tools.qwen4exp.tp6_pacing_control import engine_source

root = Path(__file__).resolve().parent
repo = root.parent.parent
names = (
    "vllm_ascend/core/qwen_prefill_scheduler.py",
    "vllm_ascend/core/qwen_prefill_pacing.py",
    "vllm_ascend/core/prefix_mamba_scheduler.py",
    "vllm_ascend/models/qwen4_exp/w4_moe.py",
)
sections = [f"\nFile: {name}\n```python\n{(repo / name).read_text()}\n```\n" for name in names]
(root / "review-context.txt").write_text("Review this inference implementation.\n" + "".join(sections))
(root / "prefill-context.txt").write_text("".join(sections[:3]))
for name, source in (
    ("d8", engine_source(8)),
    ("c256d8", engine_source(8, 256)),
    ("restore", engine_source()),
):
    (root / f"{name}.py").write_text(source)
