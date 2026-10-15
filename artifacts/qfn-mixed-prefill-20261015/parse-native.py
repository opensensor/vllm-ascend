"""Parse two representative ranks offline without loading a model."""

from pathlib import Path

from tools.qwen4exp.profile_runtime import analyse_npu

if __name__ == "__main__":
    root = Path("/srv/ai/storage/artifacts/qfn-mixed-prefill-20261015/mixed-native-v2/native")
    for rank in (0, 3):
        print(analyse_npu(root / f"rank{rank}", processes=1), flush=True)
