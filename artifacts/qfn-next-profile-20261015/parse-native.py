"""Parse representative ranks offline from one bounded resident capture."""

import argparse
from pathlib import Path

from tools.qwen4exp.profile_runtime import analyse_npu

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for rank in (0, 3):
        print(analyse_npu(args.root / f"rank{rank}", processes=1), flush=True)
