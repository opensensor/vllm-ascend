"""Parse captured scheduling A/B traces offline; never initializes a model."""

from pathlib import Path

from tools.qwen4exp.profile_runtime import analyse_npu

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/resident-live-v6/native-timeline-scheduling")


def main():
    for label in ("on-v1", "off-v1"):
        for rank in (0, 3):
            root = ROOT / label / "c1-prefix" / f"rank{rank}"
            print(analyse_npu(root, processes=1), flush=True)


if __name__ == "__main__":
    main()
