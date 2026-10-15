"""Parse all recorded ranks sequentially on CPU, leaving serving workers alone."""

from pathlib import Path

from tools.qwen4exp.profile_runtime import analyse_npu

ROOT = Path("/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")
TRACES = ROOT / "qsa-resident-v1/native-timeline-c4-coverage-v2"


def main():
    for tag in ("c1-prefix", "c4-prefix", "c1-general"):
        for rank in range(4):
            directory = TRACES / tag / f"rank{rank}"
            print("Parsing", tag, rank, flush=True)
            print(analyse_npu(directory, processes=1), flush=True)


if __name__ == "__main__":
    main()
