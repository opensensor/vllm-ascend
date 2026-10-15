"""Check C4 variability with HC fusion toggled and preserve owner counters."""

from compare import ROOT, RUN, hc, run

if __name__ == "__main__":
    for action in ("off", "on"):
        hc(action, "c4-" + action)
        run(
            "/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/qsa-concurrency-benchmark.py",
            "--run-root",
            RUN,
            "--label",
            "hc-epilogues-c4-" + action,
            "--concurrency",
            4,
            "--tokens",
            256,
        )
        hc("status", "c4-" + action + "-after")
    run(ROOT / "control.py", ROOT / "final-audit-source.py", ROOT / "final-audit-after-c4")
