"""Exercise the large affine branch in real serving, then compare uncached prefill."""

from compare import ROOT, RUN, hc, run

PROFILE = "/srv/ai/storage/artifacts/qfn-mixed-opt-20261015/profile-live.py"


if __name__ == "__main__":
    run(ROOT / "control.py", ROOT / "release-rms-intermediate.py", ROOT / "release-rms-intermediate")
    hc("verify", "hc-large-verify")
    run(
        PROFILE,
        "--run-root",
        RUN,
        "--label",
        "hc-epilogues-affine-shadow",
        "--mode",
        "spans",
        "--paragraphs",
        30,
        "--prefill-only",
    )
    hc("on", "hc-large-qualified")
    for action in ("off", "on"):
        hc(action, "hc-prefill-" + action)
        run(
            PROFILE,
            "--run-root",
            RUN,
            "--label",
            "hc-epilogues-prefill-" + action,
            "--mode",
            "spans",
            "--paragraphs",
            450,
            "--prefill-only",
        )
    run(ROOT / "finalize.py", "--run-root", RUN)
