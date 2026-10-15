"""Apply a named QSA experiment to the persistent QFN service."""

import argparse
import dataclasses
import json
import time
import uuid
from pathlib import Path

from tools.glm_perf.resident_control import Control
from tools.sidecard.qfn_resident_server import QfnResidentClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidate",
        choices=(
            "diagnostic",
            "prefix",
            "restore",
            "window_probe",
            "cold_variants",
            "cold_probe",
            "native_profile",
            "early_tp",
            "compact_probe",
            "compact_topk",
            "prefill_collectives",
            "w4_prefill_probe",
            "w4_schedule",
            "w4_fused",
            "w4_finalize",
            "w4_finalize_profile",
            "w4_owned_probe",
            "w4_owned_finalize",
            "w4_owned_profile",
            "w4_queue_probe",
            "w4_queue_finalize",
            "w4_queue_profile",
            "w4_queue_fused",
            "hc_probe",
            "hc_fused",
            "hc_profile",
            "gather_probe",
            "gather_fused",
        ),
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    w4_candidates = (
        "w4_prefill_probe",
        "w4_schedule",
        "w4_fused",
        "w4_finalize",
        "w4_finalize_profile",
        "w4_owned_probe",
        "w4_owned_finalize",
        "w4_owned_profile",
        "w4_queue_probe",
        "w4_queue_finalize",
        "w4_queue_profile",
        "w4_queue_fused",
        "hc_probe",
        "hc_fused",
        "hc_profile",
        "gather_probe",
        "gather_fused",
    )
    w4_bindings = (
        "w4_schedule",
        "w4_fused",
        "w4_finalize",
        "w4_finalize_profile",
        "w4_owned_finalize",
        "w4_owned_profile",
        "w4_queue_finalize",
        "w4_queue_profile",
        "w4_queue_fused",
        "hc_probe",
        "hc_fused",
        "hc_profile",
        "gather_probe",
        "gather_fused",
    )
    w4_finalizers = (
        "w4_fused",
        "w4_finalize",
        "w4_finalize_profile",
        "w4_owned_finalize",
        "w4_owned_profile",
        "w4_queue_finalize",
        "w4_queue_profile",
        "w4_queue_fused",
        "hc_probe",
        "hc_fused",
        "hc_profile",
        "gather_probe",
        "gather_fused",
    )
    modules = ["draft_graph_experts", "qfn_reset_barrier", "query_lens_capacity"]
    if args.candidate == "diagnostic":
        modules.extend(("qsa_admission_audit", "qfn_capture_semantics_probe"))
    elif args.candidate == "window_probe":
        modules.append("qfn_recapture_window_probe")
    elif args.candidate == "cold_probe":
        modules.append("qfn_cold_variants_probe")
    elif args.candidate in ("gather_probe", "gather_fused"):
        modules.extend(("qsa_compact_topk", "qfn_qsa_gather_staging"))
    elif args.candidate in ("hc_probe", "hc_fused", "hc_profile", "gather_probe", "gather_fused"):
        modules.extend(("qsa_compact_topk", "qfn_hc_combine"))
    elif args.candidate in w4_candidates:
        modules.extend(("qsa_compact_topk", "qfn_w4_prefill_probe"))
    elif args.candidate == "prefill_collectives":
        modules.extend(("qsa_compact_topk", "qfn_prefill_collectives"))
    elif args.candidate == "compact_probe":
        modules.append("qfn_compact_topk_probe")
    elif args.candidate == "compact_topk":
        modules.append("qsa_compact_topk")
    elif args.candidate == "early_tp":
        modules.append("qfn_early_tp_draft")
    elif args.candidate == "native_profile":
        modules.append("qfn_native_profile")
    elif args.candidate == "prefix":
        modules.append("qsa_graph_dispatch")
    if args.candidate in ("w4_finalize_profile", "w4_owned_profile", "w4_queue_profile", "hc_profile"):
        modules[-1] = "qfn_native_profile"
    source = ""
    if args.candidate in w4_candidates:
        source = (
            "import importlib\n"
            "from tools.qwen4exp import benchmark_route_finalize_310\n"
            "importlib.reload(benchmark_route_finalize_310)\n"
            "from tools.qwen4exp.resident_candidates import qfn_w4_prefill_probe\n"
            "importlib.reload(qfn_w4_prefill_probe)\n"
        )
    if args.candidate in ("hc_probe", "hc_fused", "hc_profile", "gather_probe", "gather_fused"):
        source += "from tools.qwen4exp.resident_candidates import qfn_hc_combine\nimportlib.reload(qfn_hc_combine)\n"
    if args.candidate in ("gather_probe", "gather_fused"):
        source += (
            "from tools.qwen4exp import qsa_gather_staging\n"
            "importlib.reload(qsa_gather_staging)\n"
            "from tools.qwen4exp.resident_candidates import qfn_qsa_gather_staging\n"
            "importlib.reload(qfn_qsa_gather_staging)\n"
        )
    if args.candidate == "early_tp":
        source = (
            "import importlib\n"
            "from tools.qwen4exp.resident_candidates import qfn_early_tp_draft\n"
            "importlib.reload(qfn_early_tp_draft)\n"
        )
    if args.candidate in ("prefix", "window_probe"):
        # The diagnostic imports this module before the corrected dispatcher is
        # installed. Reload only during the controller's paused preparation.
        source = (
            "import importlib\n"
            "from tools.qwen4exp.resident_candidates import qsa_graph_dispatch, qsa_admission_audit\n"
            "importlib.reload(qsa_admission_audit)\n"
            "importlib.reload(qsa_graph_dispatch)\n"
        )
    if args.candidate == "window_probe":
        source += (
            "from tools.qwen4exp.resident_candidates import qfn_recapture_window_probe\n"
            "importlib.reload(qfn_recapture_window_probe)\n"
        )
    source += "\n".join(
        f"from tools.qwen4exp.resident_candidates.{module} import replacements as factory_{index}"
        for index, module in enumerate(modules)
    )
    if args.candidate in (
        "cold_variants",
        "cold_probe",
        "native_profile",
        "early_tp",
        "compact_probe",
        "compact_topk",
        "prefill_collectives",
        "w4_prefill_probe",
        "w4_schedule",
        "w4_fused",
        "w4_finalize",
        "w4_finalize_profile",
        "w4_owned_probe",
        "w4_owned_finalize",
        "w4_owned_profile",
        "w4_queue_probe",
        "w4_queue_finalize",
        "w4_queue_profile",
        "w4_queue_fused",
        "hc_probe",
        "hc_fused",
        "hc_profile",
        "gather_probe",
        "gather_fused",
    ):
        source += (
            "\nimport importlib\n"
            "from tools.qwen4exp.resident_candidates import qsa_graph_dispatch\n"
            "importlib.reload(qsa_graph_dispatch)\n"
            "from tools.qwen4exp.resident_candidates import qsa_admission_audit\n"
            "importlib.reload(qsa_admission_audit)\n"
            "from tools.qwen4exp.resident_candidates import qsa_cold_variants\n"
            "importlib.reload(qsa_cold_variants)\n"
        )
    if args.candidate == "cold_probe":
        source += (
            "from tools.qwen4exp.resident_candidates import qfn_cold_variants_probe\n"
            "importlib.reload(qfn_cold_variants_probe)\n"
        )
        source += "factory_3 = qfn_cold_variants_probe.replacements\n"
    if args.candidate in w4_bindings:
        source += (
            "\nimport gc\n"
            "from tools.qwen4exp.resident_candidates import qfn_prefill_schedule, qfn_route_finalize\n"
            "importlib.reload(qfn_prefill_schedule)\n"
            "importlib.reload(qfn_route_finalize)\n"
            "def w4_bindings():\n"
            "    workers = [obj for obj in gc.get_objects() if type(obj).__name__ == 'NPUWorker310']\n"
            "    if len(workers) != 1:\n"
            "        raise RuntimeError('expected one resident worker during preparation')\n"
            "    worker = workers[0]\n"
            "    result = {}\n"
        )
        if args.candidate in ("w4_schedule", "w4_fused"):
            source += "    result.update(qfn_prefill_schedule.make_replacements(worker, mode='local_count'))\n"
        if args.candidate in ("w4_queue_fused", "hc_probe", "hc_fused", "hc_profile", "gather_probe", "gather_fused"):
            source += (
                "    result.update(qfn_prefill_schedule.make_replacements(worker, mode='local_count', version=3))\n"
            )
        if args.candidate in w4_finalizers:
            owner_attribute = (
                "_qfn_w4_prefill_probe_v2" if args.candidate.startswith("w4_owned") else "_qfn_w4_prefill_probe"
            )
            if args.candidate.startswith("w4_queue") or args.candidate in (
                "hc_probe",
                "hc_fused",
                "hc_profile",
                "gather_probe",
                "gather_fused",
            ):
                owner_attribute = "_qfn_w4_prefill_probe_v3"
            source += (
                f"    state = worker.{owner_attribute}\n"
                "    if not state['loaded'] or state['failed'] or not {650, 20480}.issubset(state['validated_rows']):\n"
                "        raise RuntimeError('finalization shapes have not passed the native gate')\n"
                "    result.update(qfn_route_finalize.replacements({'qwen_route_finalize_v1': state['resources']}))\n"
            )
        if args.candidate in ("hc_fused", "hc_profile", "gather_probe", "gather_fused"):
            source += "    result.update(qfn_hc_combine.make_replacements(worker))\n"
        if args.candidate in ("gather_probe", "gather_fused"):
            source += (
                "    gather_state = getattr(worker, '_qfn_qsa_gather_staging_probe', None)\n"
                "    if gather_state is not None and gather_state.get('loaded'):\n"
                "        qsa_gather_staging.upgrade_owned_wrapper(gather_state)\n"
            )
        if args.candidate == "gather_fused":
            source += "    result.update(qfn_qsa_gather_staging.make_replacements(worker))\n"
        source += "    return result\n"
    source += "\ndef replacements():\n    return {"
    source += ", ".join(f"**factory_{index}()" for index in range(len(modules)))
    if args.candidate in (
        "cold_variants",
        "native_profile",
        "early_tp",
        "compact_probe",
        "compact_topk",
        "prefill_collectives",
        "w4_prefill_probe",
        "w4_schedule",
        "w4_fused",
        "w4_finalize",
        "w4_finalize_profile",
        "w4_owned_probe",
        "w4_owned_finalize",
        "w4_owned_profile",
        "w4_queue_probe",
        "w4_queue_finalize",
        "w4_queue_profile",
        "w4_queue_fused",
        "hc_probe",
        "hc_fused",
        "hc_profile",
        "gather_probe",
        "gather_fused",
    ):
        source += ", **qsa_cold_variants.make_replacements(qualification_gate=True)"
    if args.candidate in w4_bindings:
        source += ", **w4_bindings()"
    source += "}\n"
    control = Control(generation=uuid.uuid4().hex, candidate="qsa_" + args.candidate, source=source, recapture=True)
    client = QfnResidentClient("http://127.0.0.1:8001", args.run_root / "control.token")
    result = {"control": dataclasses.asdict(control), "before": client.rpc("resident_status")}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    started = time.monotonic()
    result["after"] = client.switch(control)
    result["seconds"] = time.monotonic() - started
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "candidate": control.candidate,
                "seconds": result["seconds"],
                "pids": [row["pid"] for row in result["after"]],
            }
        )
    )


if __name__ == "__main__":
    main()
