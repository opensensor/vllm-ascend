# SPDX-License-Identifier: Apache-2.0
"""Supplement an immutable, response-budget-limited quality control.

Default is offline validation/dry-run. This changes neither benchmark inputs nor
promotion thresholds, and never relabels the parent trial. Live requests require
--execute after the completed parent controller has released its lease.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import inspect
import json
import math
import statistics
from pathlib import Path

import regex as re

from tools.glm_perf.streaming_protocol import (
    canonical_digest,
    evidence_schema,
    file_digest,
    summarize_trials,
    validate_evidence,
    verify_reference,
)
from tools.glm_perf.streaming_resident import (
    GuardedResidentController,
    StreamingTransport,
    _quality,
    load_candidate,
    materialize_language_inputs,
)

ORIGINAL_BUDGET = 192
REPAIR_BUDGET = 768
VARIANTS = ("reference", "candidate")
QUALITY_CONTROLS = ("language", "tool")
REGRESSION_CONTROLS = ("decode_c1", "decode_c4", "mixed")
MAX_REGRESSION = 0.05
PROSPECTIVE_DESIGN = {
    "date": "2026-10-09",
    "original_response_budget": ORIGINAL_BUDGET,
    "supplemental_response_budget": REPAIR_BUDGET,
    "scope": "Prospective repair of original reference language exhaustion; not a post hoc performance filter",
    "both_variants": True,
    "retest_language_and_tool": True,
    "benchmark_inputs_or_thresholds_changed": False,
    "parent_artifacts_preserved": True,
}
MATERIALIZED_FIELDS = (
    "materialized_token_ids",
    "materialized_prompt_tokens",
    "materialized_prompt_sha256",
    "renderer_request",
)


@dataclasses.dataclass(frozen=True)
class RepairPlan:
    candidate: object
    reference_sha256: str
    parent: dict
    workloads: dict
    materialized: dict
    files: dict[str, str]
    parent_sha256: str
    generation: str
    budget_failures: tuple[dict, ...]


def _read(path):
    return json.loads(Path(path).read_text())


def _length_finished(sample):
    reasons = [
        choice["finish_reason"]
        for event in sample.get("events", [])
        for choice in event.get("choices", [])
        if choice.get("finish_reason") is not None
    ]
    return bool(reasons) and set(reasons) == {"length"}


def budget_limited_failure(name, sample):
    """Admit truncation only; a completed wrong answer cannot become a budget repair."""
    if sample.get("completion_tokens") != ORIGINAL_BUDGET or not _length_finished(sample):
        return False
    text = sample.get("text", "").strip()
    if name == "language":
        if not text:
            return bool(sample.get("reasoning")) and not sample.get("tool_calls")
        # A capped incomplete JSON answer is explainable; full wrong JSON or
        # unrelated visible prose is deliberately outside this narrow repair.
        return text.startswith("{") and "}" not in text
    if text:
        return False
    calls = sample.get("tool_calls", [])
    if not calls:
        return bool(sample.get("reasoning"))
    for call in calls:
        if call.get("name") != "lookup_order":
            return False
        try:
            json.loads(call.get("arguments", ""))
        except ValueError:
            continue
        return False  # A complete wrong argument object is not truncation.
    return True


def validate_materialized(workloads, materialized):
    """Bind parent renderer IDs to the original full workload/tokenizer document."""
    stripped = copy.deepcopy(materialized)
    for row in stripped["workloads"]:
        for key in MATERIALIZED_FIELDS:
            row.pop(key, None)
    if stripped != workloads or not workloads.get("tokenizer_hashes"):
        raise ValueError("parent materialized workload sources changed")
    rows = {row["id"]: row for row in materialized["workloads"]}
    for name in QUALITY_CONTROLS:
        row = rows[name]
        ids = row.get("materialized_token_ids")
        if (
            row["input"].get("max_tokens") != ORIGINAL_BUDGET
            or row.get("rendered_input_sha256") != canonical_digest(row["input"])
            or not isinstance(ids, list)
            or not ids
            or any(type(value) is not int or value < 0 for value in ids)
            or row.get("materialized_prompt_tokens") != len(ids)
            or row.get("materialized_prompt_sha256") != canonical_digest(ids)
            or not isinstance(row.get("renderer_request"), dict)
        ):
            raise ValueError("original prompt IDs/rendered payload/budget are incomplete")
    return rows


def validate_parent(parent, candidate, materialized, *, artifact_root):
    validate_evidence(parent, kind="trials", bindings=candidate.identity, artifact_root=artifact_root)
    result = parent["result"]
    if parent["dispatch"] != candidate.dispatch or candidate.experimental:
        raise ValueError("parent dispatch or experimental qualification does not match")
    if any(result.get(key) is not True for key in evidence_schema()["x-complete-result-rules"]["trials"]):
        raise ValueError("parent nonquality controls are incomplete")
    if result.get("repeatability") != summarize_trials(result.get("primary", [])):
        raise ValueError("parent repeatability differs from complete primary samples")
    if result["repeatability"]["repeatability_passed"] is not True:
        raise ValueError("parent repeatability gate failed")
    samples = result.get("samples", [])
    if not samples or any(sample.get("complete") is not True for sample in samples):
        raise ValueError("parent request sample is incomplete")
    for name in REGRESSION_CONTROLS:
        rates = {
            variant: [
                sample.get("generation_tok_s")
                for sample in samples
                if sample.get("variant") == variant
                and sample.get("workload") == name
                and sample.get("generation_tok_s") is not None
            ]
            for variant in VARIANTS
        }
        if any(
            len(values) < 3
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values)
            for values in rates.values()
        ):
            raise ValueError("parent decode/mixed samples lack complete repeated throughput")
        regression = 1 - statistics.median(rates["candidate"]) / statistics.median(rates["reference"])
        declared = result.get("regressions", {}).get(name)
        if (
            type(declared) not in (int, float)
            or not math.isfinite(declared)
            or declared != regression
            or regression > MAX_REGRESSION
        ):
            raise ValueError("parent decode/mixed regression gate failed")
    rows = {row["id"]: row for row in materialized["workloads"]}
    failures = []
    for variant in VARIANTS:
        for name in QUALITY_CONTROLS:
            values = [s for s in samples if s.get("variant") == variant and s.get("workload") == name]
            if not values:
                raise ValueError("parent quality control is missing")
            for sample in values:
                if (
                    sample.get("prompt_sha256") != rows[name]["materialized_prompt_sha256"]
                    or sample.get("prompt_tokens") != rows[name]["materialized_prompt_tokens"]
                ):
                    raise ValueError("parent quality prompt differs from materialized IDs")
            passed = all(_quality(name, sample) for sample in values)
            if result.get("language_tool_quality", {}).get(variant, {}).get(name) is not passed:
                raise ValueError("parent quality outcome does not match actual samples")
            for sample in values:
                if not _quality(name, sample):
                    if not budget_limited_failure(name, sample):
                        raise ValueError("parent quality failure is not solely response-budget-limited")
                    failures.append({"variant": variant, "workload": name, "request_id": sample.get("request_id")})
    if not failures:
        raise ValueError("all parent quality controls passed; supplemental repair is unnecessary")
    if result.get("promotion_eligible") is not False or result.get("retained") != "reference":
        raise ValueError("failed parent must have retained the qualified reference")
    generation = result.get("final_generation")
    workers = result.get("final_workers", [])
    if (
        not isinstance(generation, str)
        or re.fullmatch(r"[0-9a-f]{32}", generation) is None
        or len(workers) != 4
        or {row.get("rank") for row in workers} != set(range(4))
        or any(
            row.get("generation") != generation
            or row.get("graphs_dirty") is not False
            or row.get("native_failed") is not False
            or row.get("fp16_scale_storage", {}).get("decode_version") != 984
            for row in workers
        )
    ):
        raise ValueError("parent final reference generation/graphs are incomplete")
    return tuple(failures)


def load_repair_plan(reference, parent_path, shadow_path, gates_path, workloads_path, *, materialized_path=None):
    parent_path, shadow_path, gates_path, workloads_path = map(
        lambda path: Path(path).resolve(strict=True), (parent_path, shadow_path, gates_path, workloads_path)
    )
    materialized_path = Path(materialized_path or parent_path.parent / "materialized-workloads.json").resolve(
        strict=True
    )
    candidate = load_candidate(reference, gates_path)
    shadow = _read(shadow_path)
    validate_evidence(shadow, kind="shadow", bindings=candidate.identity, artifact_root=shadow_path.parent)
    parent, workloads, materialized = _read(parent_path), _read(workloads_path), _read(materialized_path)
    validate_materialized(workloads, materialized)
    failures = validate_parent(parent, candidate, materialized, artifact_root=parent_path.parent)
    raw = next((a for a in parent["artifacts"] if Path(a["path"]).name == "trials-raw.json"), None)
    if raw is None or _read(parent_path.parent / raw["path"]).get("result") != parent["result"]:
        raise ValueError("parent trial result differs from its immutable raw artifact")
    paths = [parent_path, shadow_path, gates_path, workloads_path, materialized_path]
    for path, report in ((parent_path, parent), (shadow_path, shadow), (gates_path, _read(gates_path))):
        paths += [(path.parent / artifact["path"]).resolve(strict=True) for artifact in report["artifacts"]]
    return RepairPlan(
        candidate,
        reference["reference_sha256"],
        parent,
        workloads,
        materialized,
        {str(path): file_digest(path) for path in paths},
        file_digest(parent_path),
        parent["result"]["final_generation"],
        failures,
    )


def _write_once(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError("refusing to replace existing supplemental evidence")
    path.write_text(text)


def _verify_plan_files(plan):
    if any(file_digest(Path(path)) != digest for path, digest in plan.files.items()):
        raise ValueError("immutable parent/source evidence changed after admission")


def source_hashes():
    return {
        str(Path(path).resolve()): file_digest(Path(path))
        for path in (
            __file__,
            inspect.getsourcefile(GuardedResidentController),
            inspect.getsourcefile(validate_evidence),
        )
    }


def repair_workflow(controller, plan, *, promote=False):
    """Retest both complete chat controls under the existing guarded controller."""
    if (
        controller.candidate.identity != plan.candidate.identity
        or controller.reference["reference_sha256"] != plan.reference_sha256
        or controller.expected_generation != plan.generation
        or any(controller.output_dir.resolve() == Path(path).parent for path in plan.files)
    ):
        raise ValueError("controller does not match admitted parent or would overwrite its artifacts")
    _verify_plan_files(plan)
    hashes = source_hashes()
    design = {**PROSPECTIVE_DESIGN, "parent_sha256": plan.parent_sha256, "source_sha256": hashes}
    _write_once(controller.output_dir / "prospective-design.json", design)
    report = {
        "schema_version": 1,
        "kind": "quality_repair",
        "state": "pending",
        "identity": plan.candidate.identity,
        "parent_sha256": plan.parent_sha256,
        "parent_files_sha256": plan.files,
        "source_sha256": hashes,
        "design": design,
        "parent_quality_failures": list(plan.budget_failures),
        "parent_nonquality_gates_passed": True,
        "benchmark_rerun": False,
        "samples": [],
        "prompt_proofs": {},
        "quality": {},
        "promotion_eligible": False,
        "errors": [],
    }
    try:
        with controller.lease():
            controller.load()
            # Render the original chat once before either variant; only the
            # response cap changes afterward, so renderer inputs are identical.
            frozen = materialize_language_inputs(controller, plan.workloads)
            validate_materialized(plan.workloads, frozen)
            if frozen != plan.materialized:
                raise ValueError("renderer IDs/settings/source hashes changed from parent")
            rows = {row["id"]: copy.deepcopy(row) for row in frozen["workloads"] if row["id"] in QUALITY_CONTROLS}
            for name, row in rows.items():
                old_payload = copy.deepcopy(row["input"])
                row["input"]["max_tokens"] = REPAIR_BUDGET
                report["prompt_proofs"][name] = {
                    "token_ids": row["materialized_token_ids"],
                    "prompt_sha256": row["materialized_prompt_sha256"],
                    "rendered_input_sha256": row["rendered_input_sha256"],
                    "renderer_request_sha256": canonical_digest(row["renderer_request"]),
                    "old_response_payload_sha256": canonical_digest(old_payload),
                    "new_response_payload_sha256": canonical_digest(row["input"]),
                    "only_response_cap_changed": {**row["input"], "max_tokens": ORIGINAL_BUDGET} == old_payload,
                }
            for variant in VARIANTS:
                controller.select(variant)
                checks = {}
                for name in QUALITY_CONTROLS:
                    completed, _ = controller.request_workload(rows[name], variant=variant, cold=True)
                    report["samples"] += completed
                    if not completed or any(
                        sample.get("complete") is not True
                        or sample.get("prompt_sha256") != rows[name]["materialized_prompt_sha256"]
                        or sample.get("prompt_tokens") != rows[name]["materialized_prompt_tokens"]
                        or sample.get("reused_at_admission") != 0
                        or type(sample.get("completion_tokens")) is not int
                        or not 1 <= sample["completion_tokens"] <= REPAIR_BUDGET
                        for sample in completed
                    ):
                        raise ValueError("supplemental request is incomplete, reused or changed its prompt")
                    checks[name] = all(_quality(name, sample) for sample in completed)
                report["quality"][variant] = checks
            _verify_plan_files(plan)
            if source_hashes() != hashes:
                raise ValueError("supplemental controller/protocol source changed during repair")
            report["promotion_eligible"] = all(all(checks.values()) for checks in report["quality"].values())
            if promote and report["promotion_eligible"]:
                controller.select("candidate")
                controller.changed = False
                report["retained"] = "candidate"
            else:
                controller.restore()
                report["retained"] = "reference"
            report.update(
                state="complete" if report["promotion_eligible"] else "rejected",
                final_generation=controller.expected_generation,
                final_workers=controller.last_workers,
            )
    except BaseException as error:
        report["state"] = "failed"
        report["promotion_eligible"] = False
        report["errors"].append(repr(error))
        report["journal"] = copy.deepcopy(controller.journal)
        _write_once(controller.output_dir / "quality-repair.json", report)
        raise
    report["journal"] = copy.deepcopy(controller.journal)
    _write_once(controller.output_dir / "quality-repair.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--parent-trials", required=True, type=Path)
    parser.add_argument("--shadow-report", required=True, type=Path)
    parser.add_argument("--gates", required=True, type=Path)
    parser.add_argument("--workloads", required=True, type=Path)
    parser.add_argument("--parent-materialized", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    reference = verify_reference(_read(args.reference), args.reference.parent)
    plan = load_repair_plan(
        reference,
        args.parent_trials,
        args.shadow_report,
        args.gates,
        args.workloads,
        materialized_path=args.parent_materialized,
    )
    if not args.execute:
        print(json.dumps({"dry_run": True, "design": PROSPECTIVE_DESIGN, "parent_sha256": plan.parent_sha256}))
        return
    controller = GuardedResidentController(
        StreamingTransport(args.base_url),
        reference,
        args.reference.parent,
        args.output_dir,
        candidate=plan.candidate,
        expected_generation=plan.generation,
    )
    report = repair_workflow(controller, plan, promote=args.promote)
    print(json.dumps({"state": report["state"], "retained": report["retained"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
