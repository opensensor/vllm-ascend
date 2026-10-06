# SPDX-License-Identifier: Apache-2.0
"""Compose reconstruction trials with the exact current resident candidate.

Never stop workers, rewrite model weights, or change OPP paths. Every trial
restores the supplied base source, including existing fusions, in a finally
block. Native library registration remains append-only.
"""

import argparse
import ast
import hashlib
import inspect
import json
import uuid
from dataclasses import replace
from pathlib import Path

from tools.glm_perf.fused_moe_control import manifest as fused_manifest
from tools.glm_perf.reconstruction_native import PROFILES
from tools.glm_perf.resident_control import Control
from tools.glm_perf.resident_harness import ResidentClient
from tools.glm_perf.resident_native import NativeManifest
from tools.glm_perf.suite import make_groups, run_groups, summarize

BASE_FACTORY = "_reconstruction_base_replacements"
SELECTIONS = (*PROFILES, "w4a8", "int4a8", "fused_int4a8", "fused_int4a4")


def compose_source(base_source, profile, resource_name="reconstruction_v1"):
    if profile not in SELECTIONS:
        raise ValueError("unknown reconstruction selection")
    if not resource_name.startswith("reconstruction_v") or not resource_name.removeprefix("reconstruction_v").isdigit():
        raise ValueError("native resource must identify a reconstruction version")
    tree = ast.parse(base_source)
    factories = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "replacements"]
    if len(factories) != 1 or any(
        isinstance(node, ast.FunctionDef) and node.name == BASE_FACTORY for node in tree.body
    ):
        raise ValueError("base source must contain one uncomposed replacements factory")
    # Preserve the base text verbatim except for its factory's identifier.
    # Reprinting its AST would alter comments and source-based patch matching.
    lines = base_source.splitlines(keepends=True)
    factory = factories[0]
    line = lines[factory.lineno - 1]
    prefix = line[: factory.col_offset]
    tail = line[factory.col_offset :]
    if not tail.startswith("def replacements("):
        raise ValueError("base factory must use a plain replacements definition")
    lines[factory.lineno - 1] = prefix + tail.replace("def replacements(", f"def {BASE_FACTORY}(", 1)
    helper = (
        "tools.glm_perf.resident_candidates.expert_reconstruction"
        if resource_name == "reconstruction_v1"
        else f"glm_{resource_name}_helpers.expert_reconstruction"
    )
    resource_argument = "" if resource_name == "reconstruction_v1" else f", resource_name={resource_name!r}"
    return "".join(lines) + (
        "\n\ndef replacements(native_resources):\n"
        f"    from {helper} import extend_replacements\n"
        f"    return extend_replacements({BASE_FACTORY}(native_resources), "
        f"native_resources, {profile!r}{resource_argument})\n"
    )


def manifest(build_dir, gate_report, include_w3=False):
    """Bind passed binary/graph gates and worker helper hashes to one resource."""
    provenance_path = build_dir / "provenance.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text()).get("_build", {}).get("fused_moe"):
        return fused_manifest(build_dir, gate_report)
    gates = json.loads(gate_report.read_text())
    records = gates.get("records", [])
    if (
        gates.get("complete") is not True
        or not records
        or any(
            row.get("activation_pack_exact") is not True or row.get("native_reference_passed") is not True
            for row in records
        )
    ):
        raise ValueError("native manifest requires completed independent component gates")
    build_dir = build_dir.resolve(strict=True)
    options_path = build_dir / "provenance.json"
    options = json.loads(options_path.read_text()).get("_build", {}) if options_path.exists() else {}
    if options and gates.get("build_options") != options:
        raise ValueError("gate report and build disagree on native metadata options")
    if options.get("all_bits") and (
        {r.get("weight_bits") for r in records} != {2, 3, 4}
        or {r.get("weight_bits") for r in gates.get("graph_records", []) if r.get("passed")} != {2, 3, 4}
    ):
        raise ValueError("all-bit native resource requires W2/W3/W4 component and replay gates")
    if options.get("prefill_native") and (
        not any(r.get("geometry", {}).get("rows", 0) > 64 for r in records)
        or {r.get("weight_bits") for r in gates.get("prefill_graph_records", []) if r.get("passed")} != {2, 3, 4}
        or {r.get("weight_bits") for r in gates.get("lazy_metadata_records", []) if r.get("native_reference_passed")}
        != {2, 3, 4}
        or {r.get("weight_bits") for r in gates.get("moe_pipeline_records", []) if r.get("passed")} != {2, 3, 4}
    ):
        raise ValueError("prefill native resource requires large-row, replay and lazy metadata gates")
    version = options.get("version", 1)
    if type(version) is not int or version < 1:
        raise ValueError("invalid native build version")
    namespace = f"glm_reconstruction_v{version}"
    bridge_name = f"glm_reconstruction_bridge_v{version}.so"
    if gates.get("schema_version") != 1 or any(
        gates.get("binaries", {}).get(name) != hashlib.sha256((build_dir / name).read_bytes()).hexdigest()
        for name in ("glm_w4a8_pack.bin", "glm_w4a8_matmul.bin", bridge_name)
    ):
        raise ValueError("gate report does not identify these exact native binaries")
    if not gates.get("graph_records") or any(row.get("passed") is not True for row in gates["graph_records"]):
        raise ValueError("native manifest requires successful changed-input graph replay gates")
    if include_w3 and (
        not gates.get("w3_records")
        or any(row.get("passed") is not True for row in gates["w3_records"])
        or {r.get("profile") for r in gates.get("w3_graph_records", []) if r.get("passed") is True} != set(PROFILES)
        or gates.get("binaries", {}).get("reconstruction_kernel.bin")
        != hashlib.sha256((build_dir / "reconstruction_kernel.bin").read_bytes()).hexdigest()
    ):
        raise ValueError("W3 resources require gates identifying the exact decoder binary")
    here = Path(__file__).resolve().parent
    names = ["glm_w4a8_pack.bin", "glm_w4a8_matmul.bin"]
    if include_w3:
        names.append("reconstruction_kernel.bin")
    prefix = namespace + "_helpers" if options.get("helper_package") else "tools.glm_perf"
    helper_root = build_dir / prefix if options.get("helper_package") else here
    helpers = ("glm_int4.py", "reconstruction_native.py", "reconstruction_probe.py")
    adapter = (
        "expert_reconstruction.py" if options.get("helper_package") else "resident_candidates/expert_reconstruction.py"
    )
    helpers += (adapter,)
    if options.get("helper_package"):
        helpers += ("__init__.py",)
    assets = [build_dir / name for name in names] + [helper_root / name for name in helpers]
    for dependency in gates.get("support_libraries", []):
        path = Path(dependency["path"]).resolve(strict=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != dependency["sha256"]:
            raise ValueError("support library differs from the complete pipeline gate")
        assets.append(path)
    hashes = {
        prefix + "." + n.removesuffix(".py").replace("/", "."): hashlib.sha256(
            (helper_root / n).read_bytes()
        ).hexdigest()
        for n in helpers
        if n != "__init__.py"
    }
    validation = validation_source(build_dir, prefix, helper_root, hashes, namespace, include_w3, options)

    def entry(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    return NativeManifest(
        {
            "name": namespace.removeprefix("glm_"),
            "libraries": [entry(build_dir / bridge_name)],
            "assets": [entry(path) for path in assets],
            "operators": [namespace + "::launch"],
            "validation_source": validation,
        }
    )


def validation_source(build_dir, prefix, helper_root, hashes, namespace, include_w3, options=None):
    """Load a frozen private helper package, without replacing cached helpers."""
    validation = inspect.getsource(prepare_geometries) + "\n"
    pipeline_options = ""
    if options and options.get("tile_pipeline"):
        pipeline_options = f", tile_pipeline=True, output_columns={options['output_columns']!r}"
    if options and options.get("all_bits"):
        pipeline_options += ", all_bits=True"
    validation += "def prepare():\n    import importlib, hashlib, sys\n    from pathlib import Path\n"
    if options and options.get("prefill_native"):
        validation += (
            "    import torch\n"
            "    for name in ('_C_ascend::npu_w2_swiglu_310', '_C_ascend::npu_w2_route_combine_310'):\n"
            "        torch._C._dispatch_find_schema_or_throw(name, '')\n"
        )
    if prefix != "tools.glm_perf":
        validation += (
            "    import importlib.util\n"
            f"    package = {prefix!r}\n"
            f"    package_root = Path({str(helper_root)!r})\n"
            "    if package not in sys.modules:\n"
            "        spec = importlib.util.spec_from_file_location(package, package_root/'__init__.py', "
            "submodule_search_locations=[str(package_root)])\n"
            "        module = importlib.util.module_from_spec(spec)\n"
            "        sys.modules[package] = module\n"
            "        spec.loader.exec_module(module)\n"
            "    if Path(sys.modules[package].__file__).resolve() != (package_root/'__init__.py').resolve():\n"
            "        raise ValueError('native helper package already belongs to another build')\n"
        )
    validation += (
        f"    helpers = {hashes!r}\n"
        "    for name, expected in helpers.items():\n"
        "        module = importlib.import_module(name)\n"
        "        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:\n"
        "            raise ValueError('worker helper differs from manifest: ' + name)\n"
        f"    from {prefix}.glm_int4 import NativeW4Projection\n"
        f"    root = Path({str(build_dir)!r})\n"
        f"    geometries = prepare_geometries({prefix + '.reconstruction_native'!r})\n"
        "    resources = {'w4a8': NativeW4Projection(root/'glm_w4a8_pack.bin', root/'glm_w4a8_matmul.bin', "
        f"geometries, namespace={namespace!r}{pipeline_options})}}\n"
    )
    if include_w3:
        validation += (
            f"    from {prefix}.reconstruction_native import ReconstructionProjection, PROFILES\n"
            "    for profile in PROFILES:\n"
            "        resources[profile] = ReconstructionProjection(root/'reconstruction_kernel.bin', profile, "
            f"geometries, namespace={namespace!r})\n"
        )
    if options and options.get("all_bits"):
        validation += "    resources['int4a8'] = resources['w4a8']\n"
    validation += (
        "    return resources\n\ndef validate(resources):\n"
        f"    from {prefix}.reconstruction_probe import component_cases, gate_case, graph_gate\n"
        f"    weight_bits = {(2, 3, 4) if options and options.get('all_bits') else (4,)!r}\n"
        "    records = [gate_case(resources['w4a8'], geometry, repeats=2, bits=bits) "
        "for bits in weight_bits for geometry in component_cases()]\n"
        "    for bits in weight_bits:\n"
        "        graph_gate(resources['w4a8'], component_cases()[2], bits)\n"
    )
    if include_w3:
        validation += (
            "    import torch\n"
            f"    from {prefix}.reconstruction_probe import w3_gate\n"
            "    projections = {p: resources[p] for p in ('gm', 'l1', 'l1_singleton')}\n"
            "    for geometry in component_cases():\n"
            "        w3_gate(projections, geometry, torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310, 2)\n"
            "    for projection in projections.values():\n"
            "        graph_gate(projection, component_cases()[2], 3, "
            "torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310)\n"
        )
    if options and options.get("prefill_native"):
        validation += (
            f"    from {prefix}.reconstruction_probe import moe_pipeline_gate\n"
            "    for bits in weight_bits:\n"
            "        moe_pipeline_gate(resources['w4a8'], bits)\n"
        )
    return validation + (
        "    return {'passed': True, 'cases': len(records), 'weight_format': 'unchanged_signed_block32', "
        "'activation_format': 'new_group32_int8', 'model_quality': 'not_evaluated'}\n"
    )


def prepare_geometries(module_name="tools.glm_perf.reconstruction_native"):
    import importlib

    native = importlib.import_module(module_name)
    DECODE_ROWS, GLM_GEOMETRIES, ProjectionGeometry = (
        native.DECODE_ROWS,
        native.GLM_GEOMETRIES,
        native.ProjectionGeometry,
    )

    shapes = ((128, 256), (256, 512), *GLM_GEOMETRIES)
    return tuple(
        ProjectionGeometry(rows, experts, n, k) for experts in (3, 72) for n, k in shapes for rows in DECODE_ROWS
    )


def worker_identity(receipts):
    return sorted((row["rank"], row["pid"], row["weight_storage_digest"]) for row in receipts)


def run_trial(
    client,
    base_source,
    profile,
    output,
    model,
    max_tokens=64,
    resource_name="reconstruction_v1",
    require_full_coverage=False,
    quality=False,
):
    if output.exists():
        raise FileExistsError(output)
    if type(max_tokens) is not int or not 1 <= max_tokens <= 128:
        raise ValueError("trial token count must be in 1..128")
    source = compose_source(base_source, profile, resource_name)
    before = client._acknowledged_rpc("resident_status", matches=client._status_matches)
    digest = hashlib.sha256(base_source.encode()).hexdigest()
    if any(row.get("digest") != digest or row.get("mode") != "graph" or row.get("graphs_dirty") for row in before):
        raise ValueError("supplied base source is not the current clean graph candidate")
    if len({row.get("candidate") for row in before}) != 1:
        raise ValueError("workers disagree on the base candidate")
    if any("reconstruction" in row for row in before):
        raise ValueError("base still contains a reconstruction wrapper; cleanly restore it first")
    if client.request("/is_paused", method="GET").get("is_paused") is not False:
        raise ValueError("server is already paused; leave the caller-owned pause unchanged")
    if any(resource_name not in row.get("native_loaded", {}) or row.get("native_failed") for row in before):
        raise ValueError("load and validate the reconstruction native manifest first")
    saved = Control(uuid.uuid4().hex, candidate=before[0]["candidate"], source=base_source)
    result = {"profile": profile, "before": before, "restored": False}
    try:
        after = client.switch(Control(uuid.uuid4().hex, candidate=f"reconstruction_{profile}", source=source))
        if worker_identity(before) != worker_identity(after):
            raise RuntimeError("worker or weight identity changed")
        result["selected"] = after
        if any(row.get("reconstruction", {}).get("native_dispatches", 0) == 0 for row in after):
            raise RuntimeError("candidate capture did not exercise native projections on every worker")
        if require_full_coverage and any(
            row["reconstruction"].get("fallback_dispatches") != 0 or not row["reconstruction"].get("bank_coverage")
            for row in after
        ):
            raise RuntimeError("candidate capture still contains fallback expert projections")
        groups = make_groups(["short"], [])
        if quality:
            quality_cases = [case for _, cases in make_groups(["quality"], []) for case in cases]
            groups.extend(
                (f"quality_4_{index // 4}", quality_cases[index : index + 4])
                for index in range(0, len(quality_cases), 4)
            )
            groups.extend(make_groups(["tool"], []))
        bounded = [
            (name, [replace(case, max_tokens=max_tokens if case.category == "short" else 128) for case in cases])
            for name, cases in groups
        ]
        rows = run_groups(bounded, client.base_url, model, 42, timeout_s=client.timeout)
        result["summary"], result["records"] = summarize(rows), rows
        result["after_requests"] = client._acknowledged_rpc("resident_status", matches=client._status_matches)
        if require_full_coverage and any(
            row["reconstruction"].get("fallback_dispatches") != 0 for row in result["after_requests"]
        ):
            raise RuntimeError("requests still contain fallback expert projections")
        if not all(row["valid"] and row["passed"] for row in rows):
            raise RuntimeError("resident reconstruction probe failed")
    except Exception as error:
        result["trial_error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        try:
            # Factories prepare against currently installed Python methods.
            # Clear the candidate while paused before evaluating the original
            # factory, otherwise it can capture a nested native wrapper.
            if client.request("/pause?mode=wait&clear_cache=true").get("status") != "paused":
                raise RuntimeError("server did not drain before baseline restoration")
            client.switch(Control(uuid.uuid4().hex))
            restored = client.switch(saved)
            result["restored"] = worker_identity(before) == worker_identity(restored) and all(
                row.get("digest") == digest and row.get("graphs_dirty") is False and "reconstruction" not in row
                for row in restored
            )
            result["after_restoration"] = restored
            if not result["restored"]:
                raise RuntimeError("base worker/weight identity was not restored")
            if client.request("/is_paused", method="GET").get("is_paused"):
                client.resume()
        except Exception as error:
            result["restoration_error"] = str(error)
            raise
        finally:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--base-source", type=Path, required=True)
    plan.add_argument("--profile", choices=SELECTIONS, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--native-resource", default="reconstruction_v1")
    native = commands.add_parser("manifest")
    native.add_argument("--build-dir", type=Path, required=True)
    native.add_argument("--gate-report", type=Path, required=True)
    native.add_argument("--include-w3", action="store_true")
    native.add_argument("--output", type=Path, required=True)
    trial = commands.add_parser("trial")
    trial.add_argument("--base-source", type=Path, required=True)
    trial.add_argument("--profile", choices=SELECTIONS, required=True)
    trial.add_argument("--output", type=Path, required=True)
    trial.add_argument("--base-url", default="http://127.0.0.1:8001")
    trial.add_argument("--model", default="glm53-flash-selective-w3")
    trial.add_argument("--max-tokens", type=int, default=64)
    trial.add_argument("--native-resource", default="reconstruction_v1")
    trial.add_argument("--require-full-coverage", action="store_true")
    trial.add_argument("--quality", action="store_true", help="include 20 exact-answer cases and one tool call")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.command == "plan":
        args.output.write_text(compose_source(args.base_source.read_text(), args.profile, args.native_resource))
    elif args.command == "manifest":
        args.output.write_text(manifest(args.build_dir, args.gate_report, args.include_w3).payload + "\n")
    else:
        result = run_trial(
            ResidentClient(args.base_url),
            args.base_source.read_text(),
            args.profile,
            args.output,
            args.model,
            args.max_tokens,
            args.native_resource,
            args.require_full_coverage,
            args.quality,
        )
        print(json.dumps({"restored": result["restored"], "summary": result["summary"]}))


if __name__ == "__main__":
    main()
