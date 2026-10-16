# SPDX-License-Identifier: Apache-2.0
"""Authenticated opt-in component/model gates; imports no accelerator runtime.

Manifest schema 1 requires exclusive_hardware:true, physical_chips:[2,3],
cuda_device:0, coverage:[kernels|model], selected_candidates, source:{root,
manifest:{path,sha256}}, abi:{path,sha256}, npu_environment:{path,sha256},
interpreters:{npu:{path,sha256},cuda:{path,sha256}}, native_build:{root,
manifest_sha256}|null, output:absolute new directory, and cases.

Each case has name, kind:kda|mhc|multi_route|dsa|model, config:{path,sha256},
output:relative fresh path, optional source override; rank probes require
rank:0|1. Model cases add variant:reference|candidate, fixtures and
precision_contract file references; candidates name a preceding model reference
case with reference. Kernel coverage requires both ranks (DSA is one joint
case). Model coverage requires default-off reference and candidate coverage of
every selected option, matched by architecture/checkpoint/fixtures/precision.
Optional case inputs:[{path,sha256}] authenticates extra private files; include
the native_gate_override receipt whenever the config references it.

Source manifests may be existing plain path/SHA maps or schema1/files maps.
Model actor source must be config.parent/source, matching its declared source.
All paths except case outputs are absolute. Interpreters retain lexical venv
paths. SDK values come only from an authenticated private environment file;
the manifest cannot inject environment entries, commands or shell fragments.
The runner never stops services, kills processes, resets hardware or installs
OPP. Its parent must arrange the exclusive window before explicit opt-in.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

from .execution_kernel_build import (
    absolute,
    authenticate_abi,
    authenticate_build,
    authenticate_files,
    checked_sha,
    digest,
    inside,
    read_json,
)
from .execution_options import parse_options, validate_options

CANDIDATES = frozenset(("kda_decode", "mhc_pre", "mhc_post", "multi_route", "dsa_decode_graphs"))
MODULES = {
    "kda": "tools.sidecard.kda_probe",
    "mhc": "tools.sidecard.mhc_probe",
    "multi_route": "tools.sidecard.multi_route_probe",
    "dsa": "tools.sidecard.npu_dsa_probe",
    "model": "tools.sidecard.execution_model_gate",
}
RANK_KINDS = frozenset(("kda", "mhc", "multi_route"))
KERNEL_OPTIONS = {
    "kda": frozenset(("kda_decode",)),
    "mhc": frozenset(("mhc_pre", "mhc_post")),
    "multi_route": frozenset(("multi_route",)),
    "dsa": frozenset(("dsa_decode_graphs",)),
}
TOP_KEYS = frozenset(
    (
        "schema_version",
        "exclusive_hardware",
        "physical_chips",
        "cuda_device",
        "coverage",
        "selected_candidates",
        "source",
        "abi",
        "npu_environment",
        "interpreters",
        "native_build",
        "output",
        "cases",
    )
)


def file_reference(value):
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise ValueError("file reference requires exactly path and sha256")
    path = absolute(value["path"])
    if not path.is_file() or digest(path) != checked_sha(value["sha256"]):
        raise ValueError("nightly input file changed: " + str(path))
    return path


def json_reference(value):
    return read_json(file_reference(value), value["sha256"])


def source_reference(value, module):
    if type(value) is not dict or set(value) != {"root", "manifest"}:
        raise ValueError("source reference requires root and manifest")
    root = absolute(value["root"]).resolve()
    body = json_reference(value["manifest"])
    records = body.get("files") if "schema_version" in body else body
    if "schema_version" in body and (type(body["schema_version"]) is not int or body["schema_version"] != 1):
        raise ValueError("unsupported nightly source manifest")
    authenticate_files(root, records)
    required = {module.replace(".", "/") + ".py", "tools/sidecard/__init__.py"}
    if not required <= records.keys():
        raise ValueError("frozen source omits selected gate module/package")
    return root


def enabled_candidates(spec):
    options = parse_options(spec)
    result = set()
    for enabled, name in (
        (options.kda_decode, "kda_decode"),
        (options.mhc_pre_rows, "mhc_pre"),
        (options.mhc_post_rows, "mhc_post"),
        (options.multi_route, "multi_route"),
        (options.dsa_decode_graphs, "dsa_decode_graphs"),
    ):
        if enabled:
            result.add(name)
    return result


def checkpoint_identity(spec):
    root = absolute(spec.get("checkpoint"))
    reference = {"path": str(root / "native-layout.json"), "sha256": spec.get("checkpoint_sha256")}
    file_reference(reference)
    return reference["sha256"]


def private_environment(reference):
    value = json_reference(reference)
    if not all(
        type(key) is str
        and key
        and (key[0].isalpha() or key[0] == "_")
        and all(char.isascii() and (char.isalnum() or char == "_") for char in key)
        and type(body) is str
        and "\0" not in body
        for key, body in value.items()
    ):
        raise ValueError("invalid private SDK environment file")
    if value.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3" or value.get("CUDA_VISIBLE_DEVICES", "0") != "0":
        raise ValueError("private SDK environment must preserve assigned chips 2,3 and CUDA 0")
    return value


def _component(case, spec, native, selected):
    kind = case["kind"]
    if not selected & KERNEL_OPTIONS[kind]:
        raise ValueError("kernel case does not cover a selected candidate")
    if kind == "kda":
        from .kda_probe import configuration

        configuration(spec)
        if native is None:
            raise ValueError("KDA case requires authenticated execution native build")
        expected = Path(native["root"]) / native["manifest"]["addon_library"]
        if absolute(spec["library"]).resolve() != expected.resolve():
            raise ValueError("KDA addon differs from native build")
        if absolute(spec.get("base_library")).resolve() != Path(native["manifest"]["base_ops_library"]).resolve():
            raise ValueError("KDA base library differs from native build")
    elif kind == "mhc":
        validate_options(spec)
        enabled = enabled_candidates(spec)
        if not enabled & {"mhc_pre", "mhc_post"} or not enabled <= selected:
            raise ValueError("mHC case requires selected native row policies")
        if spec["execution_candidates"]["native_build"] != native["reference"]:
            raise ValueError("mHC native build differs from nightly receipt")
        checkpoint_identity(spec)
    elif kind == "multi_route":
        from .native_multi_route.adapter import authenticate_candidate

        value = spec["multi_route_candidate"]
        if "native_gate_override" not in spec:
            raise ValueError("multi-route probe requires its qualified clipped gate override")
        if set(value) != {"root", "manifest_sha256"}:
            raise ValueError("multi-route build reference differs")
        authenticate_candidate(absolute(value["root"]), checked_sha(value["manifest_sha256"]))
        checkpoint_identity(spec)
    else:
        from .npu_dsa_probe import configuration

        configuration(spec)


def _model(case, spec, source, body, models):
    from .execution_model_gate import (
        authenticate_config,
        qualify_default_reference,
        validate_fixtures,
        validate_precision,
    )

    validate_options(spec)
    authentication = authenticate_config(spec, Path(case["config"]["path"]).parent)
    validate_fixtures(json_reference(case["fixtures"]))
    validate_precision(json_reference(case["precision_contract"]))
    if (Path(case["config"]["path"]).parent / "source").resolve() != source:
        raise ValueError("model actors must use the declared frozen config.parent/source")
    if authentication["identity"]["source_sha256"] != case.get("source", body["source"])["manifest"]["sha256"]:
        raise ValueError("model diagnostic source identity differs from frozen source")
    for key, expected in (
        ("npu_python", body["interpreters"]["npu"]["path"]),
        ("npu_environment", body["npu_environment"]["path"]),
    ):
        if spec.get(key) != expected:
            raise ValueError("model owner interpreter/environment differs from nightly identity")
    enabled = enabled_candidates(spec)
    if parse_options(spec).native_required and spec["execution_candidates"]["native_build"] != body["native_build"]:
        raise ValueError("model native build differs from selected component build")
    identity = (
        authentication["architecture"],
        checkpoint_identity(spec),
        case["fixtures"]["sha256"],
        case["precision_contract"]["sha256"],
    )
    if case["variant"] == "reference":
        if enabled or "reference" in case:
            raise ValueError("model reference must have all execution candidates disabled")
        qualify_default_reference(authentication, spec, Path(case["config"]["path"]).parent)
    elif case["variant"] == "candidate":
        previous = models.get(case.get("reference"))
        if previous is None or previous["variant"] != "reference" or previous["identity"] != identity:
            raise ValueError("candidate requires a preceding matched default-off reference case")
        if not enabled or not enabled <= set(body["selected_candidates"]):
            raise ValueError("model candidate options do not match selected nightly scope")
    else:
        raise ValueError("unsupported model case variant")
    models[case["name"]] = dict(variant=case["variant"], identity=identity, enabled=enabled)


def validate_manifest(path, *, expected_sha256=None, allow_output=False, completed=()):
    """Complete read-only admission before the first subprocess or output mkdir."""
    path = absolute(str(path))
    checksum = digest(path) if expected_sha256 is None else expected_sha256
    body = read_json(path, checksum)
    if (
        set(body) != TOP_KEYS
        or type(body.get("schema_version")) is not int
        or body["schema_version"] != 1
        or body.get("exclusive_hardware") is not True
        or body.get("physical_chips") != [2, 3]
        or type(body.get("cuda_device")) is not int
        or body["cuda_device"] != 0
    ):
        raise ValueError("explicit schema1 exclusive chips [2,3]/CUDA0 nightly manifest required")
    for key, allowed in (("coverage", {"kernels", "model"}), ("selected_candidates", CANDIDATES)):
        values = body[key]
        if (
            type(values) is not list
            or not values
            or any(type(value) is not str for value in values)
            or len(values) != len(set(values))
            or not set(values) <= allowed
        ):
            raise ValueError("unsupported, empty or duplicate nightly " + key)
    abi = authenticate_abi(file_reference(body["abi"]), body["abi"]["sha256"])
    if type(body["interpreters"]) is not dict or set(body["interpreters"]) != {"npu", "cuda"}:
        raise ValueError("pinned NPU and CUDA interpreters are required")
    for value in body["interpreters"].values():
        interpreter = file_reference(value)  # Keep lexical venv paths when constructing argv.
        if not os.access(interpreter, os.X_OK):
            raise ValueError("pinned interpreter is not executable")
    if body["interpreters"]["npu"]["path"] != abi["paths"]["python_executable"]:
        raise ValueError("NPU interpreter differs from pinned native ABI")
    environment = private_environment(body["npu_environment"])
    selected = set(body["selected_candidates"])
    native = None
    if selected & {"kda_decode", "mhc_pre", "mhc_post"}:
        reference = body["native_build"]
        if type(reference) is not dict or set(reference) != {"root", "manifest_sha256"}:
            raise ValueError("selected native kernels require an execution build receipt")
        root = absolute(reference["root"])
        manifest = authenticate_build(root, checked_sha(reference["manifest_sha256"]))
        if manifest["abi_manifest_sha256"] != body["abi"]["sha256"]:
            raise ValueError("nightly ABI differs from native build")
        vendor = (root / manifest["opp_path"]).resolve()
        inherited = [Path(value).resolve() for value in environment.get("CANN_CUSTOM_OPP_PATH", "").split(":") if value]
        if vendor not in inherited:
            raise ValueError("authenticated private SDK environment lacks native vendor path")
        native = dict(reference=reference, root=root, manifest=manifest)
    elif body["native_build"] is not None:
        raise ValueError("unused nightly native build is forbidden")
    output = absolute(body["output"]).resolve()
    if output.exists() and not allow_output:
        raise ValueError("nightly output must be a new directory")
    if type(body["cases"]) is not list or not body["cases"]:
        raise ValueError("explicit nightly cases required")
    forbidden = [Path(abi["paths"][name]).resolve() for name in ("cann_root", "torch_root", "torch_npu_root")]
    if native is not None:
        forbidden.append(native["root"].resolve())
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in forbidden):
        raise ValueError("nightly output overlaps immutable native/installed SDK roots")
    names, kernels, kernel_identities, models, admitted = set(), {}, {}, {}, []
    outputs = [output / "logs", output / "execution-nightly.json"]
    for case in body["cases"]:
        if type(case) is not dict:
            raise ValueError("nightly case must be an object")
        kind, name = case.get("kind"), case.get("name")
        if (
            type(kind) is not str
            or kind not in MODULES
            or type(name) is not str
            or not name
            or any(not char.isascii() or not (char.isalnum() or char in "-_") for char in name)
            or name in names
        ):
            raise ValueError("unsupported case kind or nonunique safe name")
        required = {"name", "kind", "config", "output"}
        if kind in RANK_KINDS:
            required.add("rank")
            if type(case.get("rank")) is not int or case["rank"] not in (0, 1):
                raise ValueError("rank probe must select local rank zero or one")
        if kind == "model":
            if case.get("variant") not in ("reference", "candidate"):
                raise ValueError("unsupported model variant")
            required |= {"variant", "fixtures", "precision_contract"}
            if case.get("variant") == "candidate":
                required.add("reference")
                if type(case.get("reference")) is not str:
                    raise ValueError("candidate reference must name a preceding case")
        if not required <= set(case) or set(case) - required - {"source", "inputs"}:
            raise ValueError("case has missing or unsupported fields")
        destination = inside(output, case["output"])
        if kind != "multi_route" and destination.suffix != ".json":
            raise ValueError("file-producing gate requires a JSON output path")
        reserved = [destination]
        if kind == "dsa":
            reserved.append(destination.parent / (destination.stem + "-owners"))
        elif kind == "model":
            reserved.append(destination.with_suffix(destination.suffix + ".private"))
        if name not in completed and any(target.exists() for target in reserved):
            raise ValueError("nightly gate output/owner directory already exists")
        if any(
            left == right or left in right.parents or right in left.parents for left in outputs for right in reserved
        ):
            raise ValueError("nightly gate outputs overlap")
        source = source_reference(case.get("source", body["source"]), MODULES[kind])
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("nightly outputs must be separate from frozen source")
        spec = json_reference(case["config"])
        extra = case.get("inputs", [])
        if type(extra) is not list:
            raise ValueError("case inputs must be file-reference lists")
        extra_paths = [file_reference(reference).resolve() for reference in extra]
        if len(extra_paths) != len(set(extra_paths)):
            raise ValueError("duplicate extra input paths")
        if "native_gate_override" in spec:
            override_path = absolute(spec["native_gate_override"])
            if override_path.resolve() not in extra_paths:
                raise ValueError("native gate override lacks nightly file authentication")
            override = read_json(override_path, digest(override_path))
            expected_gates = {"glm_fused_gate_up.bin", "glm_fused_gate_up_w4.bin"}
            if override.get("swiglu_limit") != 10.0 or set(override.get("native_gate_files", {})) != expected_gates:
                raise ValueError("native gate override differs from qualified clipped contract")
            if "checkpoint" in spec:
                layout_path = absolute(spec["checkpoint"]) / "native-layout.json"
                layout = read_json(layout_path, checked_sha(spec["checkpoint_sha256"]))
                if override.get("original_provenance_sha256") != layout.get("kernel_provenance_sha256"):
                    raise ValueError("native gate override differs from checkpoint provenance")
            for filename, sha in override.get("native_gate_files", {}).items():
                file_reference(dict(path=str(inside(override_path.parent, filename)), sha256=sha))
            if "cuda_library" in override:
                file_reference(dict(path=override["cuda_library"], sha256=override["cuda_library_sha256"]))
        if kind == "model":
            if "model" not in body["coverage"]:
                raise ValueError("model case outside requested coverage")
            _model(case, spec, source, body, models)
        else:
            if "kernels" not in body["coverage"]:
                raise ValueError("kernel case outside requested coverage")
            _component(case, spec, native, selected)
            ranks = kernels.setdefault(kind, set())
            identity = (case["config"]["sha256"], case.get("source", body["source"])["manifest"]["sha256"])
            if kind in kernel_identities and kernel_identities[kind] != identity:
                raise ValueError("component rank pair config/source identities differ")
            kernel_identities[kind] = identity
            rank = case.get("rank", "joint")
            if rank in ranks:
                raise ValueError("duplicate component/rank coverage")
            ranks.add(rank)
        names.add(name)
        outputs.extend(reserved)
        admitted.append(dict(case=case, source=source, config=spec, output=destination))
    if "kernels" in body["coverage"]:
        for kind, options in KERNEL_OPTIONS.items():
            if selected & options and kernels.get(kind) != ({0, 1} if kind in RANK_KINDS else {"joint"}):
                raise ValueError("selected component lacks complete both-rank coverage: " + kind)
    if "model" in body["coverage"]:
        covered = set().union(*(value["enabled"] for value in models.values())) if models else set()
        if covered != selected:
            raise ValueError("selected candidates lack matched complete model coverage")
    return dict(manifest=body, sha256=checksum, cases=admitted, environment=environment)


def command(admitted, body, references):
    case = admitted["case"]
    interpreter = body["interpreters"]["cuda" if case["kind"] == "model" else "npu"]["path"]
    argv = [
        interpreter,
        "-m",
        MODULES[case["kind"]],
        "--config",
        case["config"]["path"],
        "--output",
        str(admitted["output"]),
    ]
    if case["kind"] in RANK_KINDS:
        argv += ["--rank", str(case["rank"])]
    if case["kind"] == "model":
        argv += ["--fixtures", case["fixtures"]["path"], "--precision-contract", case["precision_contract"]["path"]]
        if case["variant"] == "candidate":
            reference = references[case["reference"]]
            file_reference(reference)
            argv += ["--reference", reference["path"], "--reference-sha256", reference["sha256"]]
    return argv


def validate_result(admitted):
    case = admitted["case"]
    path = admitted["output"] / "summary.json" if case["kind"] == "multi_route" else admitted["output"]
    result = read_json(path, digest(path))
    flag = "ok" if case["kind"] in ("kda", "dsa") else "passed"
    if result.get(flag) is not True:
        raise ValueError("selected execution gate did not pass: " + case["name"])
    if case["kind"] in RANK_KINDS:
        if result.get("rank") != case["rank"] or result.get("physical_chip") != case["rank"] + 2:
            raise ValueError("gate returned different rank/chip ownership")
    if case["kind"] == "dsa":
        owners = result.get("owners", [])
        if (
            result.get("coordinated_tp2") is not True
            or result.get("physical_chips") != [2, 3]
            or sorted(owner.get("report", {}).get("rank", -1) for owner in owners) != [0, 1]
            or not all(owner.get("report", {}).get("ok") is True for owner in owners)
        ):
            raise ValueError("DSA gate lacks joint both-rank evidence")
    if case["kind"] == "model":
        if case["variant"] == "reference" and result.get("reference_qualified") is not True:
            raise ValueError("model reference did not qualify")
        if len(result.get("workloads", [])) != 3 or {row.get("name") for row in result.get("workloads", [])} != {
            "2K",
            "8K",
            "full-code",
        }:
            raise ValueError("model gate omitted complete workload coverage")
    return dict(path=str(path), sha256=digest(path))


def run(path):
    plan = validate_manifest(path)
    body, checksum = plan["manifest"], plan["sha256"]
    output = Path(body["output"]).resolve()
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    (output / "logs").mkdir(mode=0o700)
    report = dict(
        schema_version=1,
        passed=False,
        exclusive_hardware=True,
        physical_chips=[2, 3],
        cuda_device=0,
        manifest_sha256=checksum,
        coverage=body["coverage"],
        selected_candidates=body["selected_candidates"],
        identities=dict(
            source_manifest_sha256=body["source"]["manifest"]["sha256"],
            abi_manifest_sha256=body["abi"]["sha256"],
            npu_environment_sha256=body["npu_environment"]["sha256"],
            interpreter_sha256={name: value["sha256"] for name, value in body["interpreters"].items()},
            native_build_sha256=body["native_build"]["manifest_sha256"] if body["native_build"] else None,
        ),
        scope="component/model gates only; no service performance or release promotion",
        cases=[],
    )
    references, completed = {}, []
    try:
        for index, admitted in enumerate(plan["cases"]):
            validate_manifest(path, expected_sha256=checksum, allow_output=True, completed=completed)
            for reference in references.values():
                file_reference(reference)
            case = admitted["case"]
            environment = dict(os.environ)
            environment.update(plan["environment"])
            environment["ASCEND_RT_VISIBLE_DEVICES"], environment["CUDA_VISIBLE_DEVICES"] = "2,3", "0"
            inherited = plan["environment"].get("PYTHONPATH", "")
            environment["PYTHONPATH"] = str(admitted["source"]) + (os.pathsep + inherited if inherited else "")
            argv = command(admitted, body, references)
            log = output / "logs" / f"{index:02d}-{case['name']}.log"
            admitted["output"].parent.mkdir(parents=True, exist_ok=True)
            with log.open("xb") as stream:
                result = subprocess.run(
                    argv, cwd=admitted["source"], env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False
                )
            row = dict(
                name=case["name"],
                kind=case["kind"],
                argv=argv,
                config_sha256=case["config"]["sha256"],
                source_manifest_sha256=case.get("source", body["source"])["manifest"]["sha256"],
                returncode=result.returncode,
                log=dict(path=str(log), sha256=digest(log)),
            )
            report["cases"].append(row)
            if result.returncode != 0:
                raise RuntimeError("execution gate subprocess failed: " + case["name"])
            reference = validate_result(admitted)
            row["receipt"] = reference
            references[case["name"]] = reference
            completed.append(case["name"])
        validate_manifest(path, expected_sha256=checksum, allow_output=True, completed=completed)
        for reference in references.values():
            file_reference(reference)
        report["passed"] = True
    except Exception as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        (output / "execution-nightly.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        plan = validate_manifest(args.manifest)
        report = dict(validated=True, cases=len(plan["cases"]), hardware_started=False)
    else:
        result = run(args.manifest)
        report = dict(passed=result["passed"], cases=len(result["cases"]))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
