# Deterministic multi-route decode candidate

This isolated candidate admits one GLM decode token into the frozen v1140 bulk
route pipeline. It covers zero through eight local contributions and retains
the existing FP16 down rounding, FP32 route weighting, and deterministic
expert-major reduction. Repeated expert IDs remain separate contributions in
their original slot order within each expert. No floating-point atomics or
weight repacking are introduced.

## Qualification status

Source preparation and portable tests are offline checks. Native compilation,
all-layer arithmetic, graph replay, full-model quality, long context, and
performance require the orchestrator's exclusive assigned hardware lane. This
candidate is disabled in the service and has no measured speedup.

The previous single-contribution graph covered 7.2% of observed calls. This
candidate accepts 100% of one-row/eight-slot shapes regardless of local route
count. That is shape coverage, not measured production traffic coverage or a
prediction of speedup. Two or more rows fall back to the qualified owner.

## Authenticated source and ABI

The builder accepts only original provenance SHA256
`9c40e60ba04bbfcabb87cbdce21828dcb018c5ca0006ee60a8c543ec8ac7d690`
and clipped gate source SHA256
`520cc1eb25e2e797c266257e6f91b6c9e3f364df470860fa4f78f00dcf5725a5`.
It authenticates every archived source, helper, original comparator binary,
and bridge before executing a compiler. Six audited single-anchor edits lower
the bulk admission threshold from 16 to zero consistently across input packing,
route input, gate/down, reducer, helper geometry, and offset-table preparation.
These select the existing prepared-bank bulk path at R1; they do not fuse new
arithmetic or reduce the expert-indexed prepared workspace layout.

The original v1140 Torch bridge namespace and tensor-pointer launch ABI remain
unchanged. Its byte identity is checked explicitly. The candidate uses a unique
`glm_sidecard_multi_route_v1_helpers` Python package and constructs native Kernel
objects from candidate binaries. It does **not** load/register another bridge
copy: `MultiRouteDecode` requires an unreduced `NpuExperts` owner that has already
loaded the original authenticated helper and bridge. This avoids package-path
cache conflicts and duplicate Torch operator/class registration.

The candidate owns separate persistent scratch and configuration dictionaries;
the reference actor retains its original native object. One row uses the
candidate, other shapes call the original actor. Do not replace the actor's
native object globally. Separate simultaneous calls require separate candidate
instances and output ownership. Graph capture and lease integration belong to
the later runtime task.

## Build and probe

Commands run from the repo root. Supply the pinned original bundle and its
authenticated source directory; never use the dirty live research source tree.
The build output must be new. `--prepare-only` writes audited sources, patch,
workspace sizes, and compiler commands without invoking CANN. Its receipt has
`build_complete=false` and cannot pass candidate admission.

```bash
python -m tools.sidecard.native_multi_route.build \
  --bundle ORIGINAL_BUNDLE \
  --native-sources ORIGINAL_NATIVE_SOURCES \
  --clipped-source PINNED_CLIPPED_CPP \
  --output NEW_CANDIDATE_BUILD \
  --cann /usr/local/Ascend/cann-9.1.0
```

After a successful build, add this explicit object to the copied gate config:

```json
{
  "multi_route_candidate": {
    "root": "NEW_CANDIDATE_BUILD",
    "manifest_sha256": "SHA256_OF_multi-route-build.json"
  }
}
```

The normal config must retain checkpoint identity and clipped gate override.
Run each rank serially with physical chips 2,3 already assigned by the
orchestrator; the probe enforces this allocation and does not modify it.

```bash
python -m tools.sidecard.multi_route_probe --config CONFIG --output NEW_REPORT --rank 0
python -m tools.sidecard.multi_route_probe --config CONFIG --output ANOTHER_REPORT --rank 1
```

For all fifteen resident layers, the probe compares padded and candidate
outputs bitwise over 21 ownership/ordering fixtures and four activation scales,
then repeats the candidate for determinism. It tests all route positions, all
local contribution counts, peer IDs, zero and non-unit weights, and duplicates.
Selected cases additionally reconstruct live one-route contributions and
perform independent CPU FP32 ordered weighting/reduction. Numerical and CPU
fingerprint observers run outside any timing; a failed gate stops immediately.

## Workspace and fallback

The route-value workspace is eight FP16 rows by 4096 columns: 65,536 bytes,
compared with 1,114,112 bytes for 17 padded tokens/eight slots. Input quantization
is one FP16 token; route sorting is eight entries rather than 136. Prepared
batch scratch remains 145 expert-indexed slots to preserve the frozen layout:
input packing 19,005,440 bytes, input scales 2,375,680 bytes, hidden packing
9,502,720 bytes, and hidden scales 1,187,840 bytes. The build receipt reports all
scratch components; allocator alignment, projection configs, sorting temporary
allocations, and kernel UB are additional. A smaller slot layout needs a separate
ABI-qualified change.

Ownership masking stays on device. Zero local routes run the reducer, which
writes zero without reading inactive route workspace. Candidate admission is
fail closed for incomplete builds, checksum drift, bridge identity, helper
identity, and unsupported math. Runtime integration must preserve a validated
original actor as fallback and reject candidate selection if arithmetic or
macro performance gates fail.
