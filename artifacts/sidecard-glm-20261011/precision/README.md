# Precision-aware resident expert placement

The `precision-resident` profile spends the compact CUDA storage savings on
more resident experts. All 288 experts in the eight W3 and twelve W2 layers
execute on the RTX. In each of the 22 W4 layers, CUDA owns experts 0–159 and
216–231; the two chips of one Ascend card retain 56 active experts apiece.
Attention, shared experts, routing, mHC, and the backbone remain on Ascend.
The checkpoint and its authenticated native partition files are unchanged.

Moved experts keep the input of their former TP rank. The live FP16 backbone
outputs are not always identical across ranks: one captured 512-token prefill
had 3,151 differing values, with a maximum difference of 0.0009765625. Sending
only rank zero's input changed quantization and failed the placement check.
The corrected profile broadcasts a clone of rank one's hidden tensor and sends
both FP16 inputs in one bounded `SCARD004` request. CUDA selects the input by
the expert's original bank, then restores route/slot order before FP32 reduction.
The model's live hidden tensors remain unchanged. Existing single-input packet
formats remain supported; peers must explicitly admit the two-input format.

CUDA holds 9,632 expert/layer pairs across all 42 main-model MoE layers.
There is no cold expert streaming. Admission accounts for native-width codes,
FP16 scales, bounded scratch, the largest reused preparation bank, and a
separate 4 GiB device reserve. The opt-in profile requires one-card compact
storage and pins both the library and the checkpoint to its boundary receipt.
The existing `seed144` profile remains available for rollback.

## Qualification and release

`boundary-gate.json` covers actual W4/W3/W2 checkpoint layers with 176/288/288
CUDA experts, including the noncontiguous W4 ownership. All 27 combinations
of 128/512/640 tokens and 1/2/8 routes passed bitwise comparisons of raw FP16
expert rows and weighted FP32 partial replies. Experimental projection paths
remain disabled. The CUDA library SHA256 is
`1a3bd8034b71cd9b5a62503766111c32c45a963fb6a56235d88397515537803f`.
Executed gate sources are preserved under `executed-source/gate/`.

The service initially loads the original 72-expert native banks on both NPU
chips. Each layer compares the new complete TP composition against the old
144+72+72 placement on three real decode inputs and one real prefill input.
Comparison tolerances are 0.002 absolute and relative. Candidate outputs are
cloned before replay because native outputs can alias reusable workspace.
Both TP ranks participate in matching comparison collectives.

Only after both qualification stages succeed does an entirely CUDA-owned
layer release its native code and scale allocations, including per-expert
and projection aliases. W4 banks remain intact. The release is idempotent
and rejects active banks, incomplete metadata, and incomplete qualification.
This avoids retaining about 10.19 GiB of unused expert storage on each chip.
It does not increase configured KV capacity or qualify longer prompts.

## Verification

The standalone sidecard suite passed 434 tests, including ownership coverage,
foreign-route rejection, noncontiguous IDs, mixed-precision memory admission,
TP comparison ordering, reusable-output protection, storage alias release,
and release failure conditions. Regression tests cover distinct TP inputs,
arbitrary token order, shared token IDs across expert owners, packet allocation
bounds, incompatible modes, and a socket exchange with the existing FP32 reply.
NPU import preflight passed before replacement. Changed-file hooks and the
explicit CUDA formatting check passed. Repository-wide formatting retains
seven unrelated failing hooks; it changed none of the candidate files. Its
143 unrelated automatic edits were restored in the isolated main checkout.

Deployment V19 passed seven operational API cases, all 168 placement checks,
and all 168 serialized/concurrent execution checks. Maximum placement error
was 0.0003383383, below the unchanged 0.002 tolerance; concurrency error was
zero. All 20 entirely CUDA-owned native banks were released on both chips:
10,942,414,848 bytes (10.19 GiB) each. Device-wide NPU usage afterward was
35,633 and 35,533 MiB. The other physical card's existing workers remained
running and were not signaled or reset.

## Serving measurements and limitations

The service runs at `http://192.168.53.187:8001/v1`, model `glm-sidecard`, on
physical NPU devices 2/3 plus the RTX. CUDA allocations are 92,837,306,368
bytes (86.46 GiB); device-wide GPU usage is 89,186 MiB, with 8,154 MiB free.
The KV budget remains 3.5 GiB per chip, prefill chunks remain bounded at 512,
and configured context remains 260,352 tokens. Full-capacity prefill is not
qualified by this change. The baseline rollback deployment is staged.

Both recorded benchmarks completed the same uncached 2,048-token prompt and
64-token decode with `ignore_eos=true`:

| Measurement | V15: seed144 | V19: precision-resident |
| --- | ---: | ---: |
| Decode tokens/s | 3.6687 | 3.7942 |
| Time to first text (seconds) | 32.7847 | 39.9739 |
| Total completion time (seconds) | 49.9570 | 56.5780 |

This is a 3.4% observed decode improvement and a 21.9% cold-prefill regression.
The larger profile is an opt-in memory tradeoff, not an overall latency win.
The default profile remains `seed144`. Do not infer GPU kernel speedups from
these end-to-end records. Boundary counter deltas include both prefill and
decode; input-transfer timing also includes pending NPU backbone work, and
exchange timing includes concurrent native dispatch.

The full API validation flag remains false because greedy repeated completion
still varies, as on V15. Operational cases and both arithmetic gates passed.
The rejected single-input attempts and input-difference diagnosis remain
documented in `rank-input-diagnostic.json`; their captures are hashed and
preserved on the hardware host. They did not qualify the new placement.
