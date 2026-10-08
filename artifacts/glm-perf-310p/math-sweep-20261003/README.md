# GLM 310P isolated math sweep, 2026-10-03

This is an operator-selection experiment on one Ascend 310P while the TP4
graph-enabled GLM server remained loaded. The PyTorch probes are not a model
throughput benchmark and do **not** reproduce the grouped Ascend C kernel's
UB, L1, or Cube schedule. Each reported median is ten NPU-event measurements
after three warmups for the roofline sweep, or eight after two warmups for the
bitwise sweep.

| Probe | Median |
| --- | ---: |
| 32 MiB FP16 copy | 0.285 ms |
| 32 MiB block-scale multiply | 0.350 ms |
| 4096 x 4096 FP16 matmul, M=1 / 4 / 16 / 128 | 0.839 / 0.823 / 0.265 / 0.300 ms |
| Bitwise mask, 8.4M uint8 elements, scalar / 0-D / dense mask | 16.19 / 16.19 / 16.58 ms |
| Bitwise mask, 8.4M int16 elements, scalar / 0-D / dense mask | 0.268 / 0.235 / 0.333 ms |
| Bitwise mask, 8.4M int32 elements, scalar / 0-D / dense mask | 0.398 / 0.383 / 0.558 ms |

The supported bitwise variants returned exact outputs. `torch.remainder` on
uint8 was unsupported on this stack and was removed from the completed sweep.
The large uint8 versus int16 gap is a framework/operator-path observation,
not proof that Ascend C's existing uint8 unpack uses the same implementation.
It argues for an isolated integer-field extraction candidate with bitwise
W2/W4 parity as a hard gate, not for changing the production kernel on these
numbers alone.

An isolated integer-shift extraction branch replaced the FP16 reciprocal
multiply with `AscendC::ShiftRight` on `int16_t`. The first CANN 9.1 `opc`
builds segfaulted because the system Python used by `opc` itself segfaulted
while importing NumPy. Using the working project virtualenv in `PATH` fixed
that separate build-environment issue; matched control and candidate OPP packages
then built successfully. The candidate appeared 25–35% faster on nonempty
isolated cases but failed all 18 nonempty parity cases, so the speedup is
invalid and the experimental branch was removed from the working header.

A one-hot, unit-scale hardware probe isolated the error to code extraction:
the control matched all 4096 values in each of six W2/W4 fields, whereas the
candidate matched only 256/4096 for each W4 field and 1024/4096 (fields 0–2)
or 2048/4096 (field 3) for W2. Even zero-shift field 0 failed. In the
installed CANN 9.1 `dav_m200` implementation,
`basic_api/impl/dav_m200/kernel_operator_vec_binary_scalar_impl.h:240-260`,
all `ShiftRightImpl` overloads contain only `ASCENDC_REPORT_NOT_SUPPORT`.
The device-side definition in `basic_api/impl/kernel_log.h:277` is empty, so
the candidate wrote no shifted values to its destination UB. Its reduced
latency came from omitted work, not a valid faster decode. The one-hot
probe and matched OPP packages remain on the Threadripper host under
`/home/matteius/experiments/glm-gate-a-20261002/`; the probe script is
`tmp/glm_shift_decode_probe.py`. A supported-operation decomposition or a
different packing layout is needed before pursuing this idea further. This
proves the CANN 9.1 Ascend C API call is unusable on 310P; it does not, by
itself, prove the hardware ISA lacks a shift instruction. The local
`ascend-sources/reference/asc-devkit` tree independently contains the same
`dav_m200` stub, while later-generation backends have actual `vshr`/`vshrs`
implementations. A raw-intrinsic/ISA investigation can continue offline,
without occupying NPUs, before any isolated hardware test.

A direct compiler-only intrinsic probe narrowed this further. CANN 9.1
`ccec --npu-arch=dav-2002` accepts a raw `vshr` call at syntax-check time,
but full code generation rejects it with `function type ... of 'vshr' does
not support the given target feature`. The same invocation accepts known-good
`vmuls` on DAV-M200, and a target-CPU override accepts `vshr` on DAV-C220-vec
but still rejects it on DAV-M200-vec. Thus this is a target feature gate,
not merely malformed syntax. No NPU was used for these checks.

The DAV-M200 assembler separately recognizes `VSHR.s16` as a mnemonic
(`too few operands`, versus `unrecognized instruction mnemonic` for a bogus
name). The local Ascend 610 CPU-debug instruction allowlist under
`ascend-sources/reference/asc-tools/cpudebug` also includes `vshr`, but that
table is not a 310P hardware execution test. Both are leads for ISA work,
not proof that a 310P executes the opcode. An unsupported-op experiment may
fault the device; do not attempt binary patching or execution while the
unrelated Qwen NPU test is active.

One-off ASM probe on October 3: `tmp/glm_vshr_asm_probe/` contains a
128-element INT16 ACLRTC runner, a `NOP` inline-ASM control kernel, and a
`VSHR.s16` compiler-only candidate. On the remote CANN 9.1.0 DAV-2002
compiler, the control compiled and launched one block on NPU 1, returning
all 128 values exactly (`0/128` mismatches). All four 310P3 devices then
remained `Health OK` with the Qwen worker processes still present. The
candidate did **not** launch: ACLRTC rejects its raw ASM at operand 5 with
`invalid operand for instruction`, and the same syntax also fails for
DAV-2201. Thus the candidate encoding/grammar is unresolved; this test
neither confirms nor rules out a 310P vector-shift opcode. No OPP or
production kernel was changed, and no binary opcode words were injected
into a card used by Qwen.

The earlier NZ-packed singleton-L1 grouped candidate failed its full-case
validation: W4 8-row and 32-row singleton medians were 10.27 and 40.24 ms
versus saved baseline 7.94 and 31.50 ms, and the 416-row repeated-route case
produced non-finite output. It is rejected; the live server retained the
known-good kernel.

Raw records: `glm-math-sweep-310p-20261003.json` and
`glm-bitwise-sweep-310p-20261003.json`. The scripts are in `tmp/`.
