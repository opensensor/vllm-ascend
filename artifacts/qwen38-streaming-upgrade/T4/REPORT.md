# T4 bounded native projection candidate

Status: implemented and validated offline; hardware qualification pending.

The same integrated producer/Cube/vector loop implements gate/up and down with
M16/N128 paired low/high INT4 products. Two disjoint operand/product slots permit
Cube group j and vector consumption of j−1. Correction and FP32 group additions
stay in ascending G128 order; projection rounds to FP16 after the complete K axis.
One expert/output weight tile and three FP16 metadata banks stay resident across
row tiles. The arena reserves 140,032 UB bytes under contract v3.

This is a two-stage candidate. Producer j+1 overlap with consume j−1 is not
claimed: modulo-two metadata would alias. CO1 readback completion precedes its
next Cube write, and the final consumer drains before output. Existing padding,
empty experts, peer zeros, and partial rows retain explicit ownership/fences.

The column entry retains complete weight-bank strides and returns at most eight
N128 tiles. Down N2560 uses 8/8/4 windows; the last window uses four cores. T5
consumes/finalizes each window before the next. Gate/up remains a complete FP16
GM output because qualified nonlinear arithmetic and cross-core handoff survive.

Validation: 70 CPU tests compile and execute the actual native body under scoped
SDK stubs against an independent numerical reference. They cover all eight cores,
production widths/K, row tails, changing expert occupancy, peer zeros, guard bytes,
full and interior/tail windows, and invalid configurations. The complete T1–T7
suite passes 376 tests. Host CANN 9.1.0 ACLRTC compilation for dav-2002 succeeds
for both entrypoints; exact sources and binary are in host-compile-v3.json.

Host stubs execute synchronously. They do not prove device FP16 rounding,
asynchronous events, physical overlap, real-weight/image/MTP correctness,
whole-model memory fit, service improvement, or thermal safety. No kernel/library
was loaded or server operated. The initial host compiler did not trace driver
discovery; its no-device-open assertion is superseded by
`host-compile-driver-audit-followup.json`. T8 discovered manager-device opens in
an uncontained host build and now requires filesystem/IPC containment. No speed
multiplier is claimed.
