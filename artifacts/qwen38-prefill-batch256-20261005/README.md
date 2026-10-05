# Qwen native-W4 larger expert batch, 310P service gate

The source candidate retains the selected 1,536-token grouped-expert chunk by
default. An explicit `ascend_expert_quantization.grouped_prefill_chunk_tokens`
override permits up to 2,560 tokens, with matching 25,600-route limits in the
native INT4 pack and matmul adapters and host tilers. The candidate shifts the
large-M projection schedule switch from 128 to 256 average route slots per
expert. The four-rank candidate ran on port 8001 as
`qwen38-w4-batch2560-cann-finalize-candidate` and was stopped when the user
handed the NPUs to another agent.

The first isolated three-operator test package is
`/srv/ai/src/qwen38-prefill-batch256-opp-20261005/vendors/qwen_w4_batch256_probe_transformer`;
its installer SHA-256 is
`63d7002d8e8a1397ef6f13c1bf7e9be2466ed510eea61f1a725b7b0300adf56c`.
The isolated runtime is `/srv/ai/src/qwen38-prefill-batch256-runtime-20261005`;
its rebuilt binding SHA-256 is
`fb859c36a57a46df1bf0c8f1a22264c7bc109b9f87a0dc81de66d081d810dc39`.
The compiled candidate kernel source SHA-256 is
`5698b83025c599cd460388562c003a26ca6111c16c8900f056e2eed3fbda7f64`.

For serving, a clean build packaged all five required operators together at
`/srv/ai/src/qwen38-prefill-batch256-coherent-opp-20261005/vendors/qwen38_batch256_coherent_transformer`.
Its installer SHA-256 is
`04585860b095f399bf2609279a5d9f96eab69d1b1fed1b7ebf974db609c66e53`.
The coherent package's two native matmul kernel object hashes exactly match
the measured threshold-256 package and differ from the rebuilt threshold-128
control. Its first host API library exports the required matmul, pack,
SwiGLU-pack, down-reduce, and recurrent workspace/launch symbols. The
recurrent kernel configuration includes FP16 and FP32 state variants.

## Capacity and one-layer results

Three isolated 25,600-row operator tests passed on one 310P: native grouped
matmul, activation pack, and experimental SwiGLU-pack capacity. The latter
checks capacity only; prior large-row exact-parity failure still keeps the
custom SwiGLU pack out of the serving default. The [test log](operator-gates-batch256.log)
records the exact Python selectors.

The [2,560-token sweep](layer-batch256-cann-v2.json) used real layer-0 expert
weights at TP rank 0, one fixed router result from seeded synthetic
activations, built-in FP16 SwiGLU, and the CANN route finalizer. It includes
grouped dispatch, both native projections, activation packing, and finalizing;
it excludes router evaluation, shared experts, TP reduction, attention, and
service scheduling. All five chunk sizes produced bitwise-identical output.

| Chunk tokens | Calls per projection | Median for 2,560 tokens |
| ---: | ---: | ---: |
| 1,536 | 2 | 51.40 ms |
| 1,638 | 2 | 51.52 ms |
| 1,639 | 2 | 51.57 ms |
| 2,048 | 2 | 50.77 ms |
| 2,560 | 1 | 45.83 ms |

The 2,560-token chunk reduced this layer partial by **10.8%** against the
selected split. The prior 1,638/1,639 cliff is absent in this candidate.

The [23,410-token layer sweep](layer-long-batch256-cann-v2.json) compared the
selected and larger chunks on the same route IDs. Both outputs were bitwise
identical. The 1,536-token split took **457.74 ms** over 16 projection pairs;
2,560-token chunks took **420.97 ms** over ten pairs, an **8.0%** reduction.
This is a one-layer TP partial, not a prediction of model TTFT.

## Build and service gate

An incremental `csrc/build.sh --pkg` after changing the kernel threshold
updated the packaged source but reused identical compiled `.o` files. For the
threshold-128 control, the generated source-copy marker, both matmul generation
markers, and the two matmul objects were removed from the isolated build tree
before rebuilding. The rebuilt control objects differ in SHA-256 from the
threshold-256 candidate objects. This check is necessary before attributing
any measured difference to the tile threshold.

The threshold-128 control package is isolated at
`/srv/ai/src/qwen38-prefill-batch128-cap256-opp-20261005`. The
[control sweep](layer-batch128-control-cann-v2.json) used the same 25,600-route
cap, real weights, route IDs, and output hash as the threshold-256 candidate.
All outputs were bitwise identical. The threshold-256 schedule left the
1,536-token split unchanged (51.40 versus 51.38 ms), but reduced the
2,048-token case from 63.51 to 50.77 ms and the 2,560-token case from 58.95
to 45.83 ms. The larger-chunk gain depends on the new schedule as well as
amortizing more tokens per projection pair.

The [isolated launcher diff](candidate-launcher.diff) selects that coherent
package and runtime, raises the scheduler batch to 2,560, and adds the
`grouped_prefill_chunk_tokens=2560` model override while retaining built-in
SwiGLU, the CANN finalizer, MTP2, TP4/EP4, fixed cache, and two decode graphs.
Its [`--show` command](server-command.txt), syntax, and `--check-runtime` all
passed static checks. The launcher is staged on the Threadripper at
`/srv/ai/src/qwen38-prefill-batch256-runtime-20261005/examples/start_qwen38_flash_next_w4_310p_batch2560.sh`.
The [service wrapper](start-batch2560-service.sh) launched the candidate in a
persistent `qwen38_batch2560_20261005` tmux session. The server reported
1,068,936 cache tokens and maximum concurrency of 4.08 for 262,144-token
requests; both configured decode graphs captured. These are startup checks,
not a measured four-request workload.

Three [matched cold prompts](service-batch2560.jsonl) used the same 23,410-token
case IDs and prompt hashes as the prior
[1,536-token CANN-finalizer service](../qwen38-builtin-finalize-20261005/service-cann-v2.jsonl).
All six requests reported zero cached tokens and generated 32 tokens. The
saved prior run was on a different service instance and day, so device and run
order are not fully controlled.

| Case | Prior TTFT | 2,560-token TTFT | Saved | Text |
| ---: | ---: | ---: | ---: | --- |
| 0 | 64.863 s | 62.260 s | 2.603 s | Opening phrase changed |
| 1 | 63.861 s | 61.840 s | 2.021 s | Exact match |
| 2 | 64.151 s | 61.995 s | 2.156 s | Exact match |
| **Mean** | **64.292 s** | **62.032 s** | **2.260 s (3.52%)** | |

Effective prompt rate rose from **364.1 to 377.4 tok/s**. Case 0 said “code
excerpts” instead of “repository excerpt”; the rest of the 32-token output
matched. The candidate still needs a sustained thermal and concurrent-request
test, plus broader quality checks, before changing the source default from
1,536 to 2,560 tokens. The two stable 32-token decode cases were essentially
flat: 30.28 to 29.78 and 30.21 to 29.90 tok/s. The first case varied from
22.26 to 26.44 tok/s, so these short runs do not establish a decode-speed
gain. A later [512-token decode profile](live-decode-profile/README.md)
measured 30.91 tok/s and sampled the model worker and CANN system counters.
The later [four-rank named prefill trace](named-prefill-trace/README.md)
captured four opening chunks. Its device occupancy was 95.66–95.88% across
the ranks; native W4 projections, QSA gathers, and large hyperconnection-state
casts are the leading measured targets. The rank-1 host copy API total is
dominated by a small number of long calls whose precise source still needs a
timestamped CPU correlation.
