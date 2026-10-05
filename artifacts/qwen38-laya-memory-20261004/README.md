# Host-only Laya and QSA memory probes, 2026-10-04

These small tests ran on the Ryzen Threadripper 3970X (32 physical cores,
64 threads, 247 GiB RAM). No NPU operator, model server, or device transfer was
run. Laya 0.3.26 and CPU-only PyTorch 2.13.0 were installed in an isolated
virtual environment. The model was pinned to Hub revision
`7b928d828b7b0e022f929d9bd2e44165aa270148`. The benchmark is
[`benchmark_laya_host_memory.py`](../../tools/qwen4exp/benchmark_laya_host_memory.py).

## Resident memory and single-decision latency

The test used one synthetic technical-support classification question. The
short state was one sentence; the longer state repeated it 20 times. Times are
median wall time after one warmup, five timed calls per case. USS is memory
unique to the process and is more useful here than system-wide used RAM.

| Checkpoint | Threads | USS after load | USS after this case | Short state | Longer state |
| --- | ---: | ---: | ---: | ---: | ---: |
| English, 421M parameters | 8 | 2,799 MiB | 2,850 MiB | 203 ms | 885 ms |
| English, 421M parameters | 16 | same process | 2,866 MiB | 206 ms | 603 ms |
| Multilingual, 322M parameters | 8 | 1,746 MiB | 1,794 MiB | 79 ms | 266 ms |
| Multilingual, 322M parameters | 16 | same process | 1,807 MiB | 71 ms | 195 ms |

The multilingual checkpoint was loaded from the `multilingual` subfolder of
the pinned repository. Its cached load took 4.9 seconds on a repeat run, so
the proposed front-end service would need to keep it resident. These are CPU
measurements on this host, not published GPU timings or accuracy measurements.
The English checkpoint warned that some packaged confidence temperatures were
invalid; no confidence values were used in this benchmark.

Raw results: [`laya-english-cpu.json`](laya-english-cpu.json) and
[`laya-multilingual-cpu.json`](laya-multilingual-cpu.json).

## Batched decisions about one short state

Each synthetic question had a three-option choice. The test asked 1, 4, 8,
and 16 questions in one Laya call. Medians are from three timed calls after
warmup using the multilingual checkpoint.

| Threadripper threads | 1 decision | 4 decisions | 8 decisions | 16 decisions |
| --- | ---: | ---: | ---: | ---: |
| 8 | 73 ms | 168 ms | 286 ms | 582 ms |
| 16 | 66 ms | 125 ms | 202 ms | 385 ms |

At 16 threads the cost per decision fell from 66 ms to 24 ms at batch size 16,
while total call time rose. Unique process memory was 1,794 MiB after the
single-decision case and 1,803 MiB after the 16-decision case. This supports
batching several request-level decisions. It does not show that Laya can
replace Qwen's tensor-dependent decisions inside a layer; the test supplied
text, not Qwen hidden states, and did not compare model outputs.

Raw results: [`laya-multilingual-batched-cpu.json`](laya-multilingual-batched-cpu.json).

## Host QSA score-buffer cost

The existing exact-selector host harness was run on the same Threadripper with
2,048 queries, 10,000 FP32 groups, and top-k 512. The input score buffer was
78.125 MiB and the output index buffer was 4 MiB. Both were already in host
RAM. Each result is the median of five warmed calls; score generation and any
device transfer were outside timing.

| Pattern | 1 thread | 4 threads | 8 threads | 16 threads |
| --- | ---: | ---: | ---: | ---: |
| Random | 238.1 ms | 59.2 ms | 30.4 ms | 15.9 ms |
| Ties | — | — | 32.7 ms | 17.4 ms |

The current 310P fast-topk gate measured 26.66 ms for random scores at this
shape on one device. The eight-thread host result was already slower before
copying scores from the device and indices back. Sixteen host threads left only
10.8 ms for both transfers, synchronization, and dispatch; TP4 would also need
to share the host cores across four devices. This does not justify a host
selector integration. It remains a single-host feasibility result, not a TP4
end-to-end measurement.

Raw results: [`threadripper-host-qsa.jsonl`](threadripper-host-qsa.jsonl).

## Scope

No full 23K-token prompt, concurrent Qwen server, predictor accuracy,
document-pruning quality, or AI CPU timing was measured here. Future tests
should first collect real request and layer traces, then evaluate any batched
predictor in shadow mode before changing the Qwen path. NPU usage remains
deferred.
