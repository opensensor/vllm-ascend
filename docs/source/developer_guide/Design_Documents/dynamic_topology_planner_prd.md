# PRD: Dynamic Topology Planner for Heterogeneous Ascend Hosts

| Field | Value |
| --- | --- |
| Status | Draft; future implementation |
| Date | 2026-09-22 |
| Product | Automatic model placement and parallel-topology planning for vLLM Ascend |
| First reference model | Qwen3.8-Flash-Next W8A8_DYNAMIC with vision |
| First reference host | Three Atlas 300I Duo cards: six independent 48 GB Ascend 310P devices |
| General target | One or more Ascend cards with non-uniform divisibility, memory, and interconnect constraints |
| Companion documents | `qwen38_flash_next_1m_310p_prd.md`, `dynamic_chunked_pipeline_parallel.md` |

## 1. Problem statement

vLLM generally exposes a small set of global parallel sizes, but real models do
not have one uniformly shardable topology. Attention heads, KV heads, linear
attention state heads, routed experts, shared experts, vision encoders, hybrid
state, and KV cache can each have different divisibility and communication
requirements.

This becomes visible on a six-device host. Qwen3.8-Flash-Next has 24 full
attention query heads, 2 KV heads, 16 linear-attention key heads, 48
linear-attention value heads, and 512 routed experts. A global TP6 layout fits
24 and 48, but not 2, 16, or 512. The current adapter consequently targets TP4
and rejects TP6 even though a third physical card would add 50% more NPU memory
and compute.

The product must replace the assumption that every component uses the same
parallel axis with a deterministic planner. The planner will discover hardware
and model constraints, enumerate valid component-level layouts, estimate memory
and communication costs, and produce an auditable execution plan. It must favor
correctness and safe fallback over using every available device.

## 2. Goals

1. Generate a valid placement plan for the devices actually available to the
   user, including non-power-of-two device counts.
2. Choose sharding independently for attention, KV, linear attention, MoE,
   pipeline stages, multimodal encoders, hybrid state, and caches.
3. Support exact, replicated, padded, uneven, subgroup, and pipeline placement
   strategies through one plan representation.
4. Make six 310P devices a supported topology for Qwen3.8-Flash-Next without
   silently changing model semantics.
5. Predict per-rank persistent and peak memory before allocating model weights.
6. Minimize CPU offload when additional NPU capacity can safely hold weights or
   cache, while retaining offload as a planned fallback.
7. Emit the selected plan, rejected alternatives, capacity estimates, topology
   assumptions, and measured deviations in a durable run artifact.
8. Preserve explicit user overrides and existing known-good TP4 behavior.
9. Provide reusable planning interfaces so future models declare constraints
   instead of adding host-specific conditionals to model code.

## 3. Non-goals for the first release

- Live device hot-plug or reshaping a running engine.
- Transparent pooling of independent NPU memory domains.
- Cross-host elastic scheduling or cluster autoscaling.
- Migrating active requests between topology plans.
- Searching arbitrary kernel implementations at request time.
- Guaranteeing that all detected devices improve latency or throughput.
- Removing manual TP, PP, EP, DCP, or offload configuration.
- Automatically accepting an accuracy-changing padded layout without explicit
  masking and parity evidence.
- Treating a successful capacity calculation as inference validation.

## 4. Users and scenarios

### 4.1 End user with a fixed host

A user has one to three Ascend cards and wants the best safe configuration for a
model without reasoning about every head count and expert axis. They select
`auto` topology, inspect the proposed plan, and can pin or reject individual
choices.

### 4.2 Model adapter author

An adapter author declares model components, tensor axes, legal placement
strategies, state ownership, and kernel capabilities. The common planner
handles hardware inventory and candidate selection.

### 4.3 Operator requiring reproducibility

An operator freezes a generated plan to JSON or YAML and reuses it on equivalent
hardware. Startup fails clearly if the checkpoint, software, or hardware
fingerprint no longer matches.

### 4.4 Capacity exploration

An operator asks for plans optimized for one of the following objectives:

- maximum context with vision enabled;
- lowest decode latency;
- highest single-instance throughput;
- minimum host-memory traffic;
- maximum number of replicas.

## 5. Terminology

- **Card**: a physical PCIe card. One card may expose multiple NPU devices.
- **Device/rank**: an independently addressed NPU memory and execution domain.
- **World**: all ranks made available to one engine invocation.
- **Component**: a model subsystem with one placement contract, such as QSA,
  GDN, routed experts, the vision encoder, or an embedding table.
- **Placement strategy**: exact sharding, replication, padding, uneven sharding,
  subgroup sharding, pipeline placement, host offload, or a combination.
- **Plan**: a versioned mapping from model components and state to ranks,
  collectives, cache ownership, and memory budgets.
- **Static plan**: selected from configuration, checkpoint metadata, and
  hardware inventory without executing model kernels.
- **Calibrated plan**: a static plan whose cost estimates were refined by
  optional startup profiling.

## 6. Reference topology and model constraints

### 6.1 Six-device host

The first non-uniform target is three Atlas 300I Duo cards exposing six 48 GB
310P devices. Discovery must preserve:

- device-to-card identity;
- PCIe root complex and NUMA node;
- within-card versus cross-card links;
- free and total memory per device;
- supported dtypes, kernels, collectives, and firmware capabilities;
- host RAM, pinned-host allocation limit, NUMA locality, and swap policy.

Device ordinals alone are not a topology description.

### 6.2 Qwen3.8-Flash-Next

| Component | Model geometry | TP6 implication |
| --- | ---: | --- |
| Decoder layers | 48 | divisible across PP2, PP3, PP4, PP6, subject to state boundaries |
| Full-attention query heads | 24 | exact TP6: 4 heads/rank |
| Full-attention KV heads | 2 | replicate each KV head across three rank groups |
| GDN key heads | 16 | uneven `3/3/3/3/2/2`, pad to 18, or subgroup placement |
| GDN value heads | 48 | exact TP6: 8 heads/rank |
| Routed experts | 512 | uneven `86/86/85/85/85/85` or pad to 516 |
| Experts selected per token | 10 | routing and ownership must preserve global top-k semantics |
| QSA indexer query heads | 4 | replicate or use component-local subgroups |
| QSA indexer KV heads | 1 | replicated unless a distributed index design is selected |
| Vision encoder | checkpoint-defined | replicate, shard, or assign to a subgroup from measured cost |

The measured TP4 model load used approximately 33.3 GB per rank before
offload. Ideal aggregate division over six ranks would be approximately 22.2 GB
per rank, before imbalance, replicated tensors, padding, cache, vision workspace,
and allocator overhead. This estimate motivates TP6 work but is not a fit proof.

## 7. Product requirements

### R1. Hardware discovery

- Produce a versioned hardware inventory before model construction.
- Identify physical cards separately from logical NPU devices.
- Record memory, NUMA, PCIe, link, driver, firmware, CANN, torch-npu, and HCCL
  information needed by the cost model.
- Discovery must be read-only and must not reserve NPU memory.
- Permit an offline inventory fixture for planning and CI without hardware.
- Reject duplicate, inaccessible, or partially initialized ranks.

### R2. Model topology description

- Introduce a declarative topology description independent of a specific
  hardware backend implementation.
- Describe every shardable tensor axis, state tensor, cache group, collective,
  and kernel constraint using named fields rather than model-name checks.
- Express divisibility, alignment, maximum padding, replication, uneven-shard,
  subgroup, dtype, and co-location constraints.
- Express components that must remain together, such as projections sharing a
  fused kernel or recurrent state.
- Version and hash the description with the model configuration and checkpoint
  manifest.
- Model adapters may provide custom feasibility predicates, but those predicates
  must be deterministic, side-effect free, and unit tested.

### R3. Candidate strategies

The planner must represent at least:

1. **Exact sharding**: equal partitions when an axis divides the group size.
2. **Replication**: duplicate small tensors or heads across rank groups.
3. **Padded sharding**: add masked capacity to meet alignment or divisibility.
4. **Uneven sharding**: ranks own different logical counts with explicit sizes.
5. **Subgroup sharding**: a component uses a factor of the global world and is
   replicated across groups.
6. **Pipeline placement**: layers are assigned to stages, optionally with
   component-local TP or EP inside each stage.
7. **Host offload**: planned weight or cache placement with transfer and pinned
   memory budgets.
8. **Spare capacity**: deliberately leave devices outside the instance when all
   inclusive plans are invalid or slower.

All padding must be semantically masked. Dummy experts or heads must never be
selectable and must not affect normalization, routing probabilities, or output.

### R4. Deterministic plan generation

- Given identical inventory, model description, checkpoint manifest,
  constraints, objective, and planner version, produce the same plan.
- Separate hard feasibility constraints from soft performance scores.
- Return all rejection reasons for the highest-ranked unsuccessful candidates.
- Bound search time and candidate count. The initial implementation should use
  deterministic constrained enumeration plus beam pruning rather than require a
  general-purpose solver at runtime.
- Complete static planning in less than two seconds for the reference model on a
  normal host CPU, excluding checkpoint scanning already required by loading.
- Allow a plan-only command that does not import torch-npu or allocate a device.

### R5. Capacity model

- Account per rank for weights, scales, temporary loader tensors, KV cache,
  hybrid state, index caches, vision encoder, encoder cache, graph/workspace
  allocations, collective buffers, static offload buffers, and fragmentation.
- Distinguish persistent bytes, startup peak bytes, prefill peak bytes, and
  decode peak bytes.
- Account for padding and replication explicitly; aggregate model size divided
  by world size is never sufficient evidence.
- Model ordinary pageable RAM separately from CANN pinned-host allocations.
- Require configurable safety margins and fail before weight loading if no
  candidate satisfies them.
- After profiling, compare estimates with actual allocated and free bytes and
  record the error by component and rank.

### R6. Cost model and objectives

- Score valid plans using estimated compute balance, collective volume,
  cross-card traffic, pipeline bubbles, host transfers, memory headroom, and
  expected cache capacity.
- Prefer within-card communication where it does not create worse global
  imbalance.
- Support named objectives: `balanced`, `max_context`, `latency`, `throughput`,
  `min_host_traffic`, and `max_replicas`.
- Publish score components; do not expose only an opaque scalar.
- Permit offline benchmark profiles keyed by hardware and kernel fingerprint.
- Treat profiling data as advisory. It cannot make an invalid plan valid.

### R7. Execution-plan schema

The serialized plan must include:

- schema and planner versions;
- hardware, model, checkpoint, and software fingerprints;
- global TP, PP, EP, DCP, and replica groups where applicable;
- component-local rank groups and logical-to-physical mappings;
- exact shard sizes, padding, masks, and replicated ownership;
- pipeline layer ranges and state-transfer boundaries;
- collective type and process group for each cross-rank edge;
- weight, state, cache, workspace, and host-memory budgets per rank;
- fallback plans in priority order;
- objective, scores, assumptions, warnings, and rejected candidates.

The engine must validate the complete schema before allocating weights.

### R8. Weight loading and checkpoint mapping

- Stream directly from checkpoint tensors to each plan-defined destination.
- Do not materialize the full expert bank or full model on every worker.
- Support unequal expert counts without assuming identical local tensor shapes.
- For padded experts, initialize dummy storage deterministically and mask router
  logits before top-k selection.
- Validate that every checkpoint tensor is loaded exactly once logically, except
  explicitly replicated tensors.
- Produce per-rank byte counts and reject imbalance beyond the plan's predicted
  tolerance.

### R9. Collectives and process groups

- Create process groups from the plan instead of assuming one global TP group is
  correct for every component.
- Support equal-shape and variable-size collectives where the backend provides
  them; otherwise insert explicit pad/unpad operations.
- Keep collectives in a deterministic order across ranks to prevent deadlocks.
- Record within-card and cross-card byte volume separately.
- Validate unsupported HCCL operations during planning rather than failing in
  the first inference request.

### R10. KV cache and hybrid state

- Plan full-attention KV ownership separately from query-head ownership.
- Express replicated KV, sequence-sharded KV, and DCP ownership explicitly.
- Plan GDN/recurrent state by request and layer; state must survive chunked
  prefill, decode, preemption, and slot remapping.
- Include QSA index cache and compressed history in capacity calculations.
- Preserve block-size and per-layer layout constraints of the Ascend backend.
- Do not all-gather complete long-context caches when selected-row exchange is
  sufficient.

### R11. Pipeline-parallel fallback

- Support component-local TP inside PP stages, beginning with TP2 x PP3 for six
  devices.
- Partition model layers through the standard vLLM PP interfaces rather than
  advertising `SupportsPP` while constructing all layers on every stage.
- Define embedding, LM-head, vision, PLE, MTP, hybrid-state, and intermediate
  tensor ownership.
- Integrate with dynamic chunked pipeline parallel only after correctness under
  fixed chunking is established.
- Include pipeline transfer buffers and bubbles in the cost model.

### R12. Multimodal placement

- Keep vision enabled in the six-device acceptance path.
- Permit vision replication, TP sharding, or dedicated-subgroup execution.
- Include worst-case configured image feature size in profiling and planning.
- Allow operators to cap image resolution or encoder-cache budget explicitly;
  never silently reduce it to make a plan fit.
- Text-only fallback is diagnostic, not sufficient for multimodal sign-off.

### R13. User controls

- Preserve existing explicit parallel flags as authoritative manual mode.
- Add an `auto` topology mode and a plan-only mode through reviewed public
  configuration, not scattered environment-variable reads.
- Allow constraints such as included/excluded devices, minimum free memory,
  permitted padding, maximum host offload, required context, and required
  multimodal capability.
- Allow exporting a plan and requiring an exact saved plan at startup.
- A generated plan must be printed before expensive checkpoint loading.

### R14. Safety and fallback

- Fall back only to a previously validated, lower-ranked plan.
- Never retry a different collective topology after workers have partially
  initialized HCCL without a clean executor restart.
- Keep TP4 as the initial known-good fallback for the Qwen reference model.
- If six-device placement is infeasible, report why and whether the two spare
  devices can host another replica or service.
- Do not enable swap-backed offload by default.

### R15. Observability

- Write a durable planner artifact before loading weights.
- Record selected and rejected layouts, memory estimates, score components,
  process groups, padding, replication, and expected communication volume.
- During startup and requests, record measured per-rank memory, collective bytes,
  host-transfer bytes, load balance, expert utilization, and stage latency.
- Identify the first fatal rank and component in plan-aware error messages.
- Add no device `tensor.item()` synchronization in hot paths for reporting.

## 8. Planning algorithm

### 8.1 Phase A: normalize inputs

1. Load an offline or live hardware inventory.
2. Load model topology and checkpoint manifests.
3. Apply operator constraints and explicit user requirements.
4. Form candidate device groups using card and NUMA locality.

### 8.2 Phase B: enumerate skeletons

Enumerate bounded factorizations of the available ranks, including:

- global TP/PP/EP/DCP combinations;
- component-local subgroup sizes that divide the world;
- full-world uneven or padded candidates;
- fewer-device fallbacks and replica candidates.

Reject skeletons that violate hard layer, state, kernel, or collective
constraints before detailed memory estimation.

### 8.3 Phase C: place components

For each skeleton, choose legal strategies per component. Placement order should
start with the largest and most constrained components: routed experts, dense
weights, vision, KV cache, recurrent state, then small replicated tensors.

The initial search uses a deterministic beam. Each partial plan retains a vector
of per-rank bytes and communication edges. Dominated partial plans are removed
when another plan uses no more memory or communication on every rank.

### 8.4 Phase D: validate and score

1. Validate tensor shapes, ownership, process groups, and collective support.
2. Evaluate startup, persistent, prefill, and decode memory limits.
3. Calculate objective-specific score components.
4. Retain the selected plan and ordered fallbacks.
5. Serialize the plan before model allocation.

### 8.5 Optional calibration

Calibration may benchmark representative collectives and kernels using small,
bounded allocations. It must be explicitly enabled, cached by fingerprint, and
safe to skip. Calibration refines costs but never bypasses feasibility rules.

## 9. Six-device Qwen candidate plans

### Candidate A: component-local TP6

- Full attention: exact TP6 query sharding.
- KV heads: two logical owner groups, each replicated three ways.
- Routed experts: evaluate uneven `86/86/85/85/85/85` first; padded 516-expert
  layout is the equal-shape fallback.
- GDN value heads: exact TP6.
- GDN key heads: compare padded 18-head sharding against TP2 subgroups replicated
  three ways.
- QSA indexer: replicated or subgroup placement based on memory and collective
  cost.
- Vision: retain TP4-like sharding if supported, otherwise plan a legal subgroup
  with explicit output distribution.

This candidate maximizes resident capacity but requires component-local process
groups and new uneven/padded loader and kernel paths.

### Candidate B: TP2 x PP3

- Two-way tensor/expert sharding inside each of three pipeline stages.
- Assign 16 decoder layers per stage initially, then rebalance from measurements.
- All primary head and expert counts divide by TP2.
- Requires real Qwen pipeline partitioning, intermediate transfer, and hybrid
  state ownership.

This candidate is simpler arithmetically but may increase pipeline latency and
bubbles, especially at batch size one.

### Candidate C: TP4 plus two spare devices

- Preserve the known TP4 instance.
- Offer the remaining devices to another compatible service or an explicitly
  supported auxiliary stage.
- Do not claim additional context or speed for the primary instance.

This is the safe fallback, not the desired six-device end state.

## 10. Acceptance criteria

### 10.1 Planner correctness

- Offline fixtures cover 1, 2, 4, 6, and 8 logical devices across multiple card
  groupings.
- The planner reproduces the existing TP4 Qwen plan without changing logits.
- A six-device fixture produces Candidate A or B with complete shard sizes,
  masks, process groups, and per-rank memory budgets.
- Invalid candidates expose stable, actionable rejection reasons.
- Saved-plan replay is byte-for-byte deterministic for equivalent inputs.

### 10.2 Six-device functional gate

- Load all real W8A8 weights with vision enabled across six 310P devices.
- Return HTTP 200 and non-empty output for one deterministic text request.
- Return HTTP 200 and non-empty output for one deterministic text-plus-image
  request.
- Match TP4 reference output within agreed numerical tolerances at short context.
- Validate chunked prefill, decode, request cleanup, and a second request.
- No missing/duplicate weights, deadlocks, dummy-expert selection, or cross-rank
  state aliasing.

### 10.3 Capacity gates

1. Establish a short-context real-weight baseline.
2. Pass 60K context with vision enabled.
3. Pass the configured 162,688-token target with measured workspace margin.
4. Evaluate native 262,144 context separately; do not infer it from startup.

Each gate requires startup, a real request, per-rank peak memory, and planner
estimate error. A plan passes only when every rank retains the configured safety
margin.

### 10.4 Performance gates

- Report load time, time to first token, inter-token latency, throughput, NPU
  utilization, HCCL bytes, cross-card bytes, host-transfer bytes, and per-stage
  idle time.
- Compare TP6 candidates against TP4 with equivalent context and output settings.
- A production default must improve the selected objective without regressing
  correctness or exceeding memory margins.
- Performance thresholds will be set after the first calibrated prototype;
  device-count scaling alone is not an acceptance criterion.

## 11. Test strategy

### 11.1 CPU unit tests

- Hardware-inventory normalization and fingerprints.
- Divisibility, padding, replication, subgroup, and uneven-shard constraints.
- Expert distributions such as `512 -> 6` and head distributions such as
  `16 -> 6`.
- Memory accounting for persistent and peak phases.
- Candidate dominance, deterministic ordering, rejection diagnostics, and plan
  serialization.
- Weight-name-to-owner mapping with no missing or duplicate logical tensors.
- Masking of padded experts and heads.

### 11.2 NPU unit/integration tests

- Process-group construction and collective ordering.
- Equal, padded, and uneven tensor exchanges.
- Expert routing parity and global top-k semantics.
- GDN/QSA state parity across chunk boundaries.
- Failure injection for one rank, insufficient memory, unsupported collective,
  and stale plan fingerprints.

### 11.3 System and nightly tests

- TP4 regression on the existing two-card host.
- Six-device Qwen text and vision smoke tests.
- 60K and 162,688-token capacity tests.
- Multi-request lifecycle and cleanup.
- Long-running memory-fragmentation and load-balance tests.
- Planner estimate-versus-measurement reports retained as artifacts.

## 12. Rollout plan

### Phase 0: contracts and offline planner

- Hardware inventory schema.
- Model topology description and plan schema.
- Offline deterministic candidate enumeration and memory accounting.
- TP4 plan reproduction and six-device fixtures.

### Phase 1: equal and replicated component groups

- Component-local process groups.
- Exact sharding and replication.
- Saved-plan validation and startup observability.

### Phase 2: padded and uneven MoE

- 512-to-6 expert ownership, loading, routing, and collectives.
- Padded expert masking and uneven-exchange alternatives.
- Numerical and performance comparison before choosing the default.

### Phase 3: GDN and QSA TP6

- GDN key-head padding or TP2 subgroup replication.
- KV-head and indexer ownership.
- Hybrid state and cache planning.

### Phase 4: TP2 x PP3 fallback

- Real Qwen pipeline layer partitioning.
- Embedding, vision, PLE, state, and LM-head ownership.
- Fixed chunking correctness, followed by dynamic chunk integration.

### Phase 5: automatic selection

- Calibrated cost model.
- User-facing `auto`, plan-only, export, and replay workflows.
- Broaden model coverage only after TP4 and six-device Qwen gates remain stable.

## 13. Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Uneven HCCL exchanges perform poorly or are unsupported | TP6 stalls or deadlocks | validate at planning time; retain padded equal-shape and PP fallbacks |
| Padding changes routing semantics | silent accuracy loss | mask before normalization/top-k; parity tests and dummy-selection counters |
| Component-local groups create collective-order bugs | multi-rank hang | deterministic group graph, startup validation, fault-injection tests |
| Static estimates miss vision or workspace peaks | startup/request OOM | phase-specific budgets, measured feedback, mandatory safety margin |
| TP2 x PP3 duplicates large tensors | expected memory gain disappears | account replication before loading; reject dominated plans |
| Planner becomes model-name conditional code | poor maintainability | declarative contracts and generic strategy interfaces |
| Calibration delays startup | poor usability | offline cache, opt-in calibration, bounded samples |
| Pageable/pinned host memory is conflated | host allocation failure | separate budgets and measured CANN pinned-pool limits |
| Six ranks increase cross-card traffic | slower than TP4 | topology-aware scoring and explicit fallback |

## 14. Open questions

1. Does the 310P HCCL stack provide performant variable-size all-to-all for
   uneven expert ownership, or should padded 516 experts be the first path?
2. Is padding GDN key heads from 16 to 18 cheaper than TP2 subgroup replication
   once recurrent state and reductions are included?
3. Can the vision encoder use a different rank group without upstream vLLM
   changes to multimodal embedding distribution?
4. Which topology fields belong upstream in vLLM versus the Ascend plugin?
5. Should plan selection occur entirely before workers start, or should workers
   contribute hardware-local facts before finalization?
6. What minimum free-memory margin is safe for eager and graph modes on 310P?
7. Which plan components can share process groups without constraining future
   scheduling?
8. Should the first production objective favor full 162K context or maximum
   short-context throughput?

## 15. Deliverables

- Versioned hardware-inventory, model-topology, and execution-plan schemas.
- Offline planner CLI and machine-readable fixtures.
- Planner library with deterministic candidate generation and cost breakdown.
- Generic exact, replicated, padded, uneven, subgroup, pipeline, and offload
  placement interfaces.
- Qwen six-device adapter declarations and runtime support.
- Unit, integration, E2E, long-context, vision, and performance tests.
- Operator documentation for automatic planning, overrides, export, replay,
  diagnostics, and fallback.
- A decision record selecting the first production six-device Qwen topology from
  measured Candidate A versus Candidate B evidence.
