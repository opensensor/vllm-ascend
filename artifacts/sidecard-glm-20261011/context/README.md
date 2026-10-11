# Context capacity experiments

The retired `context-v4` profile used four Ascend ranks, an 8 GiB cache budget
per rank, one active sequence, compact KDA state and disabled prefix caching.
The automatic planner advertised **709,120 tokens**. The checkpoint's
architectural maximum is 1,048,576 positions. CUDA held expert weights;
attention and state remained on both Ascend cards.

[context-validation.json](context-validation.json) records completed chat
generation with a 32,000-token output budget, HTTP admission of a
709,119-token prompt plus one output token, and rejection of an over-limit
output budget. The large prompt was disconnected immediately after headers.
Full-capacity prefill, sustained generation and long-context quality were not
validated. [service-process.json](service-process.json) is a historical receipt.

The earlier 9 GiB profile advertised 735,360 tokens but failed its first
generation in native attention workspace allocation. Two failed model workers
retained their allocations after SIGTERM. Their saved command/cwd identities
were verified before forcefully retiring only those owned workers;
[orphan-cleanup.json](orphan-cleanup.json) records the result. No device reset
was used. The 8 GiB profile completed actual generation afterward.

[context-indexer-million.log](context-indexer-million.log) records isolated
indexer checks with 128 input rows and 262,144 pools, approximately one million
positions. This is operator evidence, not whole-model long-context validation.

The 128-token prefill budget was operationally inadequate: the user's
25,456-token prompt logged approximately 25 seconds per chunk, about five
tokens/s. Context capacity and prefill throughput are separate requirements.
The subsequent performance profile restores prefix caching, increases the
prefill budget and replaces scalar tensor serialization and CUDA projections.
