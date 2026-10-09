# T8 composition review follow-up

Final read-only review of candidate, builder and resident admission/controller
identified two gaps that the coordinator corrected before source freeze:

- Unsupported FP32 input or BF16 parameter dtype now falls back before candidate
  callbacks, preserving the FP16 packing/activation/projection contract.
- GDN, PLE and both prefix hooks now share the entire candidate owner's failure
  guard. Any layer submission error poisons the composition and prevents later
  candidate execution or baseline fallback, as an MoE submission error already did.

Added six owned CPU regression cases for these corrections. Final composition
validation passed **39 tests**, with 20 existing environment warnings, in
10.89 seconds. Scoped Ruff and formatting passed. The earlier 33-test report and
receipts remain historical; `tests-review-final.log` is the final test run.

The source-byte loader check now precedes Torch-NPU/library/kernel loading, and
actual bridge/binary inventories match the builder schema. Read-only inspection
found no additional undefined loader fields or unsafe failure fallback. Controller
unknown collective completion holds the transaction, and T6 scratch tensor slots
remain per invocation. No production files were edited by this test worker.

Byte matching alone cannot authenticate already imported code objects or prove
actual worker Torch/CANN ABI, runtime import roots, asynchronous collective error
completion, numerical quality or thermal behavior. Those remain explicit T9
admission/setup gates. No NPU/device/server/network operations occurred.
