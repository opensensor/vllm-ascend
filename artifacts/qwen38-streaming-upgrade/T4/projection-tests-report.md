# T4 projection: independent CPU validation

The actual `native_streaming.cpp` entry points, `Projection` implementation and
`OperandProducer` run against a synchronous host fixture. Independent NumPy
integer products and ordered FP32 correction match the resulting FP16 outputs
exactly. These checks establish host arithmetic and addressing consistency;
device correctness, native conversion rounding, asynchronous ordering, actual
overlap, quality, images, serving performance and thermal behavior remain pending.

## Coverage

- Full projection covers K128, K640 and K2560, rows 1/15/16/17/33, two random
  seeds, multiple experts, empty groups and tile tails. Production gate/up
  N1280/K2560 and down N2560/K640 are included. An additional E128 case has
  mostly empty experts.
- All eight block assignments run sequentially on the host. INT4 nibble decoding
  reads the actual L0A and L0B layouts and writes paired `[N/16,32,16]` products.
  Vector repeat masks, 32-byte block strides, metadata layouts, group correction,
  FP32 additions, FP16 stores and row strides execute through the actual body.
- Down column windows of 8/8/4 tiles concatenate to the complete N2560 reference.
  Gate/up windows of 8/2 tiles match N1280/K2560. Interior offsets and single
  last tiles exercise the full bank stride with a compact output stride.
- Inputs remain unchanged, output canaries survive, and local allocation peaks
  match the T2 contract. The same output buffer is reused through local-prefix
  lengths 17 → 1 → 0 → 16 → 17; peer tails are explicitly zero each time.
- Actual native entry guards reject invalid dimensions, groups and column
  windows. Actual Python resource guards reject unsupported shapes/dtypes,
  CPU inputs, invalid column bounds and a missing column resource.
  Launch-spy tests bypass only the one-NPU gate to inspect both configuration
  signatures without loading a device library or launching a device kernel.

## Fixture and limitations

The host build uses `g++ -std=c++17 -O2 -shared -fPIC -ffp-contract=off` and loads
only that CPU library through ctypes. The fixture implements integer MMAD,
packed LoadData, matrix readback and vector primitives independently. TBuf
allocations and tensor accesses are bounds checked; directional event IDs must
remain in the installed eight-ID limit, signal/wait pairs must match, and final
drain must leave no outstanding signals.

Everything executes synchronously. Event checks therefore validate paired API
use, not device barrier legality, DMA completion, overlap or task timing. The
fixture uses GCC `_Float16` conversion with the host rounding mode; real device
`CAST_NONE` still requires a separate gate. The reference deliberately executes
distinct FP32 correction operations and ascending G128 additions. It does not
claim builtin FP16 SwiGLU equivalence or full-model numerical quality.

Tests do not allocate the maximum R25600/E128/N2560 bank. They cover the production
widths and reduction groups with bounded rows and separately exercise the E128
boundary. Whole-rank memory admission remains governed by the explicit T2 budget.
No NPU runtime import, NPU open, server operation, device-library load or device
kernel launch occurred.

## Validation

- `OMP_NUM_THREADS=2 python -m pytest --noconftest -q tests/ut/qwen38_1m/test_streaming_projection.py`:
  70 passed.
- Scoped manual pre-commit checks for the Python fixture, C++ harness and stub
  header passed, including Ruff, clang-format, spelling and repository checks.
- Source hashes and the current contract identity are recorded in
  `projection-tests-validation.json`. Source changes require repeating the tests.
