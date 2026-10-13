---
name: failure-modes
description: correctness failures (lane mapping, stale LDS, REGS_PER_LANE invariants, validator-outlives-constraint, HSACO cache aliasing), performance failures (scalar stores, spills, low occupancy), and benchmark failures.
source: CK DSL Optimization Runbook (ROCm/rocm-libraries@1efeadd, projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
---

## 14. Failure Modes

### 14.1 Correctness Failures

- Wrong lane mapping.
- Wrong vector element order.
- Missing boundary zero.
- Stale LDS data.
- Missing barrier.
- Wait without barrier for cross-thread data.
- Incorrect tail store.
- Wrong group / head offset.
- Wrong stride.
- Wrong K packing.
- Accumulator slot reset too early or too late.
- Compiler flag miscompile (Section 10.3).
- DSL AST rewriter elision (Section 10.3 — only relevant for
  tracing-style DSLs; a DSL can guard it by raising on boolean
  conversion of runtime values).
- HIP-debug backend missing an op lowering (run
  the lowering-compare probe to detect).
- **`REGS_PER_LANE`-dependent invariant left over from a previous
  atom.** When switching MFMA shape (e.g. 16×16 → 32×32 quadruples
  `REGS_PER_LANE`), any code that writes only "slot 0" of a per-lane
  state tensor will silently cover 1/4 or 1/16 of the lanes. The
  sinks initialization bug in the unified-attention worked example is the canonical example —
  it manifested as an elevated `max_abs_diff` localized to specific
  query positions while the rest of the output was bit-exact.
- **Spec-validator restriction outlives the constraint it protected.**
  A "to be safe" validator block that prevents two flags from
  combining can outlive the v1 restriction it documented; if the
  restriction is no longer real, the validator is silently blocking
  the winning code path. Audit each cross-flag restriction
  periodically.
- **HSACO module-cache aliasing across specializations.** When the
  display kernel symbol is the same across two compile-time-constant
  variants, the cache can serve the wrong HSACO. Salt the kernel
  symbol with a shape hash before compile.

### 14.2 Performance Failures

- Scalar stores (the ISA-inspection probe: `buffer_store_short` >> 0).
- Scalar loads.
- Excessive barriers.
- Excessive waits.
- Register spills (the occupancy probe: `spill > 0`).
- Low occupancy (the occupancy probe: `waves_per_cu < 4`).
- Bank conflicts (the LDS-conflict analyzer).
- Bad grid scheduling.
- Tiny per-block work.
- Too much LDS.
- Over-general runtime index math.
- Uncoalesced metadata loads.
- Over-fused epilogue.

### 14.3 Benchmark Failures

- Measuring allocation.
- Measuring initialization.
- Missing synchronization.
- Not enough iterations.
- Thermal throttling (lock with `rocm-smi --setperflevel high`).
- Clock changes.
- Cache-biased results.
- Comparing different shapes / layouts.
- Comparing different precision.
- Verification included in timing.
- First-run JIT compilation included.
- The DSL launch-timing helper vs Triton autotuner interaction — use
  a HIP-event timer instead when timing
  Triton in the same window.

## Lessons from worked examples

- **Error of order 1 on a subset of scenarios means a missing term.** When only the variants with an extra score term (bias, ALiBi, mask) show errors of the same size as the values, while all others sit at dtype-ULP, an addend was never computed. Check that path before suspecting lane layout or atom bugs, whose errors are small and scattered.
- **A harness that mis-sizes the dtype reports a correct kernel as broken.** A verify path that allocates fp16 buffers for a bf16 kernel gives huge diffs on every element (a false alarm). When the diff is absurd and uniform, validate the harness against a known-good result before debugging the kernel.

- **No gain from freed VGPRs means the kernel is not occupancy-limited, and the binding resource is the one to shrink.** If cutting VGPRs adds no waves and no speed, VGPR is not the limit. LDS use is `max(staging, epilogue tile)`, so shrinking the smaller of the two changes nothing. To raise occupancy, shrink whichever resource actually sets the per-block footprint.
- **A hardcoded shape constant faults when the kernel is re-instantiated.** A constant derived from one configuration (e.g. head count 128) baked into indexing reads or writes out of bounds when the same kernel runs at a smaller value (e.g. 16 heads under another parallel layout). Derive every such constant from the instantiation's parameters and re-run accuracy after each port.
- **Atomic merges are not bitwise reproducible.** Float atomics add in arrival order, so results change run to run. For a deterministic reduction, write fp32 partials to fixed slots, sum them in a fixed order, then cast once. ([CK fmha_bwd convert_dq](https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/01_fmha))
- **fp16 atomic accumulation overflows on bf16-range data.** Packed half2 atomics saturate to ±inf when partial sums exceed fp16 range (max 65504), e.g. in a MoE top-k combine. Accumulate with f32 or bf16 atomics, or use a reduce pass. ([FlyDSL MoE 2-stage GEMM](https://github.com/ROCm/FlyDSL))

ck_dsl specifics: [primitive-map](../../dsl/ck-dsl/primitive-map.md#failure-modes).

---
Heuristics for fixing each: [decision-heuristics](decision-heuristics.md). Tolerance signatures: [define-the-problem](../method/define-the-problem.md).
