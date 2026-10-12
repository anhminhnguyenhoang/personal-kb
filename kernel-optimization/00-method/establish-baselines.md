---
name: establish-baselines
description: layered correctness gates (cheapest-first), vendor/library perf baselines, benchmark hygiene (HIP events, median+spread, bimodal/cold-cache caveats), and metadata to record. Never report speed without correctness.
source: CK DSL Optimization Runbook (ROCm/rocm-libraries@1efeadd, projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
---

## 2. Establish Baselines

### 2.1 Correctness Baselines

Use layered correctness gates, in order from cheapest to most
expensive:

1. Static tests (IR construction, transform DAG, helpers, instance
   smoke). 286 tests, ~1.7 s.
2. A docs / symbol check: import every symbol, exercise every IR
   builder method, lower every spec to LLVM/HIP/CK Tile, build the
   code object (HSACO), launch small kernels (50 checks, ~2 s).
3. Per-kernel verification from a manifest: load the HSACO, build
   inputs, run the kernel, compare against the in-process NumPy/torch
   reference.
4. Small ops vs a torch reference (20 cases, deterministic with
   seed=0).
5. Extended-kernel parity: FMHA / Sparse / Sage / MoE / Block-scale /
   MX correctness.
6. Attention parity (Triton + reference vs the DSL paths).
7. An audit of production LLVM lowering vs HIP-debug lowering across
   every shipped spec.

References do not have to be torch. The conv and GEMM bake-offs use
NumPy fp32 accumulation. Attention has a deliberate per-shape
reference paged-attention function.

### 2.2 Performance Baselines

Use the best available vendor / library baseline:

- Conv / GEMM bake-offs: CK Tile C++ lowered from the same kernel
  definition. Currently only universal GEMM and implicit-GEMM conv
  lower cleanly to CK Tile C++; the rest raise `NotImplementedError`.
- Triton: AITER ships the production `unified_attention` kernel that
  vLLM and AITER use. Path comes in via `AITER_PATH` env var.
- AITER FA / FA2 for the FMHA shapes.
- Torch eager.
- A naive scalar warp-level FMHA body is the **correctness oracle**
  for the FMHA family. Several FMHA variants still ship with the
  warp-scalar body (paged prefill, split-KV decode, backward, varlen
  on bf16 pre-F1, fp8 on the QK/PV paths) — these are intentionally
  NOT perf paths. The MFMA-tiled body replaces the scalar one in the
  production path.

Hard rule: **do not** report speed without correctness. The top-level constraint applies verbatim here.

### 2.3 Benchmark Hygiene

- Use HIP events with warmup, a per-launch synchronize, and an
  iteration loop.
- For Triton interop, prefer a direct `torch.cuda.Event` window: a
  per-iteration `synchronize()` perturbs Triton's autotuner.
- Run at least 3-5 invocations and report median + spread (defined as
  `(max - min) / median * 100 %`). Let one benchmark-summary helper
  own this.
- Some kernels are bimodal across runs even with identical code: a
  fresh process may hit one tier, steady-state runs settle to another.
  Always run at least 3-5 invocations and record both median and
  spread.
- Cold-cache vs warm-cache effects can produce single-run drops of
  more than 2× the steady-state throughput on the first run of a
  fresh process. Discard the first run unless you are explicitly
  measuring cold start.
- For multi-kernel pipelines, load the HSACO once (one launcher per
  module, or a multi-stage launcher). This amortizes the HIP module
  load.
- For graph replay timing, elide per-call sync inside the event-timed
  loop (a no-fence timing mode).
- Pin GPU clocks if you suspect throttling:
  `sudo rocm-smi --setperflevel high && sudo rocm-smi --setsclk 7`.

### 2.4 Metadata To Record

A manifest writer records most of this automatically:

- GPU model and architecture (default ISA target is `amdgcn-amd-amdhsa
  --gfx950`).
- ROCm/CUDA version, driver/runtime version.
- Library commit SHA (set via `manifest.notes`).
- Kernel source commit SHA.
- Compile options (default `-O3`).
- Target architecture flag.
- Build type and `NDEBUG`.
- Optimization flags.
- Environment variables.
- Binary code object metadata (resources from a static HSACO analysis).
- A per-stage timings record: IR build, LLVM lowering, bitcode compile,
  relocation, executable, total.

## Lessons from worked examples

- **Ask the reference what it actually runs.** Profile the library call on your exact shape and read the real kernel name (macro tile, atom, direct-to-LDS, prefetch depth). A config inferred from a similar shape or a tuning file can be the wrong family. The real one may be a tile your framework already expresses with deeper K.

ck_dsl specifics: [primitive-map](../../dsl/ck-dsl/primitive-map.md#establish-baselines).

---
Next: [bottleneck-classification](bottleneck-classification.md). Exact commands: [reproducible-commands](../../dsl/ck-dsl/reproducible-commands.md). Hygiene knobs: [knob-catalog-and-sweep](../../dsl/ck-dsl/knob-catalog-and-sweep.md).

