---
name: define-the-problem
description: pin the operation contract, shapes (compile-time vs runtime), layouts, dtypes/numerics/tolerances, and boundary conditions before choosing how the kernel runs.
source: CK DSL Optimization Runbook (ROCm/rocm-libraries@1efeadd, projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
---

## 1. Define The Problem Exactly

### 1.1 Operation Contract

Decide what the kernel computes before deciding how it should run.

- Name the operation precisely. Each kernel family (GEMM, implicit-GEMM
  conv, direct grouped conv, paged attention, the small-op family such
  as reduce / LayerNorm / RMSNorm / elementwise / transpose, and the
  FMHA variants: varlen, head grouping, paged prefill, split-KV decode,
  fp8 forward, backward, append-KV) has its own parameter set. The
  family name plus its parameters is the contract.
- State whether the operation is forward, backward-data, backward-
  weight, inference-only, training, deterministic, or approximate.
  Backward FMHA is its own kernel; everything else here is forward.
- State whether the op is standalone or part of a graph/fusion
  boundary. Fusion works on a graph of ops with a fused epilogue and
  a pattern table for what can fuse.
- State whether atomics are allowed (fp32 global atomic add, emitted
  by split-K GEMM and Stream-K).
- State all side effects: workspace (and its size), scratch buffers,
  stream usage, global memory writes, persistent state.

### 1.2 Shapes

- Record dimensions with names, not values: use explicit field names
  (conv: `N`, `Hi`, `Wi`, `C`, `K`, `R`, `S`, strides, pads,
  dilations; attention: `total_q`, `num_seqs`, `num_query_heads`,
  `num_kv_heads`, `head_size`, `block_size`, `max_seqlen_q`,
  `max_seqlen_k`, …).
- State which dimensions are compile-time constants (kernel config)
  and which are runtime (kernel arguments).
- Identify pathological boundary shapes: `1, 2, 3, 15, 16, 17, 31, 32,
  33`, powers of two, non-multiples of tile sizes. Add them to the
  parity harness.

### 1.3 Layouts

- Make layouts explicit in the kernel config: GEMM `"RCR"` / `"CRR"`
  / etc.; conv NHWC by default; attention (block, slot) paging for KV.
- For non-trivial mappings (paged KV, im2col), describe them as a
  coordinate-transform DAG with operators `pass_through`, `pad`,
  `pad_dynamic`, `embed`, `unmerge`, `merge`, `indirect`.
- State alignment and contiguity at the tensor descriptor.

### 1.4 Dtypes And Numerics

- Record input dtype, weight dtype, accumulator dtype, output dtype,
  scale dtype, bias dtype, index dtype. Keep these
  fields explicit in the kernel config; e.g. activations `bf16` with
  KV storage `fp8e4m3` for an FP8 KV cache path.
- Typical dtype catalog: `I1`, `I8`, `I16`, `I32`, `I64`, `BF16`,
  `F16`, `F32`, `FP8E4M3`, `BF8E5M2`.
- Quantization helpers: max-abs scale, scalar quantize / dequantize,
  and the quant dtype set `i8`, `fp8e4m3`, `bf8e5m2`.
- MX block-scale GEMM needs its own spec for the scale layout.
- Always set a tolerance policy. A parity harness should encode per-op tolerances; the values we
  currently believe in are
  (elementwise linear ops bit-exact, silu/gelu `<= 2e-4`, layer/rms
  norm `<= 5e-3`, reduce `<= 1.5e-3`,
  gemm `<= 7e-2`).
- For fp16 inputs with fp32 accumulation over O(100) terms, expect
  errors near the fp16 ULP floor for correct kernels. Errors two
  orders of magnitude higher almost always indicate structural bugs
  (wrong lane mapping, missing acc reset, stale LDS, etc.). See the empirical case studies for specific bug signatures.

### 1.5 Boundary Conditions

- Padding: a pad op, or a dynamic pad that returns `(offset, valid)`
  per coordinate.
- Buffer descriptors with `flags=0x00027000` (= 159744) return zero
  for out-of-bounds lanes — the canonical tail-safe primitive. It is
  built from a base pointer and a byte count (`num_bytes`).
- Causal masks, sliding window, and softcap are attention-side helpers
  (mask functions plus a log2-domain softcap).
- Split-K / persistent / stream-K decisions belong in the Stream-K
  GEMM kernel plus its persistent / Stream-K scheduling helpers.
- Page-table indirection for paged attention: an `indirect` transform
  followed by `unmerge`.

ck_dsl specifics: [primitive-map](../../dsl/ck-dsl/primitive-map.md#define-the-problem).

---
Next: [establish-baselines](establish-baselines.md). Lane-mapping K-pack tolerance bug: [work-decomposition](../levers/work-decomposition.md). The family and its parameters are the contract — see the variant lists in [algorithmic-mapping](../levers/algorithmic-mapping.md).
