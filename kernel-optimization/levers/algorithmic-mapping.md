---
name: algorithmic-mapping
description: pick the kernel before the knobs. GEMM / conv / attention / reduction / transpose / gather family instance tables, plus the implicit-GEMM VGPR-tax explainer.
source: CK DSL Optimization Runbook (ROCm/rocm-libraries@1efeadd, projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
---

## 4. Algorithmic Mapping Choices

> **Hardware Resource Consumption Warning**: Different algorithmic
> approaches for the same operation can have dramatically different
> hardware resource requirements (VGPRs, SGPRs, LDS) even when they
> achieve similar compute throughput. Always profile resource use
> alongside performance — a per-variant occupancy check is the
> answer.

### 4.1 GEMM Family

Variants to choose from: plain GEMM, batched GEMM, strided batched,
grouped GEMM, persistent GEMM, Stream-K, split-K (atomic add of fp32
partials), preshuffled GEMM, block-scaled GEMM, sparse GEMM (not yet
available), quantized GEMM (block-scaled / MX), fused-epilogue GEMM,
a simple MFMA reference GEMM, flat-MM for small decode shapes, and
batched N-D contraction.

Consider:

- Does the shape fill MFMA tile dimensions? Verify the
  shape against the available MFMA atoms.
- Is M / N / K too small? Use a flat-MM or simple MFMA kernel instead of
  the universal GEMM for tiny shapes.
- Is grouped scheduling better than padding to a common shape?
  A grouped kernel handles this.
- Is split-K useful, or only adding atomic overhead?
- Is Stream-K needed for load balancing?
- Is weight preshuffle worth the preprocessing cost?
- Is data reused enough to justify LDS?
- Is A or B naturally contiguous? (drives the row/column layout choice, e.g. RCR)

### 4.2 Convolution

Variants: direct convolution, implicit GEMM (including auto-unrolled),
im2col + GEMM (materializes the im2col operand), Winograd and FFT (not
yet available), and depthwise / small-channel direct conv (4-channel
grouped).

Consider:

- If `K` or `C * R * S` is tiny, implicit GEMM may be structurally
  weak.
- Direct conv can preserve spatial/channel reuse.
- Small-channel conv may need custom MFMA mapping: `4x4x4`,
  `16x16x16`, `16x16x32` (all three exist as f16 MFMA atoms).
- Use circular row accumulators for `R` rows.
  A streaming-row, 3-accumulator circular pipeline is the reference
  pattern.
- Stream input rows so one input row contributes to multiple output
  rows.
- Keep filter weights in registers when small.
- Stage weights in LDS only if cooperative load + register pressure
  justifies it.
- Prefer direct stores unless a proven LDS epilogue improves
  coalescing enough to pay for barriers (see Section 9.3 + Empirical
  Case Study 1).

#### 4.2a The Implicit GEMM VGPR Tax

Implicit GEMM convolution pays an unavoidable "VGPR tax" compared to
pure GEMM due to im2col coordinate computation overhead. Understanding
this helps set realistic performance expectations.

VGPR Breakdown Comparison (3×3 conv, 64×128 tile):

| Component | Pure GEMM | Implicit GEMM Conv | Δ |
|---|---|---|---|
| MFMA Accumulators | 32 | 32 | 0 |
| Address Computation | 5-8 | 15-20 | +10-12 |
| LDS Offsets | 5-8 | 8-12 | +3-4 |
| Temporary Vectors | 12 | 12 | 0 |
| Loop Indices | 3-5 | 5-8 | +2-3 |
| **TOTAL** | **57-65** | **72-84** | **+15-19** |

Why the extra VGPRs?

- Computing spatial coordinates (n, ho, wo) from a linearized output
  index.
- Computing input coordinates (hi, wi) from (ho, wo, y, x) with
  stride/dilation.
- Bounds checking for padding regions.
- Filter position tracking (y, x within R×S window).
- Group offset computation for grouped convolutions.

Practical consequences:

- Implicit GEMM convolution typically achieves **60-65 % of theoretical
  peak**.
- Pure GEMM typically achieves **75-85 % of theoretical peak**.
- The 15-20 VGPR overhead reduces occupancy: 72 VGPRs → 7 waves/CU vs
  60 VGPRs → 8 waves/CU.
- This is a **fundamental algorithmic constraint**, not a tuning
  failure.

**⚠ WARNING**: Do NOT chase VGPR reduction below the im2col floor
(~60 VGPRs for 3×3 conv with typical tile sizes). The coordinate
arithmetic is mathematically required. Attempting to reduce VGPRs
further will:

- Require smaller tile sizes (reducing reuse and performance).
- Force spilling to memory (catastrophic for performance).
- Not achieve GEMM-level efficiency regardless of optimization effort.

When to accept the VGPR tax:

- Your conv kernel is within 5 % of other mature frameworks.
- You are at 60-65 % of theoretical peak for compute-bound shapes.
- Further VGPR optimization attempts have failed or regressed.

When to consider alternatives:

- Very small spatial dimensions: direct conv may have lower overhead
  (use a direct-conv variant).
- Large filter sizes (5×5, 7×7): Winograd or FFT may amortize transform
  cost.
- Extreme channel counts with small spatial: specialized layouts.

Invest optimization time in memory access patterns, LDS swizzling, and
tile sizes rather than trying to eliminate the fundamental VGPR
overhead of im2col addressing. Use a per-variant occupancy check to confirm the
VGPR floor before deciding.

### 4.3 Attention

Variants: unified prefill + decode (scalar oracle), 2D MFMA tiled
prefill + decode, split-KV decode segment + reduce, FMHA forward (MFMA,
prefill), FMHA varlen, FMHA head grouping (GQA / MQA), FMHA paged
prefill, FMHA split-KV decode, FMHA fp8 forward, FMHA backward, FMHA
append-KV (DMA), Sage attention (per-block scaled), and block-sparse
attention (Jenga / VSA).

Consider:

- Q / K / V head dimension alignment to MFMA: pick a valid 16x16x16 atom
  for the dtype.
- Numerically stable max/sum accumulation: an online-softmax state plus
  warp xor-reductions for max and sum.
- Page-table overhead — choose between the 2D and the 3D (split-KV)
  tiled kernel. A config heuristic can decide automatically; override
  warp count and tile size for sweeps.
- Coalescing across heads / tokens / pages.
- Shared memory capacity for Q / K / V tiles.
- Register pressure from accumulators and softmax state. The 2D path
  carries Q in LDS once and async DMAs K/V (see Section 6.3).
- Store path for output and log-sum-exp.

### 4.4 Reductions And Normalization

Variants: row reduce (sum/max/min/mean/prod), LayerNorm forward,
RMSNorm forward, fused add + RMSNorm + row-dynamic quant, smoothquant
(incl. MoE), top-k softmax, and pooling.

Consider:

- Reduction axis length.
- Number of rows.
- Whether one block per row is enough.
- Whether multiple CTAs per row require atomics or a second pass.
- Vector width (sweep it as a knob).
- Numerically stable accumulation (an LDS block reduction should
  support sum / max / min / prod).
- Fusing scale / bias / activation through a fused epilogue or
  directly in the kernel.

### 4.5 Transpose, Permute, Copy

Variants: 2D transpose (LDS-staged), batched 2D transpose, in-register
sub-tile transpose, N-D permutation, and im2col copy.

- Use vectorized global loads/stores.
- Use LDS tile for non-coalesced stores.
- Avoid bank conflicts with padding or swizzle (Section 6.4a).
- Use rectangular tiles tuned to the aspect ratio.
- Avoid over-general rank logic in hot loops; a one-thread-per-element N-D permute with rank-N
  index decompose is for clarity, not for peak throughput.

### 4.6 Gather, Scatter, Paged, Sparse

- Index load overhead may dominate.
- Coalesce metadata loads.
- Cache page tables.
- Batch by locality.
- Avoid divergent branches.
- Use buffer descriptors with proper bounds (buffer resource).
- Precompute or compact index maps.
- Separate dense fast path from sparse fallback.
- For MoE, use a dedicated pipeline: token sorting, fused MoE GEMM,
  and an end-to-end fused MoE path.

## Lessons from worked examples

- **Split-K does not help when the large operand has no cross-CTA reuse.** Splitting K spreads the same bytes over more CTAs and adds a reduce, but each CTA still streams unique weight bytes. Total traffic is unchanged (or grows if the small operand is replicated) while the kernel is already at the bandwidth ceiling. Use split-K to fix under-filled grids, not to raise bandwidth.
- **Fuse to remove HBM round-trips, not launches.** Eager launch and graph replay differ by only a few µs on large-enough kernels, so fewer launches buys little. Fusing neighbouring ops (gate/up/activation/down/reduce, or a GEMM with its consumer) pays because intermediates stay in registers instead of crossing HBM.

ck_dsl specifics: [primitive-map](../../dsl/ck-dsl/primitive-map.md#algorithmic-mapping).

---
Next: [work-decomposition](work-decomposition.md). Per-family variant levers: [knob-catalog-and-sweep](../../dsl/ck-dsl/knob-catalog-and-sweep.md). Op checklists: [op-specific-checklists](../autotuning/op-specific-checklists.md).

