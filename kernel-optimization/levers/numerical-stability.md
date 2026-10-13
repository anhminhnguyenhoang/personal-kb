---
name: numerical-stability
description: "Accuracy invariants a fast kernel must keep: fp32 accumulation, online softmax, Welford statistics, the fp8 FNUZ vs OCP 2x trap, scale hygiene, and how to validate against an fp32 reference."
source: https://rocm.docs.amd.com/en/latest/reference/precision-support.html ; AMD CDNA3 and CDNA4 ISA reference guides ; arXiv 1805.02867 ; arXiv 2205.14135 ; arXiv 2209.05433 ; arXiv 2310.10537 ; Welford, Technometrics 4(3), 1962
---

# Numerical stability as a lever constraint

Speed must not cost accuracy. Most "free" speedups that break accuracy
come from five places. Check each one before you call a variant a win.

```text
inputs (bf16 / fp16 / fp8 / fp4)
   │  scale + clamp in fp32 ─────────────── (4) scale hygiene
   ▼
MFMA: D = A·B + C, C in fp32 ────────────── (1) keep the fp32 accumulator
   │
   ├─ softmax over a streamed axis ──────── (2) online softmax
   ├─ mean / variance over a row ────────── (3) Welford or two-pass
   ▼
cast to output dtype in the epilogue only
   │
   ▼
validate vs fp32 reference ──────────────── (5) metrics + cell sweep
```

## 1. Keep the fp32 accumulator

- Matrix instructions (`v_mfma_*`) take low-precision A and B but
  accumulate into fp32 C (in AGPRs or VGPRs). Do not down-cast C inside
  the K-loop. Cast once, in the epilogue.
- Long reductions (softmax denominator, norm sums, log-sum-exp)
  accumulate in fp32 whatever the I/O dtype. A bf16 running sum loses
  low bits on every add once the sum is large relative to each term.
- Down-casting the accumulator buys no speed on the matrix pipe. It only
  saves registers, and the accuracy cost grows with K.

## 2. Online softmax for a streamed axis

Keep a running max `m` and running sum `l` per row. For each new block
with block max `m_b`:

```text
m_new = max(m, m_b)
l     = l * exp(m - m_new) + sum(exp(x_b - m_new))
O     = O * exp(m - m_new) + exp(x_b - m_new) · V_b
```

Divide `O` by `l` once at the end. This never evaluates `exp` of a large
positive value and needs one pass. Keep `m`, `l` and `O` in fp32 and
reset them at the right lifecycle point (per query row, not per block).

## 3. Welford for mean and variance

`E[x²] − E[x]²` cancels badly when the mean is large relative to the
spread. Welford's update is stable in one pass:

```text
n += 1;  d = x - mean;  mean += d / n;  M2 += d * (x - mean)
var = M2 / n
```

Partial Welford states from different lanes or waves merge with the
parallel (Chan) formula. A two-pass mean-then-variance is also stable
when the row is already in registers or LDS.

## 4. fp8 formats and scale hygiene

| Format | Exponent bias | Max normal | Inf | NaN | Native on |
|---|---:|---:|---|---|---|
| E4M3 FNUZ | 8 | 240 | none | one code (negative zero) | CDNA3 |
| E4M3 OCP | 7 | 448 | none | yes | CDNA4 |
| E5M2 FNUZ | 16 | 57344 | none | one code (negative zero) | CDNA3 |
| E5M2 OCP | 15 | 57344 | yes | yes | CDNA4 |

- **The 2× trap.** The bias differs by one, so the same bit pattern
  decodes to half the value in FNUZ that it has in OCP. Reusing scales or
  pre-quantized weights across the two formats shifts every value by 2×
  and moves the clamp limit. Name the exact format in code; "fp8" alone
  is ambiguous.
- Compute `amax` in fp32, derive the scale, clamp to the format max, then
  cast. Without the clamp, overflow saturates or becomes NaN depending on
  the format.
- Pick the finest scale granularity the kernel can afford (per tensor <
  per channel < per block). A per-tensor scale on a heavy-tailed
  activation clips the tail or flushes small values.
- Block-scaled (MX) formats carry a shared power-of-two scale per block.
  The block size and scale layout are part of the numeric contract, not
  a tuning knob.
- Fusing `amax` + quantize into the producer saves a pass. Keep the
  `amax` reduction in fp32 when you fuse it.

## 5. Validate against an fp32 reference

- Dequantize every low-precision input to fp32 and run the reference math
  in fp32. Kernel-vs-kernel agreement (cosine 0.99 between two fast
  kernels) says they agree. It does not say either is correct.
- Report several metrics: cosine, max abs, mean abs, RMSE, relative RMSE,
  and mean signed error. Cosine alone hides bias and large local errors
  (see [measurement](../method/measurement.md)).
- Sweep cells: small and large batch, short and long reduction length.
  Rounding error grows with reduction length, so a short-context pass can
  hide a long-context failure. Check that log-sum-exp has no inf or NaN at
  the longest length.
- When a tuned reference kernel exists, score both it and your kernel
  against the fp32 reference. If yours is closer, the speedup carries no
  precision cost.
- Gate autotuning candidates on the same error metric, so a faster but
  wrong config is never selected
  (see [tuning-methodology](../autotuning/tuning-methodology.md)).

## Pitfalls

- Accumulator down-cast inside the K-loop: accuracy loss, no speed gain.
- bf16 or fp16 softmax denominator or norm sum: drift, sometimes NaN at
  long lengths.
- A recipe tuned for OCP fp8 run on FNUZ hardware (or the reverse): ~2×
  error everywhere, with plausible-looking output.
- Comparing to a formula when the reference makes deliberate precision
  trade-offs (e.g. `exp2` with a folded `log2(e)`, fast reciprocal square
  root). Match the reference's math, then compare.

Sources: [ROCm precision support](https://rocm.docs.amd.com/en/latest/reference/precision-support.html) · [CDNA3 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) · [CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) · [Online normalizer for softmax, arXiv 1805.02867](https://arxiv.org/abs/1805.02867) · [FlashAttention, arXiv 2205.14135](https://arxiv.org/abs/2205.14135) · [FP8 formats for deep learning, arXiv 2209.05433](https://arxiv.org/abs/2209.05433) · [Microscaling data formats, arXiv 2310.10537](https://arxiv.org/abs/2310.10537) · Welford, "Note on a method for calculating corrected sums of squares and products", Technometrics 4(3), 1962

---
Related: [define-the-problem](../method/define-the-problem.md) · [matrix-instructions](matrix-instructions.md) · [epilogue](epilogue.md) · [kernel-fusion](kernel-fusion.md)
