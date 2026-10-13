---
name: fmha-backward
description: "CK-Tile FlashAttention backward as idioms: the D = rowsum(dO∘O) pre-kernel, five GEMMs per tile with S recomputed, the kr_ktr_vr layout that serves K and Kᵀ from one LDS copy, transpose-load and IGLP variants, and deterministic dQ via reduce-then-cast."
source: https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/fmha/pipeline ; https://arxiv.org/abs/2307.08691
---

# CK-Tile FMHA backward

Backward has its own pipelines and traits. Forward tuning does not carry over.

## 1. Three kernels

```
 ① dot_do_o     D[q]   = Σ_j dO[q,j] · O[q,j]        (cheap, row reduce; optional sink grad)
 ② dq_dk_dv     main kernel, loops over the other sequence axis
 ③ convert_dq   fp32 dQ accumulators → output dtype (optionally reduce splits first)
```
Computing D first turns a reduction inside the main loop into a plain input tile.

## 2. Main kernel: five GEMMs per tile

Outer loop over a K/V tile (dK, dV stay in registers); inner loop over Q tiles:

| # | GEMM | Result |
|---|---|---|
| 0 | `Q · Kᵀ` | S, recomputed (S is never stored by forward) |
| — | `P = exp(S·scale − LSE)` | from saved forward LSE, no running max needed |
| 1 | `Pᵀ · dO` | dV += |
| 2 | `dO · Vᵀ` | dP |
| — | `dS = P ∘ (dP − D)` | elementwise |
| 3 | `dSᵀ · Q` | dK += |
| 4 | `dS · K` | dQ += (crosses K/V tiles → accumulate in fp32, atomics or splits) |

Saved forward state is only O and LSE (per row). That keeps forward writes small.

## 3. Layout tricks

- **`kr_ktr_vr`** (the name encodes how K, Kᵀ and V are held). K is stored in LDS
  pre-shuffled so one allocation can be read both as K and as Kᵀ.
- **`trload_*` variants** read the transposed operand with `ds_read_tr*`, so Kᵀ, Qᵀ and
  dOᵀ never pass through ALU shuffles.
- **`trload_qr_qtr_dor`** flips the loop for decode-like shapes: Q, dO, LSE, D stay in LDS
  and are replayed per K block.
- **IGLP variant**: same math with compiler-guided interleaving; the selector uses it when a
  head dim is padded.

The selector picks exactly one pipeline per problem class at compile time:
`{transpose-load or not} × {training, decode, padded head dim}`.

## 4. Deterministic dQ

- Default: dQ partials from different K/V tiles meet via atomic add (fast,
  order-dependent).
- Deterministic mode: each split writes its own fp32 dQ slab; `convert_dq` sums the
  `nsplits` slabs in registers in a fixed order, then casts once. Bitwise reproducible,
  costs workspace and one extra pass.

## 5. Scope note

In CK-Tile only FMHA has a hand-optimized backward. Norm, fused add-norm-quant and Welford
ops are forward-only; frameworks build their backward elsewhere.

Sources: [ck_tile fmha pipelines](https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/fmha/pipeline) ·
[01_fmha example (fmha_bwd)](https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/01_fmha) ·
[FlashAttention-2 (arXiv 2307.08691)](https://arxiv.org/abs/2307.08691)
Paths drift; verify before use. Verify specifics against the primary source.

---
Related: [fmha-forward](fmha-forward.md) · [backward-pass (FA-4)](../../papers/flashattention-4/03-backward-and-perf/backward-pass.md)
