---
name: backward-pass
description: FA4 backward — SMEM-bandwidth-bound, fixed with transposed TMEM recompute, 2-CTA MMA + DSMEM dS exchange to halve operand traffic and dQ atomics, plus deterministic mode and LPT scheduling.
source: _sources/flash-attention-v4-research.md (synthesis; see footer for primary links)
---

# Backward pass, scheduling, and performance

The backward pass fights a **different** bottleneck than the forward. From the
[roofline](../../../hardware/nvidia-blackwell/asymmetric-hardware-scaling.md): backward (1-CTA)
is **3328 SMEM cycles** vs 2560 tensor-core cycles vs 1024 exp — **shared-memory
bandwidth dominates**, not the exp unit. So the backward optimizations are all
about cutting SMEM traffic, not about exp.

## TMEM transposed-recompute

FlashAttention recomputes S and P in the backward pass (cheaper than storing
them). FA4 recomputes them in **transposed layout (Sᵀ, Pᵀ)** and stores them
**directly in TMEM in the exact operand-A layout** that the `dV` and `dK` MMAs
consume — so no transpose shuffle through SMEM is needed before the matmul.

TMEM is partitioned to fit **five accumulators** across pipeline stages:
- **S and P share one TMEM region.**
- **dP, dS, dQ share another.**

## Pipeline overlap

Same idea as forward, applied to gradients: *while computing softmax for tile j,
already issue the `dK` and `dQ` MMAs for tile j−1.* This hides the exp latency
behind the previous tile's matmuls.

## 2-CTA MMA + the dQ reduction problem

[2-CTA MMA](../../../hardware/nvidia-blackwell/blackwell-tmem-2cta.md) partitions **M = 256**
across a CTA pair with **N = K = 128**, so each CTA only stages **half of
operand B** → **halves operand-B SMEM traffic** (directly attacking the
bottleneck).

But `dQ` has a **reduction-axis mismatch**: `dQ` reduces over the outer-loop **N**
dimension, which is *already split across the CTA pair*. FA4 fixes this with a
**DSMEM exchange**: the two CTAs swap half of `dS` over distributed shared
memory, re-partitioning along the **non-reduction** axis. Each CTA then owns
**M/2 rows** while holding the **full 2N = 256** reduction, so the dQ MMA becomes
`(M/2, 2N) × (2N, d)`. Result: **halves the global atomic reductions** for dQ
accumulation.

## Deterministic mode

Training reproducibility needs deterministic dQ accumulation. FA4 serializes the
global atomic dQ updates with **semaphore-style locks + memory fences**, using
**CTA swizzling** and **shortest-processing-time-first (SPT)** ordering for the
causal mask. Cost: deterministic backward runs at **~85–90% of the
nondeterministic** throughput.

## Load-balanced scheduling

- **Causal masking** creates triangular load imbalance. FA4 swizzles batch-heads
  into **L2-sized sections** and iterates blocks in **reverse** so tiles are
  processed **longest-first** — avoids the tail-latency penalty of short-first.
- **Variable sequence length (varlen)**: a **preprocessing kernel** sorts
  batches by **max per-worktile execution time** and applies a
  **longest-processing-time-first (LPT)** heuristic to minimize tail latency.
  The schedule metadata is **cached** to avoid recompute overhead.

## Performance recap (B200, BF16)

| Comparison | FA4 result |
|---|---|
| Forward absolute | **1605 TFLOPs/s (71% util)** — first to break **1 PFLOP** |
| Forward vs cuDNN 9.13 | **1.1–1.3×** (Hot Chips: ~20–22%); cuDNN has since caught up |
| Forward vs Triton | **2.1–2.7×** |
| vs FA3 | **~2×** (~15× vs original FA on its shapes) |
| Backward | beats baselines at large seqlen; det. mode ~85–90% |
| **Where it loses** | cuDNN wins at **1K–2K seqlen**, esp. causal (FA4 ~208 vs ~315 TFLOPs) |

Takeaway: FA4's edge **grows with sequence length**. At short seqlen the pipeline
fill/drain and scheduling overhead don't amortize, and cuDNN wins.

## Status / limitations

Forward-first, Blackwell-first. Initially **BF16-only**, no **FP4**; **varlen**
and **GQA/MQA** added after the first forward-only drop. Recommended deployment:
Blackwell **inference**, fixed-length, standard attention, behind a feature flag,
with **cuDNN/SDPA + FA3 (Hopper) / FA2 (Ampere/Ada)** as fallbacks.

Sources:
- [FlashAttention-4 (Tri Dao)](https://tridao.me/blog/2026/flash4/)
- [Colfax Research deep-dive](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/)
- [arXiv HTML 2603.05451](https://arxiv.org/html/2603.05451v1)
- [Modal: reverse-engineering FA4](https://modal.com/blog/reverse-engineer-flash-attention-4)

---
Related: [overview](../01-design/overview.md) · [TMEM & 2-CTA](../../../hardware/nvidia-blackwell/blackwell-tmem-2cta.md) · [warp-specialization](../02-algorithm/warp-specialization-pipeline.md) · [glossary](../glossary.md)
