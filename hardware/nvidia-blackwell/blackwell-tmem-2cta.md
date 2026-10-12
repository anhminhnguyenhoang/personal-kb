---
name: blackwell-tmem-2cta
description: The Blackwell hardware features FA4 stands on — TCGEN05/UMMA async tensor cores, 256KB Tensor Memory (TMEM), and 2-CTA cluster MMA. Why they enable deeper pipelines than Hopper.
source: _sources/flash-attention-v4-research.md (synthesis; see footer for primary links)
---

# Blackwell's new primitives: TCGEN05, TMEM, and 2-CTA MMA

FA4 is not a port of FA3 with tweaks — it is rebuilt around three Blackwell
features that did not exist on Hopper. Knowing what they *are* makes the
pipeline note read cleanly.

## TCGEN05 / UMMA — fully asynchronous tensor cores

Blackwell's new tensor-core instruction family (`tcgen05.mma`, generically
"UMMA"). Two properties matter:

- **Fully asynchronous.** A warp issues an MMA and immediately moves on; the
  result lands later in Tensor Memory. This is what makes warp specialization
  *worth it* — you can keep multiple MMAs in flight while other warps do
  element-wise work.
- **Single-thread launch.** One thread launches a UMMA (vs Hopper's WGMMA which
  a whole warpgroup drives). This **eases register pressure** dramatically.
- **Bigger atoms.** Largest single-CTA UMMA tile is **128×256×16** — roughly
  **2× the largest Hopper WGMMA atom**. Bigger tiles → better MMA efficiency,
  fewer instructions.

## TMEM — Tensor Memory (the key enabler)

Each SM gets **256 KB of Tensor Memory**: an on-chip scratchpad wired directly
into the tensor cores, for **warp-synchronous** intermediate storage.

This is the load-bearing change. On Hopper (FA3), MMA accumulators live in
**registers** — and attention has a lot of them (S scores, O output, plus
softmax stats), so register pressure forces a more *serial* schedule and risks
spilling. On Blackwell, the accumulators **S, P, O (and in backward dS, dQ)
live in TMEM instead.** That frees the register file, which is exactly what lets
FA4 run:

- larger tiles (the 128×256×16 atoms above),
- deeper pipelines (more stages in flight),
- dedicated warp roles (softmax / correction warpgroups that each need registers).

The cost: softmax and correction warps must explicitly **load S from TMEM into
registers**, do the non-matrix math, and **copy results back** — TMEM is not
directly ALU-addressable. That TMEM↔register round-trip is part of why the
softmax warpgroup sequence has the shape it does.

## 2-CTA MMA mode

A single UMMA can span a **pair of CTAs in the same cluster**, partitioning the
output accumulator across the two. Example: a **256×256×16** result by splitting
M and N across the pair, where each CTA only stages **half of operand B**.

Why it matters: it **halves redundant shared-memory traffic** and lowers
per-CTA footprint — directly attacking the backward pass's SMEM-bandwidth
bottleneck (see [the roofline](asymmetric-hardware-scaling.md)). It pairs with
**DSMEM** (distributed shared memory — CTAs in a cluster can read each other's
SMEM), which the backward pass uses to exchange half of `dS` and fix the dQ
reduction-axis mismatch.

## FA3 vs FA4 at a glance

| Aspect | FA3 (Hopper) | FA4 (Blackwell) |
|---|---|---|
| MMA | WGMMA, warpgroup-driven | UMMA / tcgen05, single-thread, async |
| Accumulators | registers (pressure, spilling) | **TMEM** (256 KB/SM) |
| Largest atom | ~64×128×16 | **128×256×16** (~2×) |
| Cross-CTA MMA | none | **2-CTA cluster MMA** + DSMEM |
| Pipeline | 2-stage | **~5-stage** warp-specialized |

Sources:
- [FlashAttention-4 (Tri Dao)](https://tridao.me/blog/2026/flash4/)
- [Colfax Research deep-dive](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/)
- [Modal: reverse-engineering FA4](https://modal.com/blog/reverse-engineer-flash-attention-4)
- [Lambda: FA4 on Blackwell](https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet)

---
Back: [asymmetric scaling](asymmetric-hardware-scaling.md) · Next: [warp-specialized pipeline](../../papers/flashattention-4/02-algorithm/warp-specialization-pipeline.md)
