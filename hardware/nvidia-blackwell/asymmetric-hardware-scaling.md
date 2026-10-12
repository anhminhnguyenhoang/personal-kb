---
name: asymmetric-hardware-scaling
description: Why FA4 exists — Blackwell scaled tensor cores 2.25x but left SFUs and SMEM bandwidth flat, so softmax/exp became the attention bottleneck. The roofline numbers.
source: _sources/flash-attention-v4-research.md (synthesis; see footer for primary links)
---

# Asymmetric hardware scaling — the problem FA4 was built to solve

Every FlashAttention-4 design choice is downstream of one observation about
Blackwell silicon. **Read this first** — the algorithm and pipeline notes only
make sense once you see which hardware unit became scarce.

## The controlling fact

From Hopper **H100 → Blackwell B200**, the functional units scaled *unevenly*:

| Unit | H100 | B200 | Change |
|---|---|---|---|
| BF16 dense tensor-core throughput | 1 PFLOP | **2.25 PFLOPs** | **>2×** |
| SFU count (special function units — `exp`, etc.) | — | — | **unchanged** |
| Shared-memory bandwidth | — | — | **unchanged** |

The tensor cores got >2× faster. The units that do *everything else in
attention* — the exponential for softmax, and the shared-memory traffic that
shuttles tiles around — did not. So the moment you double MMA throughput, the
**non-MMA work becomes the bottleneck.** Roofline analysis put softmax SMEM
traffic and exponential ops at **25–60% above** MMA compute for typical
attention shapes.

This is what the paper means by *asymmetric hardware scaling*, and it is a
general trend, not a Blackwell quirk: tensor cores are scaling faster than the
surrounding silicon every generation. Attention kernels that were MMA-bound on
Hopper become exp-bound and SMEM-bound on Blackwell.

## The roofline, in cycles

The cleanest way to see it. Per SM, square tile **M = N = D = 128**, forward pass:

| Resource | Rate | Cycles per tile |
|---|---|---|
| Tensor cores | 8192 ops/cycle | **1024** |
| Exponential unit (SFU) | 16 ops/cycle | **1024** |
| Shared memory | 128 bytes/cycle | 768 |

The exponential unit is now **tied with the tensor cores at 1024 cycles** — softmax
is no longer "the cheap thing between two matmuls," it is co-equal with the
matmul and must be pipelined just as carefully.

Backward pass (1-CTA) is worse on a *different* axis:

| Resource | Cycles per tile |
|---|---|
| Tensor cores (five MMAs) | 2560 |
| Exponential | 1024 |
| **Shared memory** | **3328** ← bottleneck |

So FA4 fights two different enemies: **the exponential unit in the forward
pass, shared-memory bandwidth in the backward pass.** Each gets its own fix
(software-emulated exp for the former, 2-CTA MMA + TMEM layout for the latter).

## The three responses (map to the rest of the topic)

| Bottleneck | FA4 response | Note |
|---|---|---|
| Exp unit saturated (fwd) | Emulate `exp2` on FMA units; conditional rescaling cuts corrections ~10× | [software-exp](../../papers/flashattention-4/02-algorithm/software-exponential.md), [conditional-softmax](../../papers/flashattention-4/02-algorithm/conditional-softmax-rescaling.md) |
| SMEM bandwidth (bwd) | 2-CTA MMA halves operand traffic; TMEM holds accumulators | [tmem-2cta](01-blackwell-tmem-2cta.md), [backward](../../papers/flashattention-4/03-backward-and-perf/backward-pass.md) |
| Serial dependency of softmax↔MMA | 5-stage warp-specialized async pipeline | [warp-specialization](../../papers/flashattention-4/02-algorithm/warp-specialization-pipeline.md) |

## Why this rhymes with the AMD story

If you've read the [HipKittens notes](../../papers/hipkittens/README.md): both papers
are the same shape of argument. HK's thesis is "tile abstractions port, but the
*schedule* must be re-derived per architecture." FA4 is the NVIDIA-side instance
of the same lesson one generation later — the FA3 schedule was tuned for a
balanced H100, and on an *imbalanced* B200 you have to re-derive which unit to
hide behind which. The enemy just moved from register pressure (AMD) to the
exp/SMEM units (Blackwell).

Sources:
- [FlashAttention-4 (Tri Dao)](https://tridao.me/blog/2026/flash4/)
- [Colfax Research deep-dive](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/)
- [Together AI blog](https://www.together.ai/blog/flashattention-4)
- [Modal: reverse-engineering FA4](https://modal.com/blog/reverse-engineer-flash-attention-4)

---
Next: [Blackwell TMEM & 2-CTA MMA](01-blackwell-tmem-2cta.md) · then the [algorithm tricks](../../papers/flashattention-4/02-algorithm/conditional-softmax-rescaling.md)
