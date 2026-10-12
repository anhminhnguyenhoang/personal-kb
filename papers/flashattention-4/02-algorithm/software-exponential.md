---
name: software-exponential
description: FA4 computes exp2 with a cubic polynomial on FMA units instead of the SFU's MUFU.EX2, doing two at once with f32x2, blended with the hardware path on the precision-critical last tiles.
source: _sources/flash-attention-v4-research.md (synthesis; see footer for primary links)
---

# Software-emulated exponential

The second algorithm trick, and the most quoted one. Softmax needs an
exponential per score; on Blackwell the SFU that computes it is the
[bottleneck](../../../hardware/nvidia-blackwell/asymmetric-hardware-scaling.md). FA4's fix:
**stop sending every exp to the SFU — compute most of them on the FMA units.**

## Why the SFU is the problem

`exp(x)` in CUDA is normally done via the `exp2` PTX intrinsic, which `ptxas`
maps to the **`MUFU.EX2`** SASS instruction running on the **Special Function
Units (SFUs)**. There are **far fewer SFUs than CUDA cores**, so when softmax
floods them, requests **queue**. Tensor cores doubled; SFU count did not — so
the exp queue is now co-equal with MMA at **1024 cycles/tile**. Meanwhile the
FMA / CUDA-core ALUs sit relatively idle. FA4 moves the work to where the
capacity is.

## The algorithm: 2^x on FMA units

Compute `2^x` (softmax uses base-2 throughout) with classic range reduction +
polynomial, all in FMAs:

**1. Cody-Waite range reduction.** Split `2^x = 2^n · 2^f` where `n = floor(x)`
and `f = x − n ∈ [0,1)`. The integer part `2^n` is *free* — it's just an update
to the float's **exponent field** (shift `n` into the exponent bits).

**2. Cubic polynomial for the fractional part.** Approximate `2^f` on the unit
interval with a degree-3 polynomial in **Horner form** (3 FMAs):

```
2^f ≈ p0 + p1·f + p2·f² + p3·f³
   p0 = 1.0
   p1 ≈ 0.69514614
   p2 ≈ 0.22756439
   p3 ≈ 0.07711909
```

**3. Reassemble.** Shift `n` into the exponent field and add the mantissa bits
of `2^f`. Done — a full `2^x` with zero SFU traffic.

The idea of fast software exp via bit tricks goes back to **Schraudolph (1999,
*Neural Computation*)**; FA4's contribution is the **cubic** polynomial chosen to
match hardware precision (Schraudolph's original was much coarser).

## Two at once: `f32x2`

Blackwell FMAs operate on **pairs of f32 values** via the `f32x2` variant, so
each FMA evaluates the polynomial for **two exponentials simultaneously** —
doubling exp throughput again. The Horner evaluation in PTX:

```
fma.rn.ftz.f32x2 l10, l9,  l6, l5   # p3·f + p2
fma.rn.ftz.f32x2 l10, l10, l9, l4   # ·f + p1
fma.rn.ftz.f32x2 l10, l10, l9, l3   # ·f + p0
```

## Blending with the hardware path (precision)

FA4 does **not** abandon `MUFU.EX2`. It **blends**:

- The software polynomial path runs on **only some iterations**, at a **tunable
  frequency**, and is **disabled on a configurable number of the *last* S
  tiles** — the tiles whose contribution dominates the final softmax and where
  precision matters most use the **hardware** exp.
- This is most relevant for **smaller head sizes**, where the exp pressure
  relative to MMA is highest.
- The two softmax warpgroups are **explicitly desynchronized on exp** so they
  don't both hit the SFU at the same instant — reducing MUFU contention even for
  the hardware-path exps.

Net effect: spread `exp` across **both** MUFU.EX2 and FMA, with hardware used
exactly where the polynomial's error would matter, hitting hardware-comparable
accuracy at much higher throughput.

Sources:
- [FlashAttention-4 (Tri Dao)](https://tridao.me/blog/2026/flash4/)
- [Modal: reverse-engineering FA4](https://modal.com/blog/reverse-engineer-flash-attention-4) (PTX, coefficients)
- [Colfax Research deep-dive](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/)
- Schraudolph, N. (1999). "A Fast, Compact Approximation of the Exponential Function." *Neural Computation* 11(4).

---
Related: [conditional-softmax](conditional-softmax-rescaling.md) · [warp-specialization pipeline](warp-specialization-pipeline.md) · [glossary](../glossary.md)
