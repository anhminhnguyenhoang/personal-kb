---
name: conditional-softmax-rescaling
description: FA4's online-softmax change — only rescale the running output O when the row max jumps past threshold τ, skipping ~90% of corrections, with final normalization preserving correctness.
source: _sources/flash-attention-v4-research.md (synthesis; see footer for primary links)
---

# Conditional online-softmax rescaling

The first of FA4's two *algorithm* tricks. It removes most of the rescaling work
that online softmax normally does on the critical path. Cuts corrections by
**~10×** (Tri Dao, Hot Chips 2025).

## The standard online-softmax tax

FlashAttention processes K/V in tiles and maintains a running output `O` and
running row-max `m`. Classic online softmax (FA1–FA3): **every time a tile
produces a new running max `m_j > m_{j-1}`, you must rescale the entire
accumulated output** by `exp(m_{j-1} - m_j)` to keep it on the new scale:

```
O_j = exp(m_{j-1} - m_j)·O_{j-1} + exp(S_j - m_j)·V_j
```

That `exp(m_{j-1}-m_j)·O_{j-1}` rescale is a vector op over the whole O tile,
and it happens on the critical path on *every* max change. On Blackwell, where
the exp unit is already the bottleneck
([roofline](../../../hardware/nvidia-blackwell/asymmetric-hardware-scaling.md)), this is exactly
the work you cannot afford.

## The FA4 rule: rescale only on a big jump

Introduce a tunable threshold **τ**. Rescale only when the max actually jumps
enough to threaten numerical stability; otherwise accumulate against the *old*
max and defer:

```
O_j = exp(m_{j-1} - m_j)·O_{j-1} + exp(S_j - m_j)·V_j,   if  m_j - m_{j-1} > τ
O_j = O_{j-1}            +          exp(S_j - m_{j-1})·V_j,   otherwise
```

In the `otherwise` branch there is **no rescale of `O_{j-1}`** — you just add the
new contribution computed against `m_{j-1}`. Because tile maxima usually only
nudge the running max by a little, the big-jump branch fires rarely → **~90% of
rescales are skipped**.

## Why it stays correct

Two reasons:

1. **Final normalization absorbs the scale.** Online softmax already divides by
   the running denominator at the end: `O_final = O / l_final`. As long as `O`
   and `l` are accumulated on a *consistent* scale between rescale points, the
   final divide produces the exact softmax. Skipping a rescale just means more
   terms share one scale reference — still consistent.
2. **τ guards stability.** The only risk of accumulating against a stale, smaller
   max is `exp(S_j - m_{old})` overflowing when `S_j` is much larger. τ is set so
   that whenever the gap is large enough to risk overflow, the big-jump branch
   *does* fire and rescales. Small gaps can't overflow, so skipping is safe.

## Warp-granularity decision

The skip is decided **per warp, not per element.** If *no thread in the warp*
needs a rescale (all gaps ≤ τ), the entire warp skips the correction. This is
what makes it cheap — it's a warp-uniform branch, not a masked per-lane op — and
it's why FA4 has a **dedicated correction warpgroup** ([pipeline
note](warp-specialization-pipeline.md)): the rescales that *do* fire are handed
to that warpgroup so they happen **off the critical path** of the softmax/MMA
loop.

## Relationship to software exp

Conditional rescaling reduces the *number* of exp-based corrections; the
[software-emulated exponential](software-exponential.md) reduces the *cost per
exp* by moving it off the SFU. Together they attack the forward-pass exp
bottleneck from both sides.

Sources:
- [FlashAttention-4 (Tri Dao)](https://tridao.me/blog/2026/flash4/)
- [Colfax Research deep-dive](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/)
- [SemiAnalysis on the Hot Chips softmax algo](https://x.com/SemiAnalysis_/status/1963255669613592921)
- [Modal: reverse-engineering FA4](https://modal.com/blog/reverse-engineer-flash-attention-4)

---
Related: [software-exponential](software-exponential.md) · [warp-specialization pipeline](warp-specialization-pipeline.md) · back to [overview](../01-design/overview.md)
