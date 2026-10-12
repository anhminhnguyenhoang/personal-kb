# FlashAttention-4

Notes on **FlashAttention-4 (FA4)** — Tri Dao's attention kernel rebuilt for
NVIDIA **Blackwell** (B200, SM10.x). Announced at Hot Chips 2025; paper published
**2026-03-05**. FA4 is the first attention kernel reported to break the
**petaflop** barrier (~1605 TFLOPs/s, 71% B200 utilization).

All content is **synthesized from public write-ups** — Tri Dao's blog, the
Together AI and Colfax Research deep-dives, Modal's reverse-engineering post, the
arXiv paper, and Hot Chips coverage. Unlike a transcript-derived topic, there is
no single validated prose source, so every note carries a `Sources:` footer with
primary links; the consolidated research is in
[`../_sources/flash-attention-v4-research.md`](../../_sources/flash-attention-v4-research.md).

> Verify specifics (coefficients, cycle counts, perf deltas) against the
> primary sources before citing — these are synthesized from secondary coverage
> and the field moves fast (cuDNN has already absorbed several FA4 ideas).

## Reading order

Start with the hardware — every FA4 trick is a response to one Blackwell fact.

### 00 · Fundamentals (the hardware, in `hardware/nvidia-blackwell/`)
- [asymmetric-hardware-scaling](../../hardware/nvidia-blackwell/asymmetric-hardware-scaling.md) — **read this first.** Tensor cores scaled 2.25× but SFUs and SMEM didn't, so exp (forward) and SMEM bandwidth (backward) became the bottleneck. The roofline in cycles.
- [blackwell-tmem-2cta](../../hardware/nvidia-blackwell/01-blackwell-tmem-2cta.md) — TCGEN05/UMMA async tensor cores, 256 KB Tensor Memory, and 2-CTA cluster MMA; why they enable deeper pipelines than Hopper.

### 01 · Design overview
- [overview](01-design/overview.md) — the co-design thesis, the four headline techniques in one table, performance numbers, and what's still missing. **The map of the whole topic.**

### 02 · The algorithm & pipeline tricks
- [conditional-softmax-rescaling](02-algorithm/conditional-softmax-rescaling.md) — rescale O only when the row max jumps past τ → ~10× fewer corrections; why it stays correct.
- [software-exponential](02-algorithm/software-exponential.md) — compute `exp2` with a cubic polynomial on FMA units (two at once via `f32x2`), blended with the hardware SFU path on precision-critical tiles.
- [warp-specialization-pipeline](02-algorithm/warp-specialization-pipeline.md) — the 5-stage async pipeline: load / MMA / 8 softmax / 4 correction / epilogue warps, ping-pong over two Q tiles.

### 03 · Backward pass & performance
- [backward-pass](03-backward-and-perf/backward-pass.md) — SMEM-bound backward, fixed with transposed TMEM recompute + 2-CTA MMA + DSMEM dS exchange; deterministic mode; LPT scheduling; full perf recap & limitations.

### Reference
- [glossary](glossary.md) — every term, grouped by hardware / instructions / algorithm.

## One-sentence thesis

> When tensor cores scale faster than the units around them, an attention kernel
> must be **co-designed at the algorithm *and* pipeline level** — FA4 moves the
> exponential off the saturated SFU onto FMA units, skips ~90% of softmax
> rescales, and uses Blackwell's TMEM + async UMMA to run a 5-stage
> warp-specialized pipeline that keeps both the tensor cores and the SFUs busy.
