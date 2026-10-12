# FlashAttention-4 — research source material

Raw research notes synthesized from public sources for the `papers/flashattention-4/`
KB topic. Unlike the HipKittens notes, this topic has no single validated
prose source — it is a synthesis of the primary write-ups below, cross-checked
against each other. Each KB note carries a `Sources:` footer with the primary
links. This file preserves the consolidated research for provenance.

## Primary sources

- Tri Dao, "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for
  Asymmetric Hardware Scaling" — https://tridao.me/blog/2026/flash4/
- Together AI blog — https://www.together.ai/blog/flashattention-4
- Colfax Research deep-dive —
  https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/
- Modal, "We reverse-engineered Flash Attention 4" —
  https://modal.com/blog/reverse-engineer-flash-attention-4
- arXiv HTML — https://arxiv.org/html/2603.05451v1
- Lambda blog —
  https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet
- Princeton AI blog —
  https://blog.ai.princeton.edu/2026/03/12/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/
- SemiAnalysis (Hot Chips 2025 summary) —
  https://x.com/SemiAnalysis_/status/1963255669613592921

## Timeline

- Hot Chips 2025: Tri Dao announces FA4, reports ~22% faster than cuDNN on Blackwell.
- Code dropped on GitHub (Dao-AILab/flash-attention) before the paper.
- 2026-03-05: full FA4 paper published.

## 1. Asymmetric hardware scaling (the motivation)

Core trend: accelerator functional units scale asymmetrically. H100 → B200:

- BF16 dense tensor-core throughput: **1 → 2.25 PFLOPs** (>2x).
- SFU (special function unit) count: **unchanged**.
- Shared-memory bandwidth: **unchanged**.

Consequence: non-MMA units (exp, SMEM traffic, ALU) become the bottleneck.
Roofline shows softmax SMEM traffic and exponential ops now exceed MMA compute
by 25–60% for typical attention shapes.

Roofline per SM, M=N=D=128, forward pass:
- Tensor cores: 8192 ops/cycle → **1024 cycles** per tile.
- Exponential unit: 16 ops/cycle → **1024 cycles** per tile.
- Shared memory: 128 bytes/cycle → 768 cycles per tile.
- Bottleneck: compute and exponential TIED at 1024 cycles each.

Backward pass (1-CTA), per SM:
- Tensor cores: 2560 cycles (five MMAs).
- Exponential: 1024 cycles.
- Shared memory: **3328 cycles** → SMEM bandwidth dominates.

## 2. Blackwell hardware features leveraged

- **TCGEN05 / UMMA**: new fully-asynchronous tensor-core instructions; a single
  thread launches the MMA (eases register pressure). Largest single-CTA UMMA
  tile = **128×256×16**, ~2x the largest Hopper WGMMA atom.
- **Tensor Memory (TMEM)**: 256 KB per SM on-chip scratchpad wired into tensor
  cores, for warp-synchronous intermediate storage. Accumulators (S, P, O, dS,
  dQ) live here instead of registers — this is what makes deep pipelines /
  large tiles practical (Hopper kept accumulators in registers → spilling).
- **2-CTA MMA mode**: one UMMA spanning a CTA pair in the same cluster; e.g.
  256×256×16 by splitting M and N across the pair. Halves operand-B SMEM traffic.
- **DSMEM** (distributed shared memory): lets CTAs in a cluster read each
  other's SMEM; used in backward dQ reduction.

## 3. Forward pipeline: warp specialization (5-stage)

FA3 was a 2-stage pipeline. FA4 is a ~5-stage warp-specialized pipeline:
load → MMA → softmax → correction(rescale) → epilogue/store.

Modal's reverse-engineering counts five warp specializations:
1. **Load warp** — async TMA loads Q tile + streams all K,V tiles
   global→shared. Signals via barrier array.
2. **MMA warp** — tcgen05.mma (cta_group::1): runs Q×K→S and P×V→O matmuls,
   accumulators in TMEM.
3. **Eight softmax warps** = two warpgroups (128 threads each) — one online-
   softmax step per S tile in a loop.
4. **Four correction warps** = one warpgroup — rescale prior O outputs as the
   numerical-stability factor changes (off critical path).
5. **One/two epilogue warps** — store O shared→global (count depends on TMA).

Official paper framing: two softmax warpgroups, one correction warpgroup, one
warpgroup driving tensor cores + TMA.

Softmax warpgroup sequence:
1. Load 128-element row of S from TMEM to registers.
2. Reduce row_max and row_sum.
3. exp via hardware MUFU.EX2 or software-emulated path (partitioned).
4. Convert P to BF16, store to TMEM in stages.
5. Trigger P∘V MMA at ~3/4 completion of P storage.

Critical sync: the two softmax warpgroups are explicitly synchronized so they
do NOT evaluate exp simultaneously → reduces MUFU contention.

Async pipeline = producer/consumer with an array of SMEM barriers (referenced
by offset to support variable barrier counts). Data flow:
GMEM →(load)→ SMEM →(MMA)→ TMEM →(softmax/correction)→ SMEM →(epilogue)→ GMEM.

Ping-pong: two query tiles per CTA (Q^H, Q^L), 128 tokens each, alternate so
one tile's tensor-core work overlaps the other's softmax (similar to FA3).

## 4. Software-emulated exponential

Problem: MUFU.EX2 (SFU) is the bottleneck; few SFUs vs many CUDA cores → queueing.
FA4 distributes exp2 across MUFU.EX2 AND FMA units.

Algorithm (compute 2^x):
- Cody-Waite range reduction: 2^x = 2^n · 2^f, n = floor(x), f = x − n ∈ [0,1).
  Integer part = exponent-field update (cheap).
- Fractional part via cubic polynomial (Horner form, 3 FMAs):
  2^f ≈ p0 + p1·f + p2·f² + p3·f³,
  p0 = 1.0, p1 ≈ 0.69514614, p2 ≈ 0.22756439, p3 ≈ 0.07711909.
- Final: shift n into exponent field, add mantissa bits of 2^f.

Throughput trick: `fma.rn.ftz.f32x2` operates on TWO f32 values at once →
two exponentials computed simultaneously. PTX:
```
fma.rn.ftz.f32x2 l10, l9, l6, l5
fma.rn.ftz.f32x2 l10, l10, l9, l4
fma.rn.ftz.f32x2 l10, l10, l9, l3
```
Coefficient idea traces to Schraudolph 1999 (Neural Computation) fast-exp, but
FA4 uses a cubic polynomial to match hardware precision.

Blending: software path applied only on SOME iterations with a tunable
frequency, and stops on a configurable number of the LAST S tiles (where
precision matters most) — those use hardware exp2. Mostly relevant for smaller
head sizes.

## 5. Conditional online softmax rescaling

Standard online softmax rescales O every time the running row max changes.
FA4 only rescales when the new max jumps by more than threshold τ:

```
O_j = exp(m_{j-1} - m_j)·O_{j-1} + exp(S_j - m_j)·V_j,   if m_j - m_{j-1} > τ
O_j = O_{j-1} + exp(S_j - m_{j-1})·V_j,                   otherwise
```

Decision made at warp granularity (if no thread in the warp needs rescale, the
whole warp skips). Correctness preserved by the final normalization
O_final = O / l_final. Reduces rescaling/correction operations by ~10×
(Tri Dao, Hot Chips 2025). The correction warpgroup performs the rescales it
can't skip, off the critical path.

## 6. Backward pass

- Recomputes S, P in transposed layout (S^T, P^T) directly in TMEM in the
  operand-A layout dV/dK MMAs consume. S/P share one TMEM region; dP/dS/dQ
  share another (five accumulators across pipeline stages).
- Pipeline overlap: while computing softmax for tile j, already issue dK and dQ
  MMAs for tile j−1.
- 2-CTA: partition M=256 across the pair, N=K=128 → halves operand-B SMEM
  traffic. dQ reduction-axis mismatch resolved via DSMEM exchange of half of dS
  → dQ MMA becomes (M/2, 2N)×(2N, d); halves global atomic reductions.
- Deterministic mode: serialize atomic dQ updates with semaphore locks + fences;
  CTA swizzling + shortest-processing-time ordering. ~85–90% of nondeterministic
  throughput.
- Scheduling: causal masking swizzles batch-heads into L2-sized sections,
  iterating blocks reverse (longest-first) to avoid short-first imbalance.
  Varlen: preprocessing kernel sorts batches by max per-worktile time (LPT
  heuristic), metadata cached.

## 7. Implementation & performance

- Written entirely in **CuTe-DSL** (Python; lowers to PTX). ~20–30× faster
  compile vs C++ templates. Install in seconds.
- Forward BF16 on B200: up to **1605 TFLOPs/s (71% utilization)**; first
  attention kernel reported to break the **petaflop** barrier.
- vs cuDNN 9.13: 1.1–1.3× faster forward (Tri Dao). Earlier reports ~20–22%
  vs cuDNN at Hot Chips. NOTE: cuDNN has since absorbed many of these ideas and
  now offers similar perf.
- vs Triton: 2.1–2.7× faster forward.
- vs FA3: ~2× (and ~15× vs original FA on its target shapes).
- Caveat: cuDNN wins at 1K–2K seqlen, especially causal (FA4 ~208 TFLOPs vs
  cuDNN ~315 reported in one benchmark).

## 8. Limitations / status

- Forward-first, Blackwell-first. Backward exists in the paper but newer.
- Initially lacked varlen, GQA/MQA; BF16-only; did not yet use FP4 or
  (originally) 2-CTA in all paths.
- Recommended adoption: Blackwell inference, fixed-length, standard attention,
  behind a feature flag; keep cuDNN/SDPA + FA3 (Hopper) / FA2 (Ampere/Ada) as
  fallbacks.
