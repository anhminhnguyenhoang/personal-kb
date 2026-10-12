---
name: overview
description: FA4 in one note — the thesis (co-design algorithm + pipeline for asymmetric hardware), the four headline techniques, performance numbers, and what's still missing. The map of the whole topic.
source: _sources/flash-attention-v4-research.md (synthesis; see footer for primary links)
---

# FlashAttention-4 — overview & map

FlashAttention-4 (FA4) is Tri Dao's attention kernel rebuilt for NVIDIA
**Blackwell** (B200, SM10.x). Announced at **Hot Chips 2025** (~22% faster than
cuDNN), code shipped on GitHub first, full **paper published 2026-03-05**. This
note is the map; the deep notes are linked inline.

## The thesis (one line)

> When tensor cores scale faster than everything else, the attention kernel
> must be **co-designed at the algorithm *and* pipeline level** to move work off
> the now-scarce units (exp, shared memory) and onto the abundant ones (FMA,
> tensor cores).

FA4 is not "FA3 + Blackwell intrinsics." The roofline changed shape
([why](../../../hardware/nvidia-blackwell/asymmetric-hardware-scaling.md)), so the schedule was
re-derived. Two of the four headline tricks are **algorithm** changes (software
exp, conditional rescaling); two are **pipeline/hardware** changes (5-stage warp
specialization, TMEM + 2-CTA). That algorithm↔pipeline co-design is the paper's
subtitle and its actual contribution.

## The four headline techniques

| # | Technique | Attacks | Gist | Note |
|---|---|---|---|---|
| 1 | **Software-emulated exponential** | exp-unit (fwd) bottleneck | Compute `exp2` with a cubic polynomial on **FMA units** instead of the SFU; `f32x2` does two at once | [software-exp](../02-algorithm/software-exponential.md) |
| 2 | **Conditional online-softmax rescaling** | exp/correction overhead | Only rescale O when row-max jumps past threshold τ → **~10× fewer** corrections | [conditional-softmax](../02-algorithm/conditional-softmax-rescaling.md) |
| 3 | **5-stage warp-specialized pipeline** | softmax↔MMA serialization | load / MMA / softmax / correction / epilogue warps; ping-pong two Q tiles | [warp-specialization](../02-algorithm/warp-specialization-pipeline.md) |
| 4 | **TMEM + 2-CTA MMA** | SMEM bandwidth (bwd) | accumulators in Tensor Memory, MMA spanning a CTA pair halves operand traffic | [tmem-2cta](../../../hardware/nvidia-blackwell/blackwell-tmem-2cta.md), [backward](../03-backward-and-perf/backward-pass.md) |

## How a forward tile flows

```
GMEM --load warp(TMA)--> SMEM --MMA warp(UMMA)--> TMEM(S)
   --softmax warpgroups--> P (exp, →BF16) --MMA--> TMEM(O)
   --correction warpgroup--> rescale O (only if max jumped) 
   --epilogue warp--> SMEM --> GMEM
```

Two query tiles (Q^H, Q^L, 128 tokens each) **ping-pong** per CTA so one tile's
tensor-core work overlaps the other's softmax — keeping both the tensor cores
and the SFUs continuously busy. The two softmax warpgroups are deliberately
**desynchronized on `exp`** so they don't both hammer the SFU at once.

## Numbers worth remembering

All B200, BF16, forward unless noted:

- **1605 TFLOPs/s (71% utilization)** — first attention kernel reported to break
  the **petaflop** barrier.
- **vs cuDNN 9.13**: 1.1–1.3× (forward). Hot Chips reported ~20–22% vs cuDNN.
  *Caveat:* cuDNN has since absorbed many of these ideas and now matches FA4.
- **vs Triton**: 2.1–2.7× (forward).
- **vs FA3**: ~2× (and ~15× vs the original FlashAttention on its target shapes).
- **Where it loses**: cuDNN wins at **1K–2K seqlen**, especially causal (one
  benchmark: FA4 ~208 vs cuDNN ~315 TFLOPs). FA4's wins grow with sequence length.
- Backward: consistently beats baselines at large seqlen; **deterministic mode**
  retains ~85–90% of nondeterministic throughput.

## Implementation

Written **entirely in CuTe-DSL** (Python, lowers to PTX) — **~20–30× faster
compile** than C++ templates, install in seconds. This is itself a thesis: it
lowers the barrier to prototyping new attention variants without C++ template
metaprogramming. (Compare HK's opposite bet — drop to inline asm when C++ fails.)

## What's missing (status)

Forward-first, Blackwell-first. As of the paper:
- Backward exists but is newer; **varlen** and **GQA/MQA** were initially absent.
- **BF16-only**; did not yet exploit **FP4**.
- Adoption guidance: Blackwell **inference**, fixed-length, standard attention,
  behind a feature flag; keep **cuDNN/SDPA + FA3 (Hopper) / FA2 (Ampere/Ada)**
  as fallbacks until backward/varlen/GQA mature.

Sources:
- [FlashAttention-4 (Tri Dao)](https://tridao.me/blog/2026/flash4/)
- [arXiv HTML 2603.05451](https://arxiv.org/html/2603.05451v1)
- [Colfax Research deep-dive](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/)
- [Modal: reverse-engineering FA4](https://modal.com/blog/reverse-engineer-flash-attention-4)
- [Together AI blog](https://www.together.ai/blog/flashattention-4)

---
Deeper: [software-exp](../02-algorithm/software-exponential.md) · [conditional-softmax](../02-algorithm/conditional-softmax-rescaling.md) · [warp-specialization](../02-algorithm/warp-specialization-pipeline.md) · [backward & perf](../03-backward-and-perf/backward-pass.md) · [glossary](../glossary.md)
