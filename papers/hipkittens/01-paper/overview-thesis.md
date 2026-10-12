---
name: overview-thesis
description: "HipKittens paper deep-dive: the portability thesis, 5 CDNA hardware facts, and a summary of every design response + headline numbers."
source: HipKittens paper (arXiv 2511.08083), HazyResearch blog, and HipKittens GitHub repo
---

Here's a deep walkthrough of the **HipKittens** work — what they discovered about AMD silicon, the design choices that fell out of those discoveries, and the numbers that justify them.

---

## 1. The thesis

The paper's controlling claim is small but consequential:

> **Tile-based abstractions generalize across GPU vendors. The *algorithms* that instantiate them do not.**

ThunderKittens (TK) on NVIDIA proved you can write competitive Hopper kernels in ~hundreds of lines of C++ by handing the user three nouns — *tiles*, *bulk operators*, *async load/store* — and one verb pattern — *warp specialization with shared-memory deep pipelines*. HK keeps the nouns. The verb pattern is thrown out and replaced with two new schedules tailored to CDNA's static register file. Everything else in the paper is downstream of that decision.

## 2. The five hardware facts that shape every design choice

These five CDNA3/CDNA4 properties are what the paper repeatedly returns to:

| # | Hardware fact | Consequence |
|---|---|---|
| 1 | Each SIMD has 512×32-bit registers, **statically partitioned across all resident waves** (256 VGPRs + 256 AGPRs) | Producer/consumer wave specialization burns half your register budget on waves that never compute → cap at 80% of peak BF16 GEMM on a 256×256 tile |
| 2 | HIPCC **refuses to use AGPRs as input operands to MFMA** (only as outputs) | A v_accvgpr_read shuffle is inserted, costing ~19% on attention backward unless you hand-pin registers |
| 3 | MFMA instructions come in **heterogeneous shapes** (16×16×32, 32×32×16, …), each with an entirely different thread-ownership pattern — unlike NVIDIA's compositional 16×16 building block | No single swizzle works for the whole kernel; layout must be solved per *pair* of co-occurring instructions |
| 4 | `buffer_load` is the AMD analogue of TMA, but it's **per-thread address-driven**, not descriptor-driven | Swizzling has to be applied to *HBM addresses*, not to LDS layout |
| 5 | MI355X is **8 chiplets (XCDs), each with its own 4 MB L2**, behind a shared LLC | Naïve row-major scheduling gets 36 % L2 hit rate; you need a 2D-window-over-XCD remap |

Everything novel in the paper is a response to one of these.

## 3. Schedules: why wave-specialization dies and what replaces it

### Why it dies
Wave specialization on Hopper works because (a) WGMMA pulls operands from shared memory so producers can hand off via SMEM, (b) there's a `setmaxnreg` to *reallocate* registers from idle producers to busy consumers, and (c) async-copy + barrier is a first-class concept. **AMD has none of these.** Static register partition means producer waves permanently steal register budget from consumers, which shrinks the output tile, which kills throughput. They measured 80 % of peak BF16 GEMM on a 256×256 tile under this scheme — and it gets worse on attention because attention needs even more registers.

### 8-wave ping-pong (the default)
Two resident waves per SIMD alternate roles via a conditional barrier: while wave A is grinding MFMA on its tile, wave B is prefetching the next K-block into LDS; then they swap. Both waves contribute compute, so no register is wasted on a pure-producer. This is enough to reach **peak on BF16/FP8 GEMM and attention forward** — at ~500 LOC for attention forward.

### 4-wave interleave (the heavier hammer)
For imbalanced workloads (GQA non-causal backward is the canonical bad citizen — register-heavy, mixed MFMA shapes), one wave per SIMD does *intra-wave* fine-grained interleaving of load/compute at small tile granularity. Costs you in code size (GQA-bwd: 331 LOC for 8-wave → **989 LOC** for 4-wave) but recovers another ~20 % over 8-wave. This is the schedule that produces the headline 2.8–4.0× win over AITER on GQA backward.

The takeaway is conceptual: **AMD pipelines through instruction-level interleaving rather than through SMEM depth**. Hopper hides latency behind big SMEM buffers; CDNA hides it behind fine-grained MFMA ↔ memory issue interleaving inside a single wave.

## 4. Swizzling: there is no universal answer

The bank-conflict problem on AMD has more dimensions than on NVIDIA:

- LDS has **64 banks for ds_read_b128, 32 banks for ds_read_b96** — bank count itself depends on the instruction.
- The **phase order** in which threads sweep banks is instruction-dependent and *non-sequential*.
- The A and B operand layouts for the same MFMA differ, and a kernel that uses two MFMA shapes (e.g., 16×16×32 for the main GEMM and 32×32×16 for an epilogue) must satisfy *both*.

HK doesn't try to solve the general problem. They **enumerate the common instruction-pair combinations** (e.g. row-major `ds_read_b128` paired with `ds_read_b64_tr_b16` for the transpose) and ship a precomputed swizzle per pair — for the example above, a column-swap of cols 8–15 ↔ 0–7 starting at row 8 kills both sides' conflicts. Register tiles default to the smallest MFMA shape (16×16×32) because it gives the compiler maximum scheduling freedom, and devs opt up explicitly when they want it.

Global memory is a separate beast: since `buffer_load` is per-thread-address, **the swizzle is baked into the HBM address calculation**, not the LDS write path. That's a real conceptual inversion from the TMA-shaped world.

## 5. Chiplet-aware scheduling — Algorithm 1

This is one of the most generally-useful pieces of the paper because chiplets are the future direction for both vendors (B200 is 2 dies, MI355X is 8).

The hierarchy on MI355X:
- 8 XCDs × 32 CUs each, **private 4 MB L2 per XCD**
- shared LLC sits between the L2s and HBM

The conflict: **L2 wants spatial locality within a chiplet, LLC wants spatial locality across chiplets.** Row-major thread-block scheduling satisfies neither well — measured at 36 % L2 hit rate on a 9216³ BF16 GEMM.

The algorithm is two passes done at launch by remapping block IDs:

1. **XCD grouping** — re-bin the flattened block-ID space so that *C* consecutive logical blocks land on the same XCD (the hardware round-robins block IDs across XCDs, so this is the inverse remap).
2. **Hierarchical windowed traversal** — instead of walking the output matrix row-major, sweep it in vertical windows of height *W*, so that the consecutive-block group from step 1 lands on overlapping A/B rows and columns.

Tunable parameters W, C let you trade L2 vs LLC pressure. On the 9216³ GEMM:

| Schedule | L2 hit | LLC hit | Bandwidth |
|---|---|---|---|
| Naive row-major | 55 % | 95 % | 15.1 TB/s |
| Algo 1 (W=5, C=25) | **75 %** | 93 % | **18.3 TB/s** |

**+19 % perf**, no kernel changes — just the launch-time remap.

## 6. AGPR pinning — the compiler workaround

Because HIPCC won't let AGPRs feed MFMA inputs, HK exposes a `pinned_register_tile<dtype, rows, cols, start_vgpr, start_agpr>` that names exact registers and bypasses the compiler's register-class checks via inline asm. On attention backward at seq_len 8192 this recovered **909 → 1091 TFLOPS (+19 %)**. The honest framing in the paper: this is a compiler workaround you'd rather not need, but it's only expressible because HK is willing to drop into assembly when the C++ surface fails.

## 7. The numbers worth remembering

All numbers below are MI355X (CDNA4) unless noted:

- **BF16 GEMM (8192³)**: 1281 TFLOPS (8-wave) — for reference CUTLASS on B200 hits 1570 TFLOPS, so the gap to Blackwell is roughly silicon-proportional.
- **FP8 GEMM**: 3222 TFLOPS (8-wave) / 3327 TFLOPS (4-wave).
- **vs Triton BF16 GEMM**: 1.3–3.0× faster.
- **GQA causal forward (d=128)**: 1.0–2.1× vs AITER.
- **GQA non-causal backward (batch 16, seq 8192)**: 1091 TFLOPS (HK 4-wave) vs **272–384 TFLOPS (AITER)** → **2.8–4.0×**.
- **vs PyTorch SDPA backward**: 1.3–4.5×.
- **vs Mojo MHA**: 2× — Mojo is bank-conflict-bound at 50 % peak.
- **Memory-bound kernels** (fused dropout-residual-LN, RoPE): 1.1–2.4× over AITER.

Two of these are the rhetorically loaded ones: **AITER GQA-backward at 30 % of SoTA** and **PyTorch SDPA at 24 %** are the headline indictments of hand-written assembly's scalability problem. AITER doesn't even *have* a d=64 GQA-backward variant — which is the paper's strongest argument that DSLs aren't a nice-to-have, they're the only way to cover the matrix of (dtype × shape × variant) combinations the model zoo demands.

## 8. Honest framing & open problems

The blog (not in the arxiv) is more candid:
- **Some bank-conflict-avoidance behavior is undocumented in the CDNA ISA.** They reverse-engineered phase orderings.
- They explicitly position HK as following the H100 trajectory: it took ~2 years between H100 launch and SoTA open-source attention kernels; HK is trying to compress that loop for MI300/MI350.
- AMD ecosystem assessment is blunt: AITER hand-asm doesn't scale across shapes, Triton-on-AMD mis-tracks register lifetimes, TileLang is CDNA3-only and depends on Composable Kernel, Mojo's MHA is bank-conflict-bound.
- **Open problem**: most non-GEMM/non-attention AI workloads on AMD are still under-optimized; HK demonstrates the method but not the catalogue.

## 9. Why this matters for *this* repo

A few notes since you're working in aiter:
- `csrc/kernels/mla/hk/` is HK applied to **DeepSeek-MLA FP8 decode**, which is *not* a kernel the paper benchmarks. So it's a real test of whether the HK abstractions generalize past GEMM/attention to a more idiosyncratic workload (single KV head, 512 LoRA rank, RoPE-split).
- The gating in `aiter/mla.py:470` (gfx942/gfx950, fp8, small `nhead * max_seqlen_q`) maps cleanly to the regime where HK's 8-wave ping-pong is most likely to dominate ASM — small-Q decode is exactly where the ASM kernel's monolithic scheduling can't beat fine-grained MFMA/memory interleave.
- The paper's "AITER GQA-bwd at 30 % SoTA" framing is a finger pointed at this codebase. For your sparse-MLA work — where you've already noted in memory that **3D split-K + reduce beats ASM mla_decode_fwd** at low CU occupancy — the HK paper is essentially the general theory of why the ASM path keeps losing: it can't recover register-budget the way a schedule-tuned DSL kernel can.

## Adoption

The aiter library uses HipKittens for one experimental kernel: FP8 MLA (multi-head latent attention) decode. It is selected only on gfx942 (MI300X) and gfx950 (MI350X/MI355X), for FP8 queries and KV cache in the small-query-token regime, and only when experimental kernels are enabled; otherwise aiter falls back to its assembly MLA decode kernel.

Sources:
- [HipKittens: Fast and Furious AMD Kernels (arxiv 2511.08083)](https://arxiv.org/abs/2511.08083)
- [HipKittens HTML paper](https://arxiv.org/html/2511.08083v1)
- [Hazy Research blog post](https://hazyresearch.stanford.edu/blog/2025-11-09-hk)
- [HazyResearch/HipKittens (GitHub)](https://github.com/HazyResearch/HipKittens)
- [AMD GPUs go brrr (companion blog)](https://hazyresearch.stanford.edu/blog/2025-11-09-amd-brr)



---
Deeper dives: [schedules](schedules.md) · [hipcc-agpr-pinning](hipcc-agpr-pinning.md) · [chiplet-scheduling](chiplet-scheduling.md)
