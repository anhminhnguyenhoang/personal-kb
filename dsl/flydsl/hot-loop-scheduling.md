---
name: hot-loop-scheduling
description: How to know whether a FlyDSL hot-loop schedule (sched_mfma/sched_vmem/sched_dsrd/sched_dswr) is effective — the measure→diagnose→lever→re-measure loop, the roofline gate, and why memory-bound kernels win on algorithm not schedule.
source: FlyDSL hot_loop_scheduler (examples/04-preshuffle_gemm.py) and aiter small_m_hgemm
---

## Hot-loop scheduling: how to know what's effective

`sched_*(X)` is an **ordering hint**, not work. Each call lowers to
`sched_group_barrier(MASK, X, 0)` (`flydsl/expr/rocdl.py`): MASK picks the
machine (mfma `0x008` / vmem-rd `0x020` / ds-rd `0x100` / ds-wr `0x200`), `X` is
a **count of lowered instructions** to place in that group, syncID is 0. So `X`
is a tally across the whole unrolled stage — one `fx.gemm` lowers to *many* MFMA,
one `fx.copy` of a tile to *many* loads. The schedule order is the **desired
hardware order**, deliberately different from source call order (prefetch is
written first but scheduled mid-iteration so MFMAs run on top of the in-flight
load).

### You don't guess once — you measure into existence

```
   ① find the bottleneck  →  ② pull the matching lever  →  ③ re-measure  ──┐
                          └──────────── repeat until it stops improving ◄──┘
```

### ① Diagnose first (never tune blind)

| Profiler / ISA says | Means | Fix |
|---|---|---|
| MFMA util LOW, waiting on memory | cores **starve** — data late | prefetch earlier, add pipeline stages |
| VMEM ~100%, MFMA waits on it | **memory-bound** — maxing DRAM | schedule won't help; bigger tiles / data reuse |
| MFMA ~100% | **compute-bound** ✅ | done — stop tuning |
| LDS bank conflicts high | ds_reads serialize | swizzle (xor), *not* a sched fix |

### ② Levers (symptom → knob)

- Cores idle right after a load (latency not hidden) → **more `sched_mfma()` after each `sched_vmem()`** (raise mfma:load ratio).
- Load issued too late → **move `sched_vmem()` earlier** (prefetch sooner).
- One big stall at the top → **spread loads across the iteration**, don't dump them at once.
- Still stalling with 2 buffers → **add a 3rd/4th pipeline stage** (a loop change, not a sched change).

### ③ Roofline gate — is faster even possible?

```
   compute_time ≈ total_MFMA_work / matrix_core_throughput
   memory_time  ≈ (bytes_A + bytes_B) / DRAM_bandwidth

   compute_time > memory_time → COMPUTE-bound → schedule CAN help, chase MFMA≈100%
   memory_time  > compute_time → MEMORY-bound → schedule won't beat DRAM; change
                                                the ALGORITHM (reuse, split-K, tiles)
```
A kernel can never beat `max(compute_time, memory_time)`. If you're near it, stop.

### The two regimes, made concrete

- **Compute-bound (big square GEMM, example 04):** hand-write the schedule, chase MFMA≈100%. Fixed tile size → the winning mfma:load ratio is written out like sheet music.
- **Memory-bound (skinny M≤16 decode, aiter small_m):** the schedule helps only a little. Real wins are **algorithmic** — split-K, N-tile reuse, async DRAM→LDS DMA, cross-block L2 reuse. Cut HBM traffic; don't tune compute.

### Why compute the schedule instead of hand-writing it

When hundreds of autotuned tile shapes each want a different ratio, a **formula**
derives the counts per shape and the **autotuner** times them all and keeps the
winner (small_m: `MFMA_TOTAL = WARP_K_STEPS*WARP_M_STEPS*WARP_N_STEPS`,
`LDG_TOTAL = LDG_REG_A_COUNT_AS + LDG_REG_B_COUNT_AS`, then `sched_mfma(2)` per
load + `sched_mfma(1)` for the remainder so `Σ sched_mfma == MFMA_TOTAL`).
Autotuning = ② and ③ done automatically, thousands of times.

### Rule-of-thumb ladder (cheapest → most effort)

1. Measure first — know if compute- or memory-bound.
2. Memory-bound → fix the **algorithm** (reuse, stages, split-K), not the sched.
3. Compute-bound → tune the mfma:load ratio until MFMA≈100%.
4. Can't hand-pick one ratio → compute it from tile sizes + autotune.
5. Always re-measure; a "smart" schedule that's slower is a bug.

> Caveat: `sched_barrier` / `sched_group_barrier` have no ISA encoding on any
> arch; they print only as asm comments. Judge a hint by the instruction order
> it produces, not by finding barriers in the ISA. See
> [pipelining-scheduling](../../kernel-optimization/levers/pipelining-scheduling.md).

---
Related: [measurement](../../kernel-optimization/method/measurement.md) · [bottleneck-classification](../../kernel-optimization/method/bottleneck-classification.md) · [pipelining-scheduling](../../kernel-optimization/levers/pipelining-scheduling.md)
