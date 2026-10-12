---
name: warp-specialization-pipeline
description: FA4's 5-stage warp-specialized async pipeline — load / MMA / 8 softmax / 4 correction / epilogue warps, ping-pong over two Q tiles, coordinated by SMEM barrier arrays through GMEM→SMEM→TMEM→SMEM→GMEM.
source: _sources/flash-attention-v4-research.md (synthesis; see footer for primary links)
---

# The 5-stage warp-specialized pipeline

FA3 ran a **2-stage** pipeline. FA4 runs a **~5-stage** warp-specialized async
pipeline — the structural change that lets softmax, correction, MMA, and memory
all run concurrently instead of serializing. It's only practical because of
[Blackwell's async UMMA + TMEM](../../../hardware/nvidia-blackwell/blackwell-tmem-2cta.md):
single-thread MMA launch and TMEM accumulators free the registers that warp
specialization needs.

## The five warp roles

Different warps are assigned **different jobs** and run as a producer/consumer
pipeline. Per Modal's reverse-engineering, the specializations are:

| Stage | Warps | Job |
|---|---|---|
| **Load** | 1 load warp | Async **TMA** loads: brings Q tile in, then **streams all K,V tiles** global→shared. Signals completion via barrier array. |
| **MMA** | 1 MMA warp | `tcgen05.mma` (UMMA): runs **Q×K→S** and **P×V→O**; accumulators in **TMEM**. |
| **Softmax** | **8 warps = 2 warpgroups** (128 threads each) | One online-softmax step per S tile: reduce row-max/row-sum, compute `exp` (HW or [software](software-exponential.md)), make P, →BF16. |
| **Correction** | **4 warps = 1 warpgroup** | Rescale already-accumulated O when the stability factor changes ([conditional rescaling](conditional-softmax-rescaling.md)) — kept **off the critical path**. |
| **Epilogue** | 1–2 warps | Store finished O tile shared→global (count depends on whether TMA is used). |

The official paper's framing of the same thing: **two softmax warpgroups, one
correction warpgroup, and one warpgroup driving the tensor cores + TMA units.**

## Data flow through the memory hierarchy

Each stage consumes its predecessor's output and signals completion via **an
array of barriers in shared memory** (referenced by *offset* into the array,
which supports variable barrier counts):

```
GMEM  --Load(TMA)-->  SMEM  --MMA(UMMA)-->  TMEM(S)
      --Softmax-->  P (exp, BF16) in TMEM  --MMA(UMMA)-->  TMEM(O)
      --Correction-->  rescaled O  --Epilogue-->  SMEM  --> GMEM
```

Note the TMEM round-trips: softmax/correction must **load S/O from TMEM into
registers**, do the element-wise math, and write back — TMEM feeds the tensor
cores, not the ALUs directly.

## The softmax warpgroup's inner sequence

1. Load a **128-element row of S** from TMEM → registers.
2. Reduce **row_max** and **row_sum**.
3. Compute `exp` — partitioned between **hardware MUFU.EX2** and the
   **[software polynomial](software-exponential.md)** path.
4. Convert **P → BF16**, store back to TMEM **in stages**.
5. Trigger the **P∘V MMA** at ~**3/4 completion** of the P store (so the MMA
   warp can start before the whole tile is written — overlap).

Crucial detail: the two softmax warpgroups are **explicitly synchronized so they
never evaluate `exp` at the same time** → halves instantaneous SFU contention.

## Ping-pong scheduling

Each CTA works **two query tiles** — call them `Q^H` and `Q^L`, 128 tokens each
— and **alternates (ping-pongs)** between them, just like FA3. While `Q^H`'s
tensor-core matmuls run, `Q^L`'s softmax runs, and vice-versa. This is what keeps
**both the tensor cores and the SFUs continuously busy** — the whole point, given
that on Blackwell the two are co-equal in cost.

## Why this needs Blackwell

On Hopper (FA3) accumulators sit in registers; the resulting register pressure
forces a **more serial** schedule and limits how many warp roles you can carve
out. Blackwell's **TMEM** moves accumulators off-register and **single-thread
UMMA** cuts the per-MMA register footprint, so FA4 can afford dedicated softmax
*and* correction warpgroups *and* deep pipelining without spilling. The schedule
is a consequence of the hardware, not a free choice.

Sources:
- [Modal: reverse-engineering FA4](https://modal.com/blog/reverse-engineer-flash-attention-4) (5 warp roles, barrier array)
- [FlashAttention-4 (Tri Dao)](https://tridao.me/blog/2026/flash4/)
- [Colfax Research deep-dive](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/)

---
Related: [TMEM & 2-CTA](../../../hardware/nvidia-blackwell/blackwell-tmem-2cta.md) · [software-exp](software-exponential.md) · [conditional-softmax](conditional-softmax-rescaling.md) · [backward & perf](../03-backward-and-perf/backward-pass.md)
