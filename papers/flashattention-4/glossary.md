---
name: glossary
description: Every FlashAttention-4 / Blackwell term used across this topic, grouped by area — hardware, instructions/memory, and algorithm concepts.
source: compiled from the FA4 notes in this topic; primaries in _sources/flash-attention-v4-research.md
---

# FlashAttention-4 / Blackwell glossary

Grouped by area; within a group, terms build on each other.

## Hardware: units & chips

| Term | Meaning |
|---|---|
| **Blackwell** | NVIDIA's GPU architecture after Hopper. FA4's target. Chips: **B200**, GB200. SM version **SM10.x**. |
| **B200** | The Blackwell datacenter GPU FA4 is benchmarked on. ~2.25 PFLOPs BF16 dense tensor-core throughput. |
| **H100 / Hopper** | The prior-gen GPU FA3 targeted. 1 PFLOP BF16. Baseline for "asymmetric scaling." |
| **SM (Streaming Multiprocessor)** | The core NVIDIA compute block (NVIDIA's analogue of AMD's CU). |
| **Tensor core** | The matrix-multiply unit. Blackwell's is >2× H100's — the change that triggered everything. |
| **SFU (Special Function Unit)** | Hardware for transcendentals (`exp`, `sin`, …). Far fewer than CUDA cores; **did not scale** on Blackwell → the forward bottleneck. |
| **CUDA core / FMA unit** | General-purpose fused-multiply-add ALUs. Abundant; FA4 offloads `exp` onto them. |
| **Asymmetric hardware scaling** | The trend that tensor cores scale faster than SFU/SMEM/ALU, so non-MMA work becomes the bottleneck. FA4's whole motivation. |

## Instructions & memory

| Term | Meaning |
|---|---|
| **MMA** | Matrix-Multiply-Accumulate (the tensor-core operation). |
| **UMMA / TCGEN05** | Blackwell's new **fully asynchronous** tensor-core instruction family (`tcgen05.mma`). Single-thread launch; largest single-CTA tile **128×256×16**. |
| **WGMMA** | Hopper's warpgroup-driven MMA (FA3). Higher register pressure than UMMA. |
| **TMEM (Tensor Memory)** | 256 KB/SM on-chip scratchpad wired into the tensor cores; holds accumulators (S, P, O, dS, dQ). The feature that frees registers and enables FA4's deep pipeline. |
| **2-CTA MMA** | One UMMA spanning a **pair of CTAs** in a cluster; splits M/N across the pair, halving operand-B SMEM traffic. |
| **CTA** | Cooperative Thread Array = a thread block. |
| **Cluster** | A group of CTAs that can share SMEM (via DSMEM) — Hopper+ concept. |
| **DSMEM (distributed shared memory)** | Lets CTAs in a cluster read each other's SMEM; FA4 backward uses it to exchange half of `dS`. |
| **SMEM** | Shared memory. Its bandwidth **did not scale** on Blackwell → the backward bottleneck. |
| **TMA** | Tensor Memory Accelerator — async bulk global↔shared copy engine; FA4's load warp uses it. |
| **MUFU.EX2** | The SASS instruction (on the SFU) that the `exp2` PTX intrinsic compiles to. The unit FA4 tries to bypass. |
| **exp2 / f32x2** | Base-2 exponential intrinsic. `f32x2` = an FMA variant operating on **two f32 at once** (computes two exps in parallel). |
| **CuTe-DSL** | The Python DSL FA4 is written in; lowers to PTX. ~20–30× faster compile than C++ templates. |
| **PTX / ptxas / SASS** | NVIDIA's virtual ISA / its assembler / the native machine ISA. |

## Algorithm concepts

| Term | Meaning |
|---|---|
| **FlashAttention** | IO-aware exact attention that tiles K/V and never materializes the full N×N score matrix. |
| **Online softmax** | Computing softmax in one streaming pass with a running max `m` and running denominator `l`, rescaling the output `O` when the max changes. |
| **Conditional rescaling** | FA4's change: rescale `O` only when `m_j − m_{j-1} > τ`; skips ~90% of corrections; final `O/l` divide preserves correctness. |
| **τ (tau)** | The tunable threshold gating a rescale; set so any overflow-risking max jump triggers a rescale. |
| **Correction warpgroup** | The 4-warp group dedicated to performing the rescales that *do* fire, off the critical path. |
| **Software-emulated exponential** | Computing `2^x` via Cody-Waite range reduction + a **cubic polynomial** on FMA units instead of the SFU. |
| **Cody-Waite range reduction** | Split `2^x = 2^n·2^f` (n=floor(x)); integer part is a free exponent-field update, fractional part via polynomial. |
| **Horner form** | Nested polynomial evaluation `((p3·f + p2)·f + p1)·f + p0` — degree-3 = 3 FMAs. |
| **Warp specialization** | Assigning different warps different fixed jobs (load / MMA / softmax / correction / epilogue) in a producer/consumer pipeline. |
| **Warpgroup** | 4 warps = 128 threads acting as a unit (e.g. 2 softmax warpgroups, 1 correction warpgroup). |
| **Ping-pong scheduling** | Alternating two query tiles per CTA so one tile's MMA overlaps the other's softmax. |
| **5-stage pipeline** | load → MMA → softmax → correction → epilogue (vs FA3's 2-stage). |
| **Deterministic mode** | Reproducible backward via serialized atomic dQ updates; ~85–90% of nondeterministic throughput. |
| **LPT / SPT** | Longest-/Shortest-Processing-Time-first scheduling heuristics for load balancing causal & varlen work. |
| **Varlen** | Variable sequence length batching. |
| **GQA / MQA** | Grouped-/Multi-Query Attention (KV-head sharing). Initially absent from FA4. |

Sources:
- [FlashAttention-4 (Tri Dao)](https://tridao.me/blog/2026/flash4/)
- [Colfax Research deep-dive](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/)
- [Modal: reverse-engineering FA4](https://modal.com/blog/reverse-engineer-flash-attention-4)

---
Back to [topic README](README.md)
