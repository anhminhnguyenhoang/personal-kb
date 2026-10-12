---
name: glossary
description: Merged, deduplicated glossary of every HipKittens / CDNA term used across the notes in this topic.
source: HipKittens paper (arXiv 2511.08083), HazyResearch blog, and HipKittens GitHub repo
---

# HipKittens / CDNA Glossary

Merged and deduplicated from the individual walkthrough glossaries. Terms are grouped by area; within each group they build on each other.

## Hardware: compute

| Term | Meaning |
|---|---|
| **ALU** | Arithmetic Logic Unit. The silicon that does +, ×, etc. on one pair of operands per cycle. |
| **Vector ALU / SIMD unit** | A bank of N parallel ALU lanes driven by one instruction. CDNA's is **16 lanes wide**. |
| **Lane** | One column of a vector ALU; runs one thread's data path. |
| **SIMD** | A 16-wide vector ALU with its own register file (512 regs) and its own MFMA pipeline. 4 per CU. |
| **SIMD width / lane count** | Number of physical lanes in one vector unit. **16 on CDNA.** |
| **CU (Compute Unit)** | A processor containing 4 SIMDs, a shared LDS, and schedulers. MI355X has 256 CUs. NVIDIA analogue: SM. |
| **Wave / Wavefront** | 64 threads executing in lockstep on one SIMD. AMD's equivalent of a NVIDIA warp (32 threads). |
| **Wave/warp size** | Number of threads that move together. **64 on AMD, 32 on NVIDIA.** |
| **Cycles per wave instruction** | `wave_size ÷ lane_count`. **4 on CDNA** (64/16). |
| **Quad-cycle design** | CDNA's 64-thread-wave × 16-lane-SIMD × 4-cycle-issue choice. Hides ALU pipeline latency by construction. |
| **SIMD round-robin** | The CU schedules its 4 SIMDs one per cycle → one wave's worth of work issued every cycle despite each SIMD taking 4 cycles per wave. |
| **wave32 / wave64** | RDNA (graphics) can run in either mode; CDNA (compute) is wave64 only. |
| **Threadblock / Workgroup** | The set of waves that co-reside on one CU, share LDS, and sync via `s_barrier`. HIP "block" = AMD "workgroup". |

## Hardware: registers & memory

| Term | Meaning |
|---|---|
| **VGPR** | Vector general-purpose register; per-thread. 256 per wave at 2 waves/SIMD. Can be MFMA input or output. |
| **AGPR** | Accumulator register; 256 per wave. Hardware allows MFMA input, but HIPCC won't emit it — costs ~19% until you pin registers manually. |
| **Static register partition** | The 512 regs of a SIMD are divided evenly across all resident waves at compile time; not transferable. *The single fact that shapes both HK schedules.* |
| **LDS (Local Data Share)** | AMD's shared memory (per-CU, ~64 KB). 64 banks for 128-bit reads, 32 banks for 96-bit. |
| **Double buffering** | Keep two physical tile buffers in LDS so loader and computer don't race. Used by ping-pong. |
| **HBM** | High-bandwidth memory — the off-chip DRAM. The thing you're trying to avoid hitting. |
| **XCD (Accelerator Complex Die)** | A chiplet inside an MI355X. 8 per GPU, 32 CUs each, with a **private 4 MB L2 cache**. |
| **L2** | Cache local to one XCD. Fast (~3× LLC bandwidth) but only helps blocks on the same XCD. |
| **LLC (Last-Level Cache)** | The single shared cache between all XCDs' L2s and HBM. Slower but visible to everyone. |
| **L2/LLC tension** | Orthogonal cache preferences: L2 wants intra-XCD locality, LLC wants inter-XCD overlap. Optimizing one alone breaks the other. |

## Instructions

| Term | Meaning |
|---|---|
| **MFMA** | Matrix Fused Multiply-Add — the matrix-multiply instruction. Shapes like 16×16×32 mean M×N×K dims. Inputs from VGPRs, accumulates into AGPRs. |
| **`v_accvgpr_read`** | The instruction HIPCC inserts to copy AGPR → VGPR before MFMA input. Pure overhead. |
| **`buffer_load`** | HBM → LDS load instruction. AMD's nearest-to-TMA, but per-thread-addressed (so swizzling is baked into the HBM address). |
| **`ds_read` / `ds_write`** | LDS → registers / registers → LDS. |
| **`s_barrier`** | Threadblock-wide synchronization. Used for ping-pong phase boundaries. |
| **`s_waitcnt vmcnt/lgkmcnt`** | Wait for outstanding global-memory / LDS-memory operations. Used for interleaved scheduling. |

## Compiler & DSL

| Term | Meaning |
|---|---|
| **HIPCC** | AMD's HIP compiler driver, a thin wrapper around Clang/LLVM that emits CDNA assembly. AMD's analogue of `nvcc`. |
| **`pinned_register_tile`** | HK primitive that names exact register numbers (incl. AGPRs as MFMA inputs), bypassing HIPCC's allocator via inline asm. |
| **ThunderKittens (TK)** | HazyResearch's NVIDIA tile-DSL; HK is the AMD/HIP sibling. Keeps the *nouns* (tiles, bulk ops, async load/store), replaces the *verb* (wave specialization). |

## Schedules

| Term | Meaning |
|---|---|
| **8-wave ping-pong** | 2 waves per SIMD; they swap compute/load roles each phase via `s_barrier` + double-buffered LDS. Both waves always compute. Good for balanced kernels. |
| **4-wave interleave** | 1 wave per SIMD (512 regs); the wave interleaves load + MFMA instructions itself via `s_waitcnt`. Good for register-heavy, mixed-shape kernels (e.g. GQA backward). |
| **Wave specialization** | NVIDIA pattern: dedicated producer + consumer warps, with `setmaxnreg` register donation. Fails on AMD because of the static register split. |

## Chiplet scheduling (Algorithm 1)

| Term | Meaning |
|---|---|
| **Algorithm 1** | HK's chiplet-aware scheduler. A two-phase block-ID remap done at kernel entry. Two knobs: `W`, `C`. |
| **Round-robin scheduling** | The default AMD policy: block ID `xy` runs on XCD `xy % nXCD`. Stripes work across all XCDs. |
| **XCD grouping (Phase 1)** | Inverts the hardware round-robin so C consecutive logical blocks run on the same XCD. Knob: **C** (chunk size). |
| **Hierarchical windowed traversal (Phase 2)** | Within an XCD's chunk, visits output tiles in vertical windows of height W so each window reuses both a row tile of A and a column tile of B. Knob: **W** (window height). |
| **L2-greedy trap** | Pushing W and C to maximize L2 hit rate (e.g. 79%) starves the LLC (24%) and ends up *slower* than the naive baseline. |
| **Thread block cluster** | NVIDIA's primitive (Hopper+) for co-locating blocks on adjacent SMs. AMD has no equivalent — Algorithm 1 is the workaround. |

---
Sources: [HipKittens paper (arXiv 2511.08083)](https://arxiv.org/abs/2511.08083) · [HK blog](https://hazyresearch.stanford.edu/blog/2025-11-09-hk) · [HazyResearch/HipKittens](https://github.com/HazyResearch/HipKittens)
