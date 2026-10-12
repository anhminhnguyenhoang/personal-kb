---
name: schedules
description: "8-wave ping-pong and 4-wave interleave, built up from CDNA hardware: why wave-specialization dies on AMD, layout/barrier mechanics, decision rubric, glossary."
source: HipKittens paper (arXiv 2511.08083), HazyResearch blog, and HipKittens GitHub repo
---

# 8-Wave Ping-Pong & 4-Wave Interleave: A Comprehensive Walkthrough

I'm going to build this up from hardware first, because every design choice in these two schedules is a direct response to something specific about AMD CDNA silicon. If you understand the hardware, the schedules become obvious. If you don't, they look arbitrary.

---

## Part 1 · The Hardware You Need to Picture

### 1.1 The CU and its 4 SIMDs

A CDNA **CU** (Compute Unit) is the basic compute building block. MI355X has 256 CUs (32 per XCD × 8 XCDs).

Inside one CU:

```
┌─────────────────────────────────────────────────────────────┐
│                       Compute Unit (CU)                     │
│                                                             │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│   │  SIMD 0  │  │  SIMD 1  │  │  SIMD 2  │  │  SIMD 3  │    │
│   │ 512 regs │  │ 512 regs │  │ 512 regs │  │ 512 regs │    │
│   │ 1 MFMA pipe│ 1 MFMA pipe│ 1 MFMA pipe│ 1 MFMA pipe│    │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│                                                             │
│   ┌──────────────────────────────────────────────────┐      │
│   │            LDS (shared memory) — ~64 KB          │      │
│   │            64 banks (for ds_read_b128)           │      │
│   └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

A **SIMD** is a 16-wide vector ALU with its own matrix-multiply (MFMA) pipeline and its own 512-entry register file.

### 1.2 A "wave" is 64 threads in lockstep

A **wave** = 64 threads (on AMD; NVIDIA's equivalent is a 32-thread *warp*). The 64 threads execute the same instruction at the same time, on a 16-wide SIMD over 4 clock cycles.

### 1.3 Static register partition — the most important fact

A SIMD has 512 vector registers, split into **256 VGPRs** + **256 AGPRs**. If you put N waves on one SIMD, those 512 registers are **statically divided** at compile time:

| Waves per SIMD | Regs per wave | Notes |
|---|---|---|
| 1 wave | 512 | Maximum room per wave |
| **2 waves** | **256** | **Both schedules below assume this or 1** |
| 4 waves | 128 | Too few for big tiles |
| 8 waves | 64 | Useless for MFMA |

**This is fixed for the lifetime of the wave.** It is not dynamic, not transferable, not steal-able. A wave that finishes early *still owns* its registers.

Internalize this: **a wave is a permanent register reservation, not a thread of execution that can release resources.**

### 1.4 The two pipelines we're trying to keep busy

Every CDNA kernel is, at heart, a race to keep these two pipelines fed simultaneously:

```
   ┌──────────────────┐         ┌────────────────────────┐
   │  MFMA pipeline   │         │  Memory pipeline       │
   │  (matrix multiply)│         │  (buffer_load + ds_*) │
   │                  │         │                        │
   │  ~32 cycles per  │         │  ~hundreds of cycles   │
   │  MFMA instr      │         │  per HBM load          │
   └──────────────────┘         └────────────────────────┘
```

If either sits idle, you leave performance on the table. The whole game is: *how do I issue instructions so both are simultaneously busy?*

### 1.5 The instructions you'll see referenced

- **`v_mfma_*`** — the matrix-multiply instruction. Inputs come from VGPRs, output accumulates into AGPRs. Comes in shapes like `16×16×32` (M×N×K) and `32×32×16`.
- **`buffer_load_*`** — read from HBM (global memory). Can target *directly into LDS* (skipping registers) — this is AMD's nearest equivalent to NVIDIA's TMA.
- **`ds_read_*`** / **`ds_write_*`** — read/write LDS (shared memory). `ds_read_b128` reads 128 bits per lane.
- **`s_barrier`** — synchronize all waves in a thread block.
- **`s_waitcnt vmcnt(N)`** — wait until at most N outstanding global loads remain. `lgkmcnt(N)` waits on LDS ops similarly. This is how AMD expresses "wait for memory".

---

## Part 2 · The Problem Scheduling Solves

Naive pseudocode for a GEMM tile:

```
for k in K_tiles:
    load A_tile[k] from HBM
    load B_tile[k] from HBM
    wait for loads
    accumulate += A_tile @ B_tile     # MFMA
```

Timeline:

```
time →
Load  ████████████████░░░░░░░░░░░░░░░░████████████████░░░░░░░░░░░░░░░░
MFMA  ░░░░░░░░░░░░░░░░████████░░░░░░░░░░░░░░░░░░░░░░░░████████░░░░░░░░
                      ↑                                ↑
                      both pipes idle waiting          same again
```

Each pipeline does its work then waits. **We need to overlap them.** The two schedules below are two distinct ways to do that overlap on CDNA hardware.

---

## Part 3 · How NVIDIA Solves It (Wave Specialization) — and Why It Dies on AMD

NVIDIA's **wave specialization** (a.k.a. producer/consumer) divides the warps in a threadblock into two roles:

```
NVIDIA Hopper:  Producer warps          Consumer warps
                ────────────────         ────────────────
                Issue TMA loads          Issue WGMMA from
                into shared memory       shared memory
                (never touch regs)       (heavy register use)

                ──── setmaxnreg ────►    Steal producer's registers
                                          to widen output tile
```

Three things make this work *on Hopper*:

1. **TMA** loads HBM → SMEM without consuming the issuing warp's registers.
2. **WGMMA** reads operands straight from SMEM, so producers can hand off through SMEM with no register intermediary.
3. **`setmaxnreg`** lets producer warps shrink their register count after they're done, donating registers to consumers — *dynamic reallocation*.

**AMD has none of these.** Specifically:
- AMD has `buffer_load` to LDS (good — analogous to TMA-to-SMEM).
- But MFMA inputs **must come from VGPRs**, not LDS. So even if you "produce" into LDS, the consumer still has to issue `ds_read` into its own registers before computing.
- **Registers don't reallocate.** A producer wave that finishes early still owns 256 VGPRs forever.

If you try wave specialization on AMD with, say, 4 producer + 4 consumer waves:

```
8 waves × 64 regs/wave = 512 regs
Producers (4 waves):  permanently hold 4 × 64 = 256 regs they barely use
Consumers (4 waves):  only 256 regs left → tile too small
Measured result:  ~80% of peak BF16 GEMM, with no path to recover
```

That's the dead end. Now we can understand why the two HK schedules look the way they do — they're both designed so **every resident wave actually computes**.

---

## Part 4 · 8-Wave Ping-Pong (the default schedule)

### 4.1 The layout

Picture one CU running an 8-wave threadblock:

```
                    ┌─────── CU ───────┐
                    │                  │
   SIMD 0:    Wave 0 ────────  Wave 4       ← one A-wave, one B-wave
   SIMD 1:    Wave 1 ────────  Wave 5
   SIMD 2:    Wave 2 ────────  Wave 6
   SIMD 3:    Wave 3 ────────  Wave 7
                    │                  │
                    └──────────────────┘

   Group A = {W0, W1, W2, W3}      ← one A-wave on each SIMD
   Group B = {W4, W5, W6, W7}      ← one B-wave on each SIMD

   Each SIMD: 2 waves resident → 256 regs per wave (the split)
```

Two waves per SIMD, paired. Both waves are real computing waves — *neither is a permanent producer*.

### 4.2 The dance

The two waves on a SIMD swap roles in lockstep. Drawing one SIMD over time:

```
                     Phase 1                Phase 2                Phase 3
time →               ────────                ────────               ────────

Wave A (group A):  ████ MFMA on T0 ████   ░░ load T2 → LDS[0] ░░  ████ MFMA on T2 ████
Wave B (group B):  ░░ load T1 → LDS[1] ░░ ████ MFMA on T1 ████   ░░ load T3 → LDS[0] ░░
                              │                       │                       │
                              ▼                       ▼                       ▼
                         s_barrier               s_barrier               s_barrier
                        (swap roles)            (swap roles)            (swap roles)

LDS state:
  buffer[0]: T0 ─────► T0 still here ─────► overwritten with T2
  buffer[1]: empty ──► T1 loaded ────────► T1 still here ─────► overwritten with T3
```

Read the rows: at any given moment, **one wave is computing while its partner is loading the *next* tile**. After each phase, they swap. This is the "ping-pong".

Three things to notice:

1. **Double-buffered LDS.** Two physical tile slots (buffer[0] and buffer[1]) so the compute-wave's reads don't race with the load-wave's writes.
2. **The barrier is the swap signal.** A conditional `s_barrier` at the phase boundary ensures the loader has finished writing before the computer reads it.
3. **Every wave computes** — not just half. No register is wasted on a perpetual producer. That's the structural difference from wave specialization.

### 4.3 Why 8 waves specifically

It's a forced count:

```
4 SIMDs  ×  2 roles needed (compute + load)  ×  1 wave per role per SIMD  =  8 waves
```

- Fewer than 2 waves per SIMD → no partner to swap roles with.
- More than 2 → register budget per wave drops below what MFMA needs.

So "8-wave" isn't a tuning knob; it's the unique solution to "every SIMD always has both a computer and a loader resident, and every wave has enough registers to do real work."

### 4.4 What it buys you

With 256 regs/wave, a single threadblock can hold a **256×256 output tile** (HK's measurement). That's:

- 256 × 256 = 65,536 BF16 accumulators across the threadblock
- ÷ 512 threads = 128 accumulators per thread = 256 bytes
- That's well under the 256 register budget per wave, leaving room for input tile registers + temporaries.

Result: **peak (or very near peak) on BF16 GEMM, FP8 GEMM, and attention forward**, expressed in ~500 LoC for attention forward.

### 4.5 What it can't do

The ping-pong assumes a relatively **balanced** compute/memory ratio. If your kernel does 10× more compute per memory load than expected, the load wave finishes early and idles waiting for the barrier. Vice versa for memory-heavy kernels. The schedule has no way to mix instructions *within* a wave — each wave is either-or per phase.

That's the failure mode 4-wave interleave addresses.

---

## Part 5 · 4-Wave Interleave (the harder hammer)

### 5.1 The layout

```
                    ┌─────── CU ───────┐
                    │                  │
   SIMD 0:        Wave 0   (only)         ← 1 wave per SIMD → 512 regs/wave
   SIMD 1:        Wave 1   (only)
   SIMD 2:        Wave 2   (only)
   SIMD 3:        Wave 3   (only)
                    │                  │
                    └──────────────────┘
```

**One wave per SIMD.** That wave has the entire 512-register file to itself. No partner, no role swap.

### 5.2 The dance — but now inside a single wave

A single wave handles both compute and memory. The trick is to **interleave individual instructions** so the MFMA pipeline and the memory pipeline are both busy. AMD's `s_waitcnt` lets you order this precisely.

Sketch of one iteration of the inner loop:

```
Wave (one SIMD), instructions issued in order:
─────────────────────────────────────────────────────────────────
  buffer_load   chunk_i+1, region A   ─┐
  v_mfma_16x16  chunk_i,    accum 0    │   ← MFMA #1 starts (~32 cycles)
  buffer_load   chunk_i+1, region B    │
  v_mfma_16x16  chunk_i,    accum 1    │   ← MFMA #2 starts
  ds_read_b128  chunk_i+2  (from LDS)  │
  v_mfma_16x16  chunk_i,    accum 2    │   ← MFMA #3
  s_waitcnt     vmcnt(0)              ─┘   ← now wait for the HBM loads
  ds_write      chunk_i+1 → LDS
  v_mfma_16x16  chunk_i,    accum 3
  s_waitcnt     lgkmcnt(0)                ← wait for LDS reads
  ... (continues for ~20 instructions per iteration)
```

Pipeline view of what's actually running concurrently:

```
time →
MFMA pipe:   ████ MFMA1 ████ MFMA2 ████ MFMA3 ████ MFMA4 ████ MFMA5 ████
Memory pipe: ████ load_HBM ████ load_HBM ████ ds_read ████ ds_write ████
                  ↑both pipes loaded simultaneously throughout iteration↑
```

The 16×16×32 MFMA shape is HK's default for this schedule because it's the **smallest** matrix instruction — smaller MFMAs give the scheduler more places to slot a load between, increasing overlap granularity.

### 5.3 Why one wave per SIMD?

With 1 wave per SIMD and 512 registers all to itself, you can:

- Hold accumulators for **multiple MFMA shapes simultaneously** (16×16 for the GEMM, 32×32 for an epilogue or different operand layout).
- Hold **partial softmax state** (running max, running sum, exp values) in registers alongside MFMA accumulators.
- Keep enough input-tile registers in flight to issue several MFMAs back-to-back without waiting.

This is exactly the regime GQA backward needs: it uses **mixed MFMA shapes** (16×16×32 *and* 32×32×16 in the same kernel), needs **row- and column-major loads from the same LDS tile**, and carries **online-softmax state** through the iteration. None of that fits in 256 registers per wave.

### 5.4 Why the code explodes (331 → 989 LoC)

8-wave ping-pong's mental model is "load phase, compute phase, swap, repeat." Three abstractions.

4-wave interleave's mental model is "every ~20 instructions, decide which load and which MFMA goes next so both pipes stay busy." There's no clean phase abstraction — you're hand-rolling the instruction stream.

For a multi-shape, multi-layout kernel like GQA backward, you end up writing out:
- One stretched-out interleaved sequence per MFMA shape
- Explicit `s_waitcnt` between the right instructions
- Hand-placed `ds_read` for both row and column layouts of the same data
- Explicit register-pinning hints (HK's `pinned_register_tile`) to dodge HIPCC's AGPR limitation

That's where 989 LoC comes from. Almost all of it is mechanical instruction sequencing, not algorithm.

### 5.5 What it buys you

| Kernel | 8-wave | 4-wave | Win |
|---|---|---|---|
| BF16 GEMM | 1281 TFLOPS | (similar) | — |
| FP8 GEMM | 3222 | 3327 | small |
| **GQA non-causal backward** | 894 | **1091** | **+22% over 8-wave; 2.8–4.0× over AITER ASM** |

Where compute/memory is balanced, 8-wave's simplicity wins or ties. Where the kernel is register-pressured or imbalanced, 4-wave's instruction-level control wins big.

---

## Part 6 · Side-by-Side Visualization

```
              ┌─────────────────────────┬─────────────────────────┐
              │      8-WAVE PING-PONG   │     4-WAVE INTERLEAVE   │
              ├─────────────────────────┼─────────────────────────┤
 Waves/CU     │           8             │           4             │
 Waves/SIMD   │           2             │           1             │
 Regs/wave    │          256            │          512            │
 LDS buffers  │      double (2)         │     single + streaming  │
 Concurrency  │   inter-wave (A vs B)   │   intra-wave (instrs)   │
 Sync         │      s_barrier          │  s_waitcnt vmcnt/lgkm   │
 Tile size    │   large (256×256)       │  small (per-MFMA)       │
 Compute/load │ different waves do them │ same wave does both     │
 LoC (MHA-bwd)│         331             │        989              │
 Sweet spot   │ balanced GEMM, attn fwd │ mixed-shape, reg-heavy  │
              └─────────────────────────┴─────────────────────────┘
```

Or in one mental image:

```
   8-WAVE PING-PONG               4-WAVE INTERLEAVE
   ────────────────               ─────────────────

   ┌───┐    ┌───┐                 ┌─────────────┐
   │ A │◄──►│ B │  swap            │  one wave   │
   │MFM│    │LD │  every           │  with both  │
   │A  │    │   │  phase           │   pipes     │
   └───┘    └───┘                 │   loaded    │
   "two waves taking turns"        │  in parallel│
                                   └─────────────┘
                                   "one wave juggling everything"
```

---

## Part 7 · A Decision Rubric

If you ever have to pick between the two for a new kernel on CDNA3/4:

```
Start with 8-wave ping-pong.

If your kernel:
   ├─ uses only ONE MFMA shape           → 8-wave likely fine
   ├─ has roughly balanced compute/load  → 8-wave likely fine
   ├─ fits comfortably in 256 regs/wave  → 8-wave likely fine
   └─ achieves close to peak             → STOP, ship it

If 8-wave caps below peak AND your kernel:
   ├─ mixes MFMA shapes (16×16×32 + 32×32×16)        ─┐
   ├─ needs row- AND column-major reads of same data  ├─► try 4-wave
   ├─ carries large per-iteration state (softmax)     │
   ├─ is heavily compute-imbalanced or memory-imbalanced ┘
   └─ accept ~3× more LoC for ~20–30% perf
```

Anecdotally from the paper: GEMM and attention forward are 8-wave territory; GQA backward, FlashAttention-style backward passes, and anything with online normalization are 4-wave territory.

---

## Part 8 · Glossary (in one place)

| Term | Meaning |
|---|---|
| **Wave** | 64 threads executing in lockstep on one SIMD. AMD's equivalent of a NVIDIA warp (32 threads). |
| **CU** (Compute Unit) | A processor containing 4 SIMDs, a shared LDS, schedulers. MI355X has 256 CUs. |
| **SIMD** | A 16-wide vector ALU with its own register file (512 regs) and its own MFMA pipeline. |
| **VGPR** | Vector General-Purpose Register. Holds per-thread values. 256 per wave (with 2 waves/SIMD). |
| **AGPR** | Accumulator GPR. Used as MFMA *outputs*, also 256 per wave. HIPCC won't let them feed MFMA *inputs*. |
| **MFMA** | Matrix Fused Multiply-Add — the matrix-multiply instruction. Shapes like 16×16×32 mean M×N×K dims. |
| **LDS** | Local Data Share — AMD's name for shared memory (per-CU, ~64 KB). 64 banks for 128-bit reads. |
| **buffer_load** | HBM → LDS load instruction. AMD's nearest-to-TMA, but per-thread-addressed. |
| **ds_read / ds_write** | LDS → registers / registers → LDS. |
| **s_barrier** | Threadblock-wide synchronization. Used for ping-pong phase boundaries. |
| **s_waitcnt vmcnt/lgkmcnt** | Wait for outstanding global-memory / LDS-memory operations. Used for interleaved scheduling. |
| **Wave specialization** | NVIDIA pattern: dedicated producer waves and consumer waves. Fails on AMD because of static register split. |
| **Static register partition** | The 512 regs of a SIMD are divided evenly across all resident waves at compile time; not transferable. *This single fact is the reason both schedules look the way they do.* |
| **Double buffering** | Keep two physical tile buffers in LDS so loader and computer don't race. Used by ping-pong. |
| **8-wave ping-pong** | 2 waves per SIMD; they swap roles each phase. Both waves always compute. Good for balanced kernels. |
| **4-wave interleave** | 1 wave per SIMD; the wave interleaves load and MFMA instructions itself. Good for register-heavy, mixed-shape kernels. |

---

## Part 9 · The One-Sentence Takeaway

> **8-wave ping-pong gets parallelism *across* waves; 4-wave interleave gets parallelism *within* a wave. Both exist because AMD's static register partition forbids the NVIDIA trick of making some waves donate their registers.**

Once you've held that sentence in your head, the rest of the HipKittens paper basically writes itself.

Sources:
- [HipKittens paper (arXiv 2511.08083, HTML)](https://arxiv.org/html/2511.08083v1)
- [HipKittens blog post](https://hazyresearch.stanford.edu/blog/2025-11-09-hk)



---
Prereqs: [cu-simd-wave-vector-alu](../../../hardware/amd-cdna/cu-simd-wave-vector-alu.md) · [threadblocks-and-registers](../../../hardware/amd-cdna/threadblocks-and-registers.md). Related: [overview-thesis](overview-thesis.md)
