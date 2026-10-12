---
name: threadblocks-and-registers
description: AMD threadblock=workgroup terminology, and the full register-budget derivation of why 8-wave ping-pong holds a 256x256 tile.
source: HipKittens paper (arXiv 2511.08083), HazyResearch blog, and HipKittens GitHub repo
---

# HipKittens: Threadblocks, Workgroups, and the 256×256 Tile

Notes from conversation on AMD GPU register budgeting and the HipKittens 8-wave ping-pong schedule.

---

## Q1: What is referred to as a "threadblock" in AMD GPU?

In AMD GPU terminology, a **threadblock = workgroup** — the set of waves that co-reside on one CU, share LDS, and synchronize with `s_barrier`.

HIP adopted CUDA-like launch syntax (`<<<gridDim, blockDim>>>`), so the HIP "block" directly maps to the AMD workgroup. The HipKittens docs use "threadblock" throughout as the familiar CUDA term.

| NVIDIA          | HIP (AMD API)         | AMD Hardware docs  |
|-----------------|-----------------------|--------------------|
| Thread block / CTA | Block (`blockDim`) | **Workgroup**      |
| Warp (32 threads)  | —                  | **Wavefront / Wave** (64 threads on CDNA) |
| SM                 | —                  | **CU (Compute Unit)** |

From the file:
- `s_barrier` — *synchronize all waves in a thread block* (AMD instruction, AMD hardware)
- "Picture one CU running an **8-wave threadblock**" — one CU executes the threadblock; 8 waves share that CU's LDS

---

## Q2: Elaborate — "With 256 regs/wave, a single threadblock can hold a 256×256 output tile"

### Step 0: Hardware constraint — 256 regs/wave in 8-wave ping-pong

A SIMD has **512 registers total** (256 VGPR + 256 AGPR). With **2 waves per SIMD**, each wave gets exactly **256 registers** — fixed at compile time, non-negotiable.

The 8-wave ping-pong threadblock:

```
4 SIMDs × 2 waves/SIMD = 8 waves per threadblock
8 waves × 64 threads/wave = 512 threads per threadblock
```

### Step 1: The output tile is shared across all 512 threads

A **256×256 output tile** means the threadblock is responsible for a 256-row × 256-column tile of the output matrix C (in GEMM: `C = A × B`).

Total scalar output elements:

```
256 × 256 = 65,536 BF16 elements
```

These 65,536 accumulators must live in registers for the entire K-loop — MFMA accumulates into them in-place across every K-tile iteration. Spilling to LDS between every MFMA would destroy latency hiding.

### Step 2: Distribute across threads

Spread evenly across all 512 threads:

```
65,536 accumulators ÷ 512 threads = 128 accumulators per thread
128 × 2 bytes (BF16) = 256 bytes per thread
256 bytes ÷ 4 bytes/register = 64 registers per thread (per wave slot)
```

### Step 3: Check fit — the margin matters

Each wave has **256 registers**. The output tile consumes **64** of them:

```
256 − 64 = 192 registers remaining per wave
```

Those 192 registers fund:
- Input tile A registers (current K-slice of A, LDS → VGPR before each MFMA)
- Input tile B registers (same for B)
- `ds_read` temporaries (staging while data moves LDS → VGPR)
- Ping-pong phase flags and loop counters

192 registers is a comfortable margin. This is why 8-wave ping-pong works: the 2-waves-per-SIMD split gives each wave exactly 256 registers — enough to hold a large accumulator tile AND have room for the feed machinery.

### Why this matters — contrast with 4-waves-per-SIMD

With **4 waves per SIMD**, each wave gets only **128 registers**. The output tile would need to shrink (e.g., 128×256) losing arithmetic intensity, or spill to LDS on every accumulate.

The 8-wave (2-per-SIMD) configuration is the **unique sweet spot** where:
1. There is still a partner wave on each SIMD to overlap compute/load (can't go 1-per-SIMD without losing ping-pong)
2. Each wave still has 256 registers, enough for a 256×256 tile

Going to 1-wave-per-SIMD (4-wave interleave) gives 512 registers per wave, but then a single wave must interleave its own MFMAs and loads at the instruction level — harder to write and less tolerant of imbalance.

### Summary

```
8-wave threadblock on one CU:
  4 SIMDs × 2 waves = 8 waves × 64 threads = 512 threads

Register budget per wave (2 waves/SIMD):
  512 total ÷ 2 = 256 regs/wave

Output tile distribution:
  256×256 tile = 65,536 BF16 values
  ÷ 512 threads = 128 values/thread = 64 registers/thread

Registers remaining per wave after output tile:
  256 − 64 = 192 registers  →  input tiles + temporaries fit
```

---

## Q3: How do we know 256 regs/wave → 256×256 tile?

The passage mixes units in a way that obscures the derivation. Here is the precise version.

### What "256 registers per wave" actually means

A register on AMD is a **64-lane vector slot** — it stores one 32-bit value per thread simultaneously. When the scheduler gives a wave 256 registers, it means 256 such slots, each serving all 64 threads at once:

```
Register slot N:
  thread 0  → 32 bits
  thread 1  → 32 bits
  ...
  thread 63 → 32 bits
```

So "256 registers per wave" means **each thread in that wave independently has 256 × 32-bit slots** — the same 256 slots serve all 64 threads in parallel.

### The actual derivation (backwards from the tile)

**Given:** 8-wave threadblock, 2 waves/SIMD → 256 regs/wave, 64 threads/wave → 512 threads total.

Spend **64 of the 256 registers** on the output accumulator tile (25% of budget, leaving 192 for inputs/temporaries):

```
64 register slots × 2 BF16/slot  (packing 2×16-bit into one 32-bit slot)
= 128 BF16 values per thread

128 BF16/thread × 64 threads/wave = 8,192 BF16 per wave

8,192 BF16/wave × 8 waves = 65,536 BF16 total

√65,536 = 256  →  a 256×256 square tile
```

### Why the passage's comparison looks odd

The passage says:

> *128 accumulators per thread = 256 bytes. That's well under the 256 register budget per wave.*

It compares **256 bytes** (storage, per-thread) against **256 registers** (slot count, per-wave) — different units. What it is really saying:

| Quantity | Value |
|---|---|
| Output tile registers needed | 64 slots/wave |
| Register budget | 256 slots/wave |
| Fraction used | 25% |
| Remaining for inputs/temps | 192 slots/wave |

64 is "well under" 256. The 256-byte figure is a size sanity check (128 BF16 × 2 bytes = 256 bytes per thread), not a direct register comparison.

### Working forwards from 256 regs/wave

Given 256 regs/wave, what is the largest square tile that leaves ~75% of the register budget free?

```
25% of 256 = 64 registers/wave for output

64 regs × 2 BF16/reg = 128 BF16/thread
128 × 64 threads     = 8,192 BF16/wave
8,192 × 8 waves      = 65,536
√65,536              = 256
```

### Key point

The 256×256 tile is not derivable from 256 regs/wave alone — you also need the threadblock shape (8 waves × 64 threads = 512 threads) and the budget split decision (~25% for output). The 256×256 is the largest square tile that fits that budget. HipKittens uses it as the reference point because it is the configuration that maximises arithmetic intensity without register pressure.


---
See also: [cu-simd-wave-vector-alu](cu-simd-wave-vector-alu.md) · [schedules](../../papers/hipkittens/01-paper/schedules.md) · [glossary](../../papers/hipkittens/glossary.md)
