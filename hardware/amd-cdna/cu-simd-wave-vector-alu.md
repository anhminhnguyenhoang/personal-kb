---
name: cu-simd-wave-vector-alu
description: What a vector ALU is and why CDNA picks 16 lanes; CU/SIMD/wave/quad-cycle foundations.
source: HipKittens paper (arXiv 2511.08083), HazyResearch blog, and HipKittens GitHub repo
---

# What is a Vector ALU and Why is it 16-Wide?

Good question — the "16-wide SIMD" detail is one of those things people repeat without ever explaining the number. The answer is genuinely elegant once you see it.

---

## 1. What an ALU is (foundation)

An **ALU** (Arithmetic Logic Unit) is the piece of silicon inside a processor that actually does math — adds, multiplies, compares, ANDs, etc. A scalar CPU core has one ALU that operates on one number at a time:

```
   ┌──────────┐
   │   ALU    │   one input pair in, one result out, per cycle
   │  a + b   │
   └──────────┘
```

That's it. The whole rest of the CPU exists to feed this thing.

## 2. What "vector" means

A **vector ALU** is the same idea but parallelized: one instruction operates on many numbers at once, each in its own *lane*.

```
   ┌─────────────────────────────────────────────────────┐
   │              Vector ALU (N lanes wide)              │
   │  ┌────┐ ┌────┐ ┌────┐ ┌────┐  ...  ┌────┐          │
   │  │ALU0│ │ALU1│ │ALU2│ │ALU3│       │ALUN│           │
   │  └────┘ └────┘ └────┘ └────┘       └────┘           │
   │   a0+b0  a1+b1  a2+b2  a3+b3  ...   aN+bN           │
   └─────────────────────────────────────────────────────┘
            ↑
   one instruction "v_add a, b" → N adds happen in parallel
```

The "single instruction, multiple data" (SIMD) name comes from this: *one* fetched/decoded instruction drives *many* data-path lanes. You pay the instruction-fetch cost once and amortize it across N operations.

## 3. Why exactly 16 lanes on CDNA

Now the specific question: **AMD's SIMD has 16 lanes, but a wave is 64 threads. Where does the mismatch go?**

Answer: **time**. The SIMD chews through the wave over **4 clock cycles**:

```
   wave = 64 threads, SIMD = 16 lanes:

   cycle 0:   lanes 0–15  execute the instruction for threads 0–15
   cycle 1:   lanes 0–15  execute the instruction for threads 16–31
   cycle 2:   lanes 0–15  execute the instruction for threads 32–47
   cycle 3:   lanes 0–15  execute the instruction for threads 48–63
              ──────────
              after 4 cycles, the whole wave has executed the instruction once
```

So when someone says "a wave executes in lockstep on a 16-wide SIMD," they mean the wave's 64 threads get strip-mined over 4 cycles on the same 16-wide physical hardware. From the programmer's perspective it looks atomic — the next instruction can't start using a result until all 64 lanes have finished. From the silicon's perspective it's a 4-cycle pipelined sequence.

## 4. Why 16 and not 32 or 64? — The CU-level trick

Here's the elegant part. A CU has **4 SIMDs**, and that count is not coincidental. Each SIMD takes 4 cycles to finish one wave's instruction. The CU front-end **issues to a different SIMD each cycle, round-robin**:

```
   CU's instruction issue schedule:

   cycle 0:  issue to SIMD 0 (working on wave A)  → SIMD 0 busy cycles 0–3
   cycle 1:  issue to SIMD 1 (working on wave B)  → SIMD 1 busy cycles 1–4
   cycle 2:  issue to SIMD 2 (working on wave C)  → SIMD 2 busy cycles 2–5
   cycle 3:  issue to SIMD 3 (working on wave D)  → SIMD 3 busy cycles 3–6
   cycle 4:  issue to SIMD 0 (next instr of wave A — finished at cycle 3) ✓
   cycle 5:  issue to SIMD 1 (next instr of wave B — finished at cycle 4) ✓
   ...
```

Effective throughput: **one full wave (64 ops) per clock cycle**, at the cost of **only 16 ALU lanes per SIMD × 4 SIMDs = 64 ALUs per CU**.

This is the deal: a single 64-wide SIMD would also do 64 ops/cycle, but it would need 64 ALU lanes built and powered, plus a 64-wide register-file read port, plus a 64-wide bypass network, etc. The 4-SIMD-with-16-lanes-each design hits the same throughput with **the same total ALU count but a much simpler per-SIMD datapath**. You also get four independent execution contexts that can run completely different waves — better latency tolerance.

The numbers are picked specifically so wave_size / lane_width = num_SIMDs:

```
       64 threads/wave   ÷   16 lanes/SIMD   =   4 cycles per wave
                                                  ↕
                                                  matches
                                                  ↕
       4 SIMDs per CU    →   1 wave issued per cycle, full CU throughput
```

Break any of those numbers and the schedule misaligns. That's why "16" is the answer — it's the only width that makes the 4-SIMD round-robin run at exactly one wave per cycle for a 64-thread wave.

## 5. Why this also hides latency for free

Most ALU instructions have ~4-cycle pipeline latency (the result of `a + b` isn't ready until ~4 cycles after issue). If the SIMD were 64-wide and finished a wave in 1 cycle, the next instruction in the same wave couldn't start for 3 idle cycles.

With the 16-wide × 4-cycle design, the wave's *own* execution takes 4 cycles to finish. By the time you'd want to use the result for the next instruction in the same wave, it's already available. **The 4-cycle latency is hidden inside the 4-cycle execution.** This is called the **"quad-cycle"** design and it's why GCN/CDNA can usually issue dependent instructions back-to-back without explicit scheduling.

## 6. Contrast with NVIDIA

NVIDIA's terminology and numbers are different but the underlying logic is similar:

| | AMD CDNA | NVIDIA (Hopper) |
|---|---|---|
| Wave/warp size | 64 | 32 |
| SIMD lane count | **16** | 32 (per "sub-partition") |
| Cycles per wave/warp instr | 4 | 1 |
| Schedulers per CU/SM | 4 SIMDs | 4 sub-partitions |
| Effective throughput | 1 wave/cycle per CU | 1 warp/cycle per SM |

NVIDIA made a different tradeoff: smaller warps (32) running on 32-wide SIMDs in one cycle. Same idea of "issue is per-cycle, one warp/wave at a time," just different parameter point. RDNA (AMD's graphics architecture, since 2019) actually moved closer to NVIDIA's choice — 32-wide SIMDs and "wave32" mode — but **CDNA kept the 16-wide-4-cycle design** because for compute (especially MFMA matrix ops), the latency-hiding from quad-cycle execution is more valuable than the per-instruction latency win.

## 7. Why this matters for HipKittens

The two HK schedules we walked through are both designed around this:

- **8-wave ping-pong** puts 2 waves on each SIMD. While one wave's instruction is mid-flight on its 4-cycle journey, the SIMD scheduler is already issuing the partner wave's instruction. This is why the swap *works at all* — the SIMD's quad-cycle execution naturally interleaves the two waves' instructions.
- **4-wave interleave** puts 1 wave per SIMD. With only one wave per SIMD, you don't get the wave-interleave latency hiding for free — you have to engineer it yourself by interleaving compute and memory instructions *within* the single wave's instruction stream.

The 16-wide SIMD isn't just trivia — it's why CDNA wants ≥2 resident waves per SIMD to hide latency, and why 1-wave-per-SIMD is a harder mode to fill the pipeline in.

---

## Glossary additions

| Term | Meaning |
|---|---|
| **ALU** | Arithmetic Logic Unit. The silicon that does +, ×, etc. on one pair of operands per cycle. |
| **Vector ALU / Vector unit / SIMD unit** | A bank of N parallel ALU lanes driven by one instruction. CDNA's is **16 lanes wide**. |
| **Lane** | One column of a vector ALU; runs one thread's data path. |
| **SIMD width / lane count** | Number of physical lanes in one vector unit. **16 on CDNA.** |
| **Wave/warp size** | Number of threads that move together. **64 on AMD, 32 on NVIDIA.** |
| **Cycles per wave instruction** | `wave_size ÷ lane_count`. **4 on CDNA** (64/16). |
| **Quad-cycle design** | GCN/CDNA's specific 64-thread-wave × 16-lane-SIMD × 4-cycle-issue choice. Hides ALU pipeline latency by construction. |
| **SIMD round-robin** | The CU schedules its 4 SIMDs one per cycle, so it can issue a wave's worth of work every cycle despite each SIMD taking 4 cycles per wave. |
| **wave32 / wave64** | RDNA (graphics) can run in either mode; CDNA (compute) is wave64 only. |

---

## The one-sentence takeaway

> A **vector ALU** is N parallel ALU lanes driven by one instruction; CDNA picks **N = 16** because that's the lane width that makes a 64-thread wave finish in exactly 4 cycles — and 4 SIMDs per CU running staggered gives you full one-wave-per-cycle throughput with a quarter of the per-SIMD silicon you'd need for a single 64-wide unit, with ALU pipeline latency hidden for free as a bonus.


---
See also: [threadblocks-and-registers](threadblocks-and-registers.md) · [schedules](../../papers/hipkittens/01-paper/schedules.md) · [glossary](../../papers/hipkittens/glossary.md)
