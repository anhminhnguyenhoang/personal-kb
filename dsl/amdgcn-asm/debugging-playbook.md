---
name: debugging-playbook
description: How to find bugs in a hand-written gfx942/gfx950 asm kernel — symptom-to-cause table, sentinel/all-ones/identity tests, a shape pyramid that exposes loop and grid bugs, round-trip checks on the binary, and in-kernel timers.
source: https://llvm.org/docs/AMDGPUUsage.html ; https://llvm.org/docs/CommandGuide/llvm-objdump.html ; AMD CDNA3 and CDNA4 ISA reference guides (s_memtime, s_memrealtime, MFMA layouts) ; https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/
---

# Debugging hand-written asm kernels

A broken asm kernel rarely crashes. It returns wrong numbers, on time. Judge every change on
correctness before you time it.

## 1. Symptom → likely cause

| Symptom | Check first |
|---|---|
| Output untouched (sentinel survives) | Store path skipped; early-exit branch; wrong EXEC; grid/tile index math |
| All zeros | Zero AGPRs allocated (`next_free_vgpr == accum_offset`); loads all OOB (bad `num_records` / base) |
| Garbage, cosine ≈ 0 | Wrong MFMA lane layout; kernargs read from wrong offsets/SGPRs; WG id read from a preloaded kernarg SGPR |
| Close but wrong (cosine 0.9–0.99) | Partial register clobber; missing barrier; a hazard (missing wait states) |
| Wrong scale, right shape (e.g. ×4, ×0.5) | fp8 FNUZ vs OCP; scale register clobbered; scale applied twice |
| Error grows with K or sequence length | Clobber that hits from trip 2 on; `vmcnt`/`lgkmcnt` miscount after a reorder |
| About half the expected value | Loop runs K/2 times; one of two accumulator sets never added |
| Wrong only in some tiles | Grid/tile index; OOB edge handling; persistent loop state not reset between tiles |
| Changes run to run | Async load landing in a live register; LDS race between waves; atomics order (expected ULP-level only) |
| NaN in a few lanes | Load destination overlaps an address register; division by a zero max (init reductions with a tiny non-zero) |
| Hang | Unequal `s_barrier` count across waves; spin barrier with workgroup-scope poll or non-resident workgroups |
| Fails to launch | Missing/empty `.args` metadata; `.co` not a shared object; LDS or VGPR request over the limit |
| Much faster than before | Instructions dropped. Check correctness and instruction count before celebrating |

## 2. Fast tests (minutes)

1. **Sentinel fill.** Fill the output with a value the kernel never writes (e.g. -1.0 or NaN
   bits). Any survivor marks a tile that never stored.
2. **All-ones inputs.** For GEMM every output must equal K exactly. Zero, K/2, or noise
   means a structural bug, not a numeric one.
3. **Identity input.** A = I gives C = B. A transposed result shows a lane-layout mistake
   at once (square random data can hide it).
4. **Per-tile error map.** Compute error per output tile, not one global number. The bad
   tiles point at the workgroup, wave, or edge case.
5. **Per-wave split.** If waves 1..3 are wrong and wave 0 is right, suspect LDS hand-off
   (barriers, per-wave bases).

## 3. Shape pyramid

Run in order and stop at the first failure:

| Step | Shape | Exposes |
|---|---|---|
| 1 | Copy kernel: load tile, store tile | kernargs, descriptors, store path |
| 2 | One tile, one K step | MFMA layout, epilogue |
| 3 | One tile, 2 K steps | loop-carried state, ping-pong toggle, clobbers from trip 2 |
| 4 | One tile, many K steps | waitcnt counts, prefetch order |
| 5 | Many tiles | grid math, OOB edges, persistent-loop resets |
| 6 | Production shapes incl. ragged edges | everything at once |

Use random data for timing. All-zero inputs can run faster than real data and mislead
benchmarks.

## 4. Binary hygiene

- **Round trip first.** Disassemble the working binary, rebuild it unchanged, and confirm
  the `.text` is bit-identical before your first edit. A disassembler or assembler gap (a
  DPP modifier dropped, a branch target left numeric, a `.set` swallowed) then shows up
  before it can masquerade as a speedup.
- After each edit, diff the disassembly of old vs new. Only your intended lines should change.
  Compare instruction counts.
- One change per validated step. Revert on any correctness drop.
- Read the metadata (`llvm-readelf --notes`): VGPR/AGPR/SGPR counts, LDS size, scratch size.
  A VGPR count that crossed an occupancy step explains a "cleaner but slower" kernel.

## 5. Tolerances

- Know the floor before you call a result a bug: run the reference twice, run the kernel
  twice. Kernels with float atomics or a different reduction order are not bit-stable.
- Low-precision accumulation (bf16 atomics, fp8 inputs) has its own floor against an fp32
  reference. Compare against a reference with the same input precision first.
- MFMA uses RNE with denormal flush. Match that in the reference when you need bit equality.

## 6. Timing inside the kernel

```asm
s_memtime     s[40:41]          ; 64-bit shader-clock counter (SMEM, counts on lgkmcnt)
s_waitcnt     lgkmcnt(0)
; ... region ...
s_memtime     s[42:43]
s_waitcnt     lgkmcnt(0)
s_sub_u32     s42, s42, s40
s_subb_u32    s43, s43, s41      ; store the delta from one lane for host readback
```

`s_memrealtime` reads a constant-rate clock instead (immune to clock changes). Either one
is coarse at a few cycles but fine for loop-level phases. For per-instruction stalls use a
thread trace (rocprofv3 ATT), not timers.

Sources: [LLVM AMDGPUUsage — code object metadata, kernel descriptor](https://llvm.org/docs/AMDGPUUsage.html) ·
[llvm-objdump](https://llvm.org/docs/CommandGuide/llvm-objdump.html) ·
[AMD CDNA3 ISA — SMEM s_memtime / s_memrealtime](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) ·
[AMD CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) ·
[ROCprofiler-SDK / rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/)
Verify specifics against the primary source.

---
Related: [registers-agpr-liveness](registers-agpr-liveness.md) · [waitcnt-and-hazards](waitcnt-and-hazards.md) · [kernel-abi-and-prologue](kernel-abi-and-prologue.md) · [failure-modes](../../kernel-optimization/failure-modes/failure-modes.md)
