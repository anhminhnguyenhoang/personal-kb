---
name: hipcc-agpr-pinning
description: HIPCC toolchain, the AGPR-as-MFMA-input refusal that costs ~19%, and HK's pinned_register_tile workaround.
source: HipKittens paper (arXiv 2511.08083), HazyResearch blog, and HipKittens GitHub repo
---

# Topic A · HIPCC

## A.1 What it actually is

**HIPCC** = **HIP** **C**ompiler **C**ompiler driver. It's AMD's compiler driver for HIP — AMD's CUDA-equivalent C++ runtime/API. Under the hood it's a thin wrapper around **Clang/LLVM** with an AMDGPU backend:

```
Your .cu/.hip file
        │
        ▼
┌───────────────────────────┐
│         HIPCC             │  ← driver script
│  (parses flags, dispatches)│
└───────┬───────────────────┘
        │
        ▼
┌───────────────────────────┐
│    Clang (HIP frontend)   │  ← parses C++/HIP, emits LLVM IR
└───────┬───────────────────┘
        │
        ▼
┌───────────────────────────┐
│   LLVM AMDGPU backend     │  ← register allocation, scheduling
└───────┬───────────────────┘
        │
        ▼
┌───────────────────────────┐
│  GCN/CDNA assembly + ISA  │  ← v_mfma_*, ds_read_*, buffer_load_*
└───────────────────────────┘
```

Think of it as AMD's `nvcc` analogue. Same place in the build, similar surface, *very* different backend behavior.

## A.2 The specific complaint the paper has

CDNA has two classes of vector registers:

| Class | Count per wave (2 waves/SIMD) | Purpose |
|---|---|---|
| **VGPR** (vector GPR) | 256 | General per-thread values, MFMA *inputs* |
| **AGPR** (accumulator GPR) | 256 | Designed as MFMA *output* accumulators |

The hardware actually allows AGPRs as MFMA inputs too — but **HIPCC's register allocator refuses to emit code that does that**. If your tile happens to land in AGPRs and then needs to feed an MFMA, the compiler inserts an extra instruction:

```
v_accvgpr_read_b32  v0, a0      ← move AGPR → VGPR (one per register)
v_accvgpr_read_b32  v1, a1
...
v_mfma_f32_16x16x32_bf16  ..., v0, v1, ...
```

That copy is pure overhead. On attention backward — which is register-pressured and mixes MFMA shapes — these inserted copies cost **~19% of throughput**.

## A.3 How HipKittens dodges it

HK ships a primitive called `pinned_register_tile<dtype, rows, cols, start_vgpr, start_agpr>`. It lets the developer say "this tile lives at *exactly* these register numbers, including AGPRs as MFMA inputs." Under the hood that means inline assembly with explicit register clobbers — bypassing the compiler's allocator entirely.

```
// conceptually:
pinned_register_tile<bf16, 16, 16, /*vgpr=*/8, /*agpr=*/0> tile_K;

// HK emits inline asm that names a0..aN as MFMA input operands,
// which HIPCC's high-level path won't generate on its own.
```

The paper doesn't print the exact syntax (it defers to an appendix), but conceptually that's the move: opt out of HIPCC's register management for the hot tiles.

## A.4 The bigger compiler critique

The paper makes this part of a broader point about AMD compilers: HIPCC for C++, plus Triton's AMD backend (mentioned separately as struggling with register lifetime tracking and failing to lower vector loads), together leave a real gap between "what the silicon can do" and "what the compiler will emit." HipKittens fills that gap by being a thin C++ DSL whose hot paths are essentially hand-written assembly wrapped in tile-shaped abstractions.

---


---
Related: [chiplet-scheduling](chiplet-scheduling.md) · [overview-thesis](overview-thesis.md) · [glossary](../glossary.md)
