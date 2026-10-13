# HIP (C++) on AMD CDNA

Hand-written HIP kernels for gfx942 (MI300) and gfx950 (MI350). This folder holds the HIP-specific *how*. The general *why* lives in [`../../kernel-optimization/`](../../kernel-optimization/README.md). HipKittens design is covered in [`../../papers/hipkittens/`](../../papers/hipkittens/README.md).

Reach for raw HIP only when CK / ck_tile, rocWMMA, HipKittens, FlyDSL, or Triton cannot express the fusion, or when you must own the exact ISA.

```
 intrinsics-map ──► kernel-patterns ──► signal-recipes ──► mxfp8-grouped-gemm
   (what calls)       (how to wire)      (what to change)    (worked design)
                         └── hipkittens-idioms (same ideas via a tile library)
```

## Notes

- [intrinsics-map](intrinsics-map.md) — Builtin map: MFMA and scaled MFMA, buffer descriptors, direct-to-LDS sizes, transposed reads, DPP/permute, sched masks, waitcnt encoding.
- [kernel-patterns](kernel-patterns.md) — Wave64 idioms, DPP reductions, uniform values, pinned MFMA loops, phase-templated pipelines, persistent and split-K grids, extension traps.
- [signal-recipes](signal-recipes.md) — Profiler signal to HIP code change table with gotchas, an apply order, and the rule for when porting to HIP pays.
- [mxfp8-grouped-gemm](mxfp8-grouped-gemm.md) — MXFP8 MoE GEMM design: block-to-expert dispatch, E8M0 scale preshuffle and K-tail pad, dgrad reuse, why wgrad differs.
- [hipkittens-idioms](hipkittens-idioms.md) — HipKittens code idioms checked against the repo: tile types, allocator, swizzle prefill, scaled MMA, pinned ranges.
