---
name: kernel-authoring-primitives
description: "FlyDSL authoring details: host-built tiled atoms, LDS-DMA/async/transpose/atomic atoms, the example-04 G→LDS→MFMA→global operand chain, and the API-stability boundary."
source: https://github.com/ROCm/FlyDSL (docs/kernel_authoring_guide.md, docs/language/storage_and_allocator.md, docs/api_stability.md, examples/01-04, python/flydsl/expr/rocdl/*.py)
---

# FlyDSL kernel-authoring primitives

## 1. Build tiled atoms on the host

- Build `TiledMma` / `TiledCopy` in the `@jit` launcher and pass them as typed kernel args
  (example 04). Their static layout joins the cache key, and the kernel stays shape-generic.
- `known_block_size=[x,y,z]` on `@flyc.kernel` feeds `max_flat_workgroup_size`. Set it when
  launch dims are dynamic, or the backend falls back to its default work-group bound.

## 2. Less common atoms

| Kind | Atom | Notes |
|---|---|---|
| G→LDS DMA | `fx.rocdl.BufferCopyLDS{32,128}b()` | compiler inserts the `vmcnt` wait; no 8-B form (`BufferCopyLDS64b` raises) |
| async G→LDS (CDNA4) | `rocdl.cdna4.BufferLoadAsyncLDS`, `GlobalLoadAsyncLDS` | you track with `asyncmark()` / `wait_asyncmark(n)` |
| LDS transpose read (CDNA4) | `rocdl.cdna4.LDSReadTrans(gran, bits)` | `ds_read_*_tr_*`; no explicit transpose |
| atomic | `fx.UniversalAtomic{Add,Max,Min,…}` | |
| MMA fallback | `fx.rocdl.UniversalFMA(ty)` | scalar FMA |

## 3. The example-04 operand chain

The canonical GEMM chain (example 04):

```
 gA (buffer) ─BufferCopy128b─► regs ─UniversalCopy128b─► sA (LDS, swizzled, 2 stages)
                                                            │ make_tiled_copy_A
                                                            ▼
 gB (buffer, preshuffled view) ─BufferCopy128b─► mma_frag_B[stage]   (direct to VGPR)
                                                            ▼
                                   fx.gemm(tiled_mma, C, A[k], B[k, stage], C)
                                                            ▼
 mma_frag_C (f32) ─ .load().to(f16) ─► frag_f16 ─BufferCopy16b (tiled_copy_C)─► gC
```

- Pass `fragment_A` / `fragment_B` / `fragment_C` the **block-level** tile (`bA`, `sA[...,0]`),
  not a `partition_*` output.
- Advance a K-tile by the buffer atom's `soffset=next_k * k_stride`. The per-lane
  addresses stay the same, and only a scalar offset changes.
- Index a sub-K slice of a fragment with a nested coordinate: `A[None, None, (None, k)]`.

## 4. API stability

`docs/api_stability.md` defines what is stable: exports of `flydsl.expr` direct-child modules
and `flydsl.compiler`, with a deprecation window. Raw `_mlir` dialect ops, `_`-prefixed helpers
(for example `_get_buffer_flags`) and the `kernels/` library are **not** stable.
Prefer stable atoms in kernels you want to keep.

Sources: [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) — `docs/kernel_authoring_guide.md`, `docs/language/storage_and_allocator.md`, `docs/api_stability.md`, `examples/01-vectorAdd.py`, `examples/02-tiledCopy.py`, `examples/03-tiledMma.py`, `examples/04-preshuffle_gemm.py`, `python/flydsl/expr/rocdl/universal.py`, `python/flydsl/expr/rocdl/cdna4.py`, `python/flydsl/expr/gpu.py` (`SharedAllocator`), `python/flydsl/compiler/kernel_function.py` (`known_block_size`), `python/flydsl/utils/smem_allocator.py`, `kernels/common/buffer_ops.py`; [AMD CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) (LDS DMA widths, `ds_read_*_tr_*`).
Paths drift; verify before use.

---
Related: [layout-algebra](layout-algebra.md) · [pitfalls](pitfalls.md) · [compilation-pipeline](compilation-pipeline.md) · [matrix-instructions](../../kernel-optimization/levers/matrix-instructions.md)
