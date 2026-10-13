---
name: kernel-authoring-primitives
description: "Map of FlyDSL authoring primitives — @kernel/@jit boundary types, copy and MMA atoms, tiled copy/MMA and the G→LDS→VGPR→MFMA→global chain, buffer tensors, LDS allocation (SharedAllocator), loops and branches."
source: https://github.com/ROCm/FlyDSL (docs/kernel_authoring_guide.md, docs/language/storage_and_allocator.md, docs/api_stability.md, examples/01-04, python/flydsl/expr/rocdl/*.py)
---

# FlyDSL kernel-authoring primitives

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
```

## 1. Two decorators, one boundary

```
 host  ──►  @flyc.jit  launch(...)          builds layouts / TiledMma / TiledCopy, picks grid
              │  kernel(args).launch(grid=, block=, smem=, stream=)
              ▼
 device     @flyc.kernel  k(...)            traced once per cache key → HSACO
```

| Boundary type | Meaning | Cache key? |
|---|---|---|
| `fx.Tensor` | memref descriptor (torch → DLPack) | dtype + layout directives |
| `fx.Constexpr[T]` | baked into IR | **value** |
| `fx.Int32` / `fx.Int64` / `fx.Float32` … | runtime scalar | type only |
| `fx.Stream` | HIP stream | type only |
| `fx.TiledMma`, `fx.TiledCopy` | built on host, passed as typed args | their static layout |

- Build tiled atoms in the `@jit` launcher and pass them in (example 04). Kernel code then
  stays shape-generic.
- `known_block_size=[x,y,z]` on `@flyc.kernel` sets the block size that the backend uses
  to derive `max_flat_workgroup_size`. Static launch dims are inferred. Set it explicitly when
  the launch dims are dynamic, so the backend does not fall back to its default
  work-group size bound.
- `smem=` at launch = dynamic LDS bytes (for `get_dyn_shared` views).

## 2. Atoms = one instruction

| Kind | Atom | Notes |
|---|---|---|
| copy, generic | `fx.UniversalCopy{8..128}b()` | plain load/store; register ↔ LDS |
| copy, buffer | `fx.rocdl.BufferCopy{8,16,32,64,128}b(cache_modifier)` | buffer descriptor; atom state `soffset` |
| copy, G→LDS DMA | `fx.rocdl.BufferCopyLDS{32,128}b()` | compiler inserts the `vmcnt` wait |
| copy, async G→LDS (CDNA4) | `rocdl.cdna4.BufferLoadAsyncLDS`, `GlobalLoadAsyncLDS` | you track with `asyncmark()` / `wait_asyncmark(n)` |
| copy, LDS transpose read (CDNA4) | `rocdl.cdna4.LDSReadTrans(gran, bits)` | `ds_read_*_tr_*`; no explicit transpose |
| atomic | `fx.UniversalAtomic{Add,Max,Min,…}` | |
| MMA | `fx.rocdl.MFMA(m,n,k, ab_ty, acc_ty)` | CDNA MFMA |
| MMA, scaled (CDNA4) | `rocdl.cdna4.MFMA_Scale(...)` | MXFP4/6/8 with block scale |
| MMA, RDNA / gfx1250 | `fx.rocdl.WMMA(...)` | wave32 |
| MMA, MX-scaled (gfx1250) | `fx.rocdl.WMMAScale(...)` | E8M0 scales as atom state |
| copy, TDM tile DMA (gfx1250) | `fx.rocdl.make_tdm_atom(g, extents, num_warps=)` | whole-tile async G↔LDS; needs a raw VA, not a buffer tensor; drain with `tdm_ops.tensor_wait` |
| MMA fallback | `fx.rocdl.UniversalFMA(ty)` | scalar FMA |

LDS DMA widths are 1/2/4 B on all targets, plus 12/16 B on gfx950. There is no 8-B
form, and `BufferCopyLDS64b` raises.

## 3. Tiled ops and the operand chain

```
 TiledCopy = copy_atom × TV layout × tile      make_tiled_copy(atom, layout_tv, tile)
 TiledMma  = mma_atom  × atom layout (M,N,K reps) [× permutation tile]

 make_tiled_copy_A/B/C(atom, tiled_mma)   → copy whose TV layout matches the MMA operand
 thr = tiled.get_slice(tid)               (TiledMma: thr_slice(tid))
   thr.partition_S(src) / partition_D(dst)   per-thread view of a tile
   thr.retile(frag)                           MMA fragment seen through the copy's TV layout
 thr_mma.make_fragment_A/B/C(tile[, stages=n])  register fragment (+ stage mode)
 fx.copy(atom, src, dst, pred=, soffset=)   fx.gemm(tiled_mma, D, A, B, C, traversal_order=)
```

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

## 4. Global memory: buffer tensors

```python
A = fx.rocdl.make_buffer_tensor(A)                       # max_size=True: size 0xFFFFFFFF
A = fx.rocdl.make_buffer_tensor(A, max_size=False)       # size = cosize(layout)·bytes, OOB checked
A = fx.rocdl.make_buffer_tensor(A, num_records_bytes=n)  # explicit bound
```

Out-of-range lanes then read 0 and drop stores. That is the cheap way to handle partial tiles
(see [varlen-idioms §4](varlen-idioms.md)).
Raw `buffer_ops.buffer_load(rsrc, off, vec_width, dtype)` is legacy and lives in the kernel
library (`kernels/common/buffer_ops.py`), not in `flydsl.expr`. Its offset is in
**elements of `dtype`**. Use it only for a scalar-base access with no layout form.

## 5. LDS allocation

| API | Status | How |
|---|---|---|
| `fx.SharedAllocator(static=True)` | current | `allocate(T)` per buffer → `Storage[T]`; `.peek()` / `.poke(v)` |
| `fx.SharedAllocator(static=False)` | current | compile-time offsets from one `get_dyn_shared` base |
| `fx.get_dyn_shared(dtype)` + `make_view` | current | raw dynamic LDS; set `smem=` at launch |

```python
lds = fx.SharedAllocator().allocate(fx.Array[fx.Float16, TM * TN]).peek()
v4  = fx.SharedAllocator().allocate(fx.Align[fx.Vector[fx.Float32, 4], 16])
```

- One `SharedAllocator` per kernel. With `static=True` the compiler sizes a static LDS global,
  so leave `smem` unset. With `static=False` the launch infers `smem` from `allocated_bytes`.
- Allocation does not initialize memory. Group buffers with `@fx.struct`; overlay buffers that
  are never live together with `@fx.union`.
- Capacity: 64 KB (gfx942), 160 KB (gfx950), 320 KB (gfx1250). Use `flydsl.runtime.device`
  helpers to check.

## 6. Control flow (AST-rewritten)

| You write | Becomes | Index var |
|---|---|---|
| `for i in fx.range_constexpr(N)` | unrolled at trace time | Python `int` |
| `for i in range(a, b, s)` | `scf.for`; variables assigned in the body become carried | SSA (i32) |
| `for iv, st in range(a, b, s, init=[...])` … `yield [...]` | `scf.for` with explicit iter_args | SSA |
| `if fx.const_expr(flag):` | trace-time branch, no IR | — |
| `if runtime_cond:` | `scf.if` | — |
| `x = a if c else b` | `arith.select` / `scf.if` | — |

Rules and traps for these are in [pitfalls](pitfalls.md).

## 7. Values, wave ops, scheduling

- Scalars: `fx.Int32(..)`, `fx.Int64(..)`, `fx.Float32(..)`. Prefer them over `arith.constant`.
  Use `fx.Index(..)` only where an API needs the `index` type.
- Vectors: `fx.Vector` (`v[i]`, `.bitcast(T)`, `.to(T)`, `Vec.filled`, `Vec.from_elements`,
  `cond.select(a,b)`). Min/max: `fx.max` / `fx.min` (NaN-propagating; `fx.maxnumf` is not).
  Abs: `abs(v)` or `fx.absf(v)`. Integer ceil-div: `fx.ceildiv`.
- Fragments: `frag.load()` → vector, `frag.store(vec)`, `frag.fill(0)`,
  `make_fragment_like(frag, new_dtype)`.
- Wave: `fx.rocdl.readfirstlane`, `ds_bpermute`, `gpu.shuffle_xor(v, sh, 64)` (wave64:
  shifts 32,16,8,4,2,1). A shuffle lowers to `ds_bpermute` (LDS crossbar, `lgkmcnt` wait).
- Waits: `fx.rocdl.s_waitcnt(vmcnt=, lgkmcnt=, expcnt=)` packs the per-arch bitfield for
  gfx942, gfx950, gfx11xx and gfx120x, and raises on gfx1250. An unset counter means "no
  wait", so name every counter you need. Async LDS copies use `asyncmark()` / `wait_asyncmark(n)`.
- Ordering hints: `rocdl.sched_{mfma,vmem,dsrd,dswr}(n)`, `sched_barrier(0)`. See
  [hot-loop-scheduling](hot-loop-scheduling.md).
- Fast math: `rocdl.rcp` (one VALU op, `v_rcp_f32`) and the `exp2` intrinsic (`v_exp_f32`)
  have lower precision than the `math.*` versions.
- Debug print: `fx.printf("tid={} v={}", tid, v)`.

## 8. API stability

`docs/api_stability.md` defines what is stable: exports of `flydsl.expr` direct-child modules
and `flydsl.compiler`, with a deprecation window. Raw `_mlir` dialect ops, `_`-prefixed helpers
(for example `_get_buffer_flags`) and the `kernels/` library are **not** stable.
Prefer stable atoms in kernels you want to keep.

Sources: [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) — `docs/kernel_authoring_guide.md`, `docs/language/storage_and_allocator.md`, `docs/api_stability.md`, `examples/01-vectorAdd.py`, `examples/02-tiledCopy.py`, `examples/03-tiledMma.py`, `examples/04-preshuffle_gemm.py`, `python/flydsl/expr/rocdl/universal.py`, `python/flydsl/expr/rocdl/cdna4.py`, `python/flydsl/expr/gpu.py` (`SharedAllocator`), `python/flydsl/compiler/kernel_function.py` (`known_block_size`), `python/flydsl/utils/smem_allocator.py`, `kernels/common/buffer_ops.py`; [AMD CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) (LDS DMA widths, `ds_read_*_tr_*`).
Paths drift; verify before use.

---
Related: [layout-algebra](layout-algebra.md) · [pitfalls](pitfalls.md) · [compilation-pipeline](compilation-pipeline.md) · [matrix-instructions](../../kernel-optimization/levers/matrix-instructions.md)
