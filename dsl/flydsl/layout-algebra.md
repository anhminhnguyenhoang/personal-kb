---
name: layout-algebra
description: CuTe-style layout algebra as FlyDSL exposes it — shape:stride layouts, the four divide flavours and when to use each, thread-value layouts, swizzled LDS as a composed layout, and preshuffle as a layout view.
source: https://github.com/ROCm/FlyDSL (docs/cute_layout_algebra_guide.md, docs/layout_system_guide.md, examples/04-preshuffle_gemm.py, examples/notebooks/04_layout.ipynb, 05_tiled_copy_and_swizzle.ipynb); https://github.com/NVIDIA/cutlass (include/cute/layout.hpp, swizzle.hpp)
---

# Layout algebra in FlyDSL

FlyDSL uses the CuTe algebra (CUTLASS `include/cute/`) unchanged in meaning.
The difference is the target: AMD wave64, MFMA, LDS, buffer descriptors.
All `fx.*` layout calls emit IR. Call them inside `@flyc.kernel` or `@flyc.jit`.

## 1. One layout = one function

```
 Layout  L = Shape : Stride          e.g. (8,16):(1,8)   (column-major 8×16)
 index   = Σ coord_i · stride_i      crd2idx((3,5), L) = 3·1 + 5·8 = 43
 inverse = div/mod by shape          idx2crd(43, L)    = (3,5)

 size(L)   = Π shape      = 128      (domain)
 cosize(L) = max(idx) + 1 = 128      (codomain = bytes/elements you must allocate)
```

- Shapes nest: `(9, (4, 8))`. Nesting = tiling hierarchy (block → wave → lane → value).
- Leaves are static ints or runtime SSA values. Static leaves fold away in
  `fly-layout-lowering`. Dynamic leaves cost address arithmetic.
- Stride 0 = broadcast (one value seen by every coordinate on that mode).
- `make_ordered_layout(shape, order)` builds a compact layout from a stride order.
  `order=(1,0)` = row-major for 2-D.

## 2. Operations

The op list (`composition`, `complement`, `coalesce`, `right_inverse`, `recast_layout`, products,
divides, `select`/`group`/`slice`) is in [flydsl-kernel-authoring](https://github.com/ROCm/FlyDSL/tree/main/.claude/skills/flydsl-kernel-authoring) §2. The sections below explain the parts the
list does not: how divides group modes, how TV layouts and swizzles behave.

## 3. Divides: same tiles, different mode order

All divides split a layout by a tiler into *(inside-tile, which-tile)*. They only
differ in how the result modes are grouped.

```
 A: (M, K)        tiler: (BM, BK)

 logical_divide → ((BM, M/BM), (BK, K/BK))     per-dim pairs, hierarchy kept
 zipped_divide  → ((BM, BK), (M/BM, K/BK))     all tile modes first, then rest
 tiled_divide   → ((BM, BK), M/BM, K/BK)       tile mode, rest flattened
 flat_divide    → (BM, BK, M/BM, K/BK)         everything flat
```

Pick by how you will index next:

| Need | Use | Index |
|---|---|---|
| this block's tile | `zipped_divide` | `slice(t, (None, bid))` |
| this block's tile, keep the K-tile axis for a loop | `flat_divide` | `t[None, None, bid_m, None]` → `(BM, BK, k)` |
| 1-D block → per-thread split | `logical_divide` twice | `slice(.., (None, bid))`, then `(None, tid)` |

There is no `local_tile` / `local_partition` helper. Use divide + slice.
The second divide before `slice(..., (None, tid))` matters: the coordinate rank must
match the layout rank.

## 4. Products and thread-value (TV) layouts

Products place one layout *inside* every element of another.

```
 thr = (4,1):(1,1)    4 threads along M
 val = (1,8):(1,1)    8 values per thread along N

 raked_product(thr, val)    T0 T1 T2 T3 T0 T1 ...  (interleaved; coalesced lanes)
 blocked_product(thr, val)  T0 T0 T0 ... T1 T1 ... (contiguous chunk per thread)
```

A **TV layout** maps `(thread, value) → coordinate in the tile`.
`make_layout_tv(thr, val)` returns `(tile_mn, layout_tv)`. A tiled copy is
`copy_atom × layout_tv × tile`. See [kernel-authoring-primitives](kernel-authoring-primitives.md).

To check a TV layout, draw it: `fx.utils.print_typst(layout_or_tiled_copy, file=...)`
writes a Typst diagram of the layout, a `TiledCopy`, or a swizzle.

## 5. Swizzle = a composed layout

An LDS tile with a power-of-two row pitch puts one column of every row on the same bank.
A swizzle XORs some high offset bits into low offset bits.

```
 Swizzle(B, M, S)      (FlyDSL: fx.SwizzleType.get(mask=B, base=M, shift=S))
   yyy = bits [M+S, M+S+B)      ← "row" bits
   zzz = bits [M,   M+B)        ← "column-chunk" bits
   out = off XOR ((off & yyy_mask) >> S)

 composed = make_composed_layout(static(SwizzleType.get(3,3,3)), base_layout)
 sA       = make_view(get_dyn_shared(Float16), composed)
```

Coordinate → base layout → offset → swizzle → final offset.
With `(3,3,3)` on an f16 tile of row pitch 64 elements: offset bits 3–5 pick a
16-B chunk (8 elements) in the row, and bits 6–8 are the low 3 bits of the row index.
So each of 8 consecutive rows sees a different chunk order. A 16-B vector access still
reads one whole chunk, so the vector width does not change.

- Copy the `(B,M,S)` triple from a known-good kernel for your dtype and tile. Recompute
  it when the dtype or row pitch changes.
- The same swizzle is in the view for writes and reads. You cannot pair a swizzled
  store with a linear load by accident.
- Units: a swizzle in a composed layout acts on the **element index** (layout `crd2idx`
  result). A swizzle attached to a *pointer* type (`apply_swizzle`) acts on the raw
  **byte address**. Re-derive the triple when you move it between these forms or change
  the element width.
- Swizzle is built in: `SwizzleType` and composed layouts. Prefer it over hand-written XOR.

## 6. Preshuffle is a view, not a copy

The host permutes the weight once (for example `shuffle_weight(B, layout=(16,16))`). The
kernel then *describes* the permuted bytes with a nested layout and tiles that view as
if it were plain B:

```python
preshuffle_layout_B = fx.make_layout(
    ((16, N // 16), (8, 4, K // 32)),
    ((8, 16 * K), (1, 8 * 16, 8 * 16 * 4)))
B_view = fx.Tensor(fx.make_view(fx.get_iter(B), preshuffle_layout_B))
```

The tiled-copy and MMA code does not change. Only this view changes. Remove the view and
you get plain B. The MFMA K-fragment layout stays the same
([varlen-idioms §5](varlen-idioms.md)).

## 7. Static vs dynamic shapes

`mark_layout_dynamic(leading_dim=, divisibility=)` ([flydsl-kernel-authoring](https://github.com/ROCm/FlyDSL/tree/main/.claude/skills/flydsl-kernel-authoring) §10): the divisibility promise is in
the cache key and keeps 128-bit accesses on a runtime shape. Without it, every shape is a new
compile (static) or a slow path (unknown alignment).

Sources: [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) — `docs/cute_layout_algebra_guide.md`, `docs/layout_system_guide.md`, `python/flydsl/expr/primitive.py`, `include/flydsl/Dialect/Fly/Utils/LayoutUtils.h` (`layoutCrd2Idx`), `lib/Dialect/Fly/Utils/PointerUtils.cpp` (`applySwizzleOnPtr`), `examples/04-preshuffle_gemm.py`, `examples/notebooks/04_layout.ipynb`, `examples/notebooks/05_tiled_copy_and_swizzle.ipynb`; [CUTLASS CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute) — `layout.hpp`, `swizzle.hpp`; PyCute reference `torch/distributed/_pycute/layout.py` ([pytorch/pytorch](https://github.com/pytorch/pytorch)).
Paths drift; verify before use.

---
Related: [kernel-authoring-primitives](kernel-authoring-primitives.md) · [lds-and-prefetch-idioms](lds-and-prefetch-idioms.md) · [memory-hierarchy](../../kernel-optimization/levers/memory-hierarchy.md)
