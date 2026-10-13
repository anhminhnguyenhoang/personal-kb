---
name: tile-distribution
description: "CK's core layout idea: coordinate-transform chains (pass-through, pad, merge, unmerge, embed, xor) plus tile_distribution_encoding (R/H/P/Y spaces) that map (warp, lane) to tensor coordinates and register slots at compile time, and the tile_window / distributed-tensor verbs built on them."
source: https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/tile_distribution.html ; https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/tile_window.html ; https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/core/tensor
---

# Tile distribution and coordinate transforms

## 1. Coordinate transforms (both front-ends)

A tensor descriptor (classic) or `tensor_adaptor` (CK-Tile) is a chain of transforms
from logical ("top") indices to the memory ("bottom") offset.

| Transform | Does | Typical use |
|---|---|---|
| pass_through | keep a dim | untouched dims |
| pad (right pad) | extend a dim, mark the tail invalid | ragged M/N/K, head-dim padding |
| unmerge | split one dim into several | tile a dim: `M → (M0, M1)` |
| merge | fuse several dims into one | flatten `(N, Ho, Wo) → M` for implicit-GEMM conv |
| embed | `offset = Σ idx·stride` | sliding windows (conv input) |
| xor | `col ^= f(row)` | LDS bank swizzle |

- All lengths that are compile-time constants fold away. `CalculateOffset` becomes a few
  integer ops.
- Runtime-length merges use **magic division** (precomputed multiplier + shift) so `idiv`
  becomes `mulhi + shift + sub`.
- Conv forward, backward-data and backward-weight are the **same GEMM core** with a
  different transform chain (which tensor plays A, B, C). The transpose lives in the
  descriptor, not the kernel.

## 2. The distribution encoding

`tile_distribution_encoding<Rs, Hss, Ps2RHssMajor, Ps2RHssMinor, Ys2RHsMajor, Ys2RHsMinor>`

```
  X  : tensor coords of the tile           (e.g. M, N)
  H  : each X split into a factor list     M = Repeat × WarpPerBlock × LanePerWarp × Vector
  P  : who executes  (P0 = warp id, P1 = lane id)
  Y  : per-thread element loops → become VGPR slots
  R  : replication (same data held by several P), optional

   (warp, lane) ──P→H──┐
                       ├─→ X coordinate ─→ memory address   (PsYs2Xs adaptor)
   y-index ─────Y→H────┘
   y-index ─────────────→ register slot                     (Ys2D descriptor)
```

- Each H factor is owned by exactly one of P (spread across threads) or Y (looped in a
  thread). The innermost Y factor with unit stride sets the **vector width** of each load.
- Typical H pattern: `<Repeat, Warp, Lane, Vector>`, e.g. `<4,2,8,4>` per dim.
- A block distribution = **outer** (warps × iterations per warp) embedded with **inner**
  (the WarpGemm's lane layout): `make_embed_tile_distribution_encoding(outer, WG::CWarpDstrEncoding{})`.
- R dims are how a reduction result is shared (all lanes in a row hold the row max).
- You rarely write an encoding by hand. The Policy derives it from tile sizes and the
  chosen WarpGemm. Hand-written encodings are the top source of silent lane-mapping bugs.

## 3. Views, windows, registers

| Object | Holds | Key property |
|---|---|---|
| `buffer_view` | pointer, address space, coherence, invalid-value | OOB reads return a fixed value (zero) via the buffer range check, no branch |
| `tensor_view` | buffer_view + descriptor | `pad_tensor_view` adds right padding per dim |
| `tile_window` | view + origin + lengths + distribution | precomputes per-thread coordinates once; `move_tile_window` is an add |
| distributed tensor | `thread_buffer` in VGPRs | size known at compile time; `operator[]` is constexpr indexing |

Tile verbs: `load_tile`, `store_tile`, `update_tile` (atomic add), `async_load_tile`
(global→LDS, no VGPR staging), `shuffle_tile`, `slice_tile`, `sweep_tile` (lambda per
Y element), `set_tile_if` (masked write, no branch), `block_tile_reduce` (+ `_sync`
across lanes), `cast_tile`.

## 4. Idioms the abstraction gives you for free

- **Vector width is derived, not chosen**: the window scans Y dims for the longest
  stride-1 run and emits the widest `buffer_load` that fits.
- **Bounds-check DCE**: OOB checking is a template flag. Aligned bulk tiles compile with
  no guard; only padded edges pay.
- **Iteration order is a type**: a space-filling curve over Y (row-major or snake) picks
  the access order. Snake order reuses the last element of each row.
- **LDS write vs read windows can differ.** Write K to LDS with the DRAM distribution,
  read it back with the MFMA operand distribution. The LDS descriptor between them holds
  the swizzle.
- **Fuse cheap transforms into the load** (`load_tile_with_elementwise`) or into the
  LDS read (transpose-on-read where the target has `ds_read_tr*`).

Sources: [Tile distribution docs](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/tile_distribution.html) ·
[Tile window docs](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/tile_window.html) ·
[ck_tile/core/tensor](https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/core/tensor) ·
[CK-Tile FlashAttention blog (source)](https://github.com/ROCm/rocm-blogs/blob/release/blogs/software-tools-optimization/ck-tile-flash/README.md)
Paths drift; verify before use.

---
Related: [architecture-classic-vs-tile](architecture-classic-vs-tile.md) · [gemm-pipelines-and-schedulers](gemm-pipelines-and-schedulers.md) · [memory-hierarchy](../../kernel-optimization/levers/memory-hierarchy.md)
