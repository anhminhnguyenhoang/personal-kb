---
name: varlen-idioms
description: "Verified pure-FlyDSL layout-API idioms for varlen/jagged kernels — device scalar read, runtime base-offset views, runtime early-exit (scf.IfOp), bounded buffer descriptors (OOB-drop), plain-B MFMA K-layout, bf16 broadcast-bias epilogue"
source: distilled from building a varlen grouped-GEMM in FlyDSL
---

# Varlen / jagged kernel idioms (FlyDSL layout API)

These are idioms for varlen/jagged kernels (grouped GEMM over packed rows with
device-resident offsets), written in pure `flydsl.expr` (`import flydsl.expr as fx`)
with no other dependencies. They combine a **runtime base-offset** with the
layout/partition API, so a varlen kernel need not fall back to manual `buffer_load`.

> **Verify against current source before relying.** The installed venv can lag the
> repo — e.g. the installed `make_buffer_tensor` had no `num_records_bytes` kwarg
> even though the repo source did, which is why the bounded descriptor below is
> built inline from `make_ptr` rather than via the helper.
>
> Uses private APIs (`_get_buffer_flags`, `SmemPtr._view_cache`) that may change between FlyDSL versions.

## The varlen contract

For each group `b` over its packed row slice `[s,e) = [seq_offsets[b], seq_offsets[b+1])`:

```
Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b][None, :]
  (M_b × N)     (M_b × K)      (K × N)     (1 × N broadcast)
```

`seq_offsets` is **device-resident** int32 — the host does not know any `M_b` at
launch, so group→row resolution happens on-device. That single constraint forces
every idiom below.

## 1. Device scalar read (e.g. `seq_offsets[b]`)

There is **no `get_scalar` helper for globals**. Read a device scalar with a
buffer resource + `buffer_load`:

```python
rsrc = fx.buffer_ops.create_buffer_resource(SEQ_OFFSETS, max_size=True)
seq_start = fx.buffer_ops.buffer_load(rsrc, fx.Int32(b),   vec_width=1, dtype=fx.T.i32())
seq_end   = fx.buffer_ops.buffer_load(rsrc, fx.Int32(b)+fx.Int32(1), vec_width=1, dtype=fx.T.i32())
```

**Scalarize uniform values** with `readfirstlane` so they land in SGPRs:
`seq_start = fx.rocdl.readfirstlane(fx.T.i32(), seq_start)`. This matters in the
epilogue — a divergent C descriptor forces a per-lane store waterfall (257
`v_readfirstlane_b32` + a 64-wide exec-mask loop collapse to ~0 once scalarized).

## 2. Runtime (possibly unaligned) base-offset view

To make `flat_divide` re-tile from a runtime row, add the offset to the iterator
then re-wrap as a buffer tensor:

```python
a_row_off = fx.Int64(seq_start) * fx.Int64(K)   # i64 BEFORE the stride multiply
A_g  = fx.make_view(fx.add_offset(fx.get_iter(A), fx.make_int_tuple(a_row_off)), fx.get_layout(A))
A_buf = fx.rocdl.make_buffer_tensor(A_g, ...)
gA_k  = fx.flat_divide(A_buf, (BLOCK_M, BLOCK_K))[None, None, block_m_idx, None]
```

- The `add_offset` offset must be a **scalar leaf** via `fx.make_int_tuple(x)`. A
  Python `(x,)` tuple fails with *"offset must be a scalar leaf IntTuple"*.
- **i64 is load-bearing at scale.** Build the row-base offset in i64 *before* the
  stride multiply. `seq_start*K` overflows i32 when `seq_start` reaches ~millions
  of rows (e.g. 7.86M·512 = 4.0e9 > 2³¹) and the kernel GPU-faults.

## 3. Runtime early-exit — positive guard, never bare `return`

A plain Python `if runtime_cond:` inside `@flyc.kernel` is auto-rewritten to
`scf.IfOp` (ast_rewriter `ReplaceIfWithDispatch`). Use the **positive guard**
wrapping the whole body — a bare `return` inside a dynamic `if` may not lower
cleanly:

```python
if start_m < M_b:        # runtime → scf.IfOp guarding all compute + store
    ... mainloop + epilogue ...
```

Caveat (branch-local rule): a value defined only inside a branch
leaks as a `NameError` when used after it — hoist it or use a ternary. Empty
groups (`M_b == 0`) are handled for free: every tile fails the guard.

## 4. Bounded buffer descriptor (HW OOB-drop for partial tiles)

The fix that makes the skewed varlen case correct. A short group's
partial bottom tile must not read/write past its allocation into the next group.
Bound the buffer-desc to the group's real byte size; the hardware then OOB-drops
out-of-range lanes (reads return 0, stores are masked):

```python
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
from flydsl.expr.buffer_ops import _get_buffer_flags

def make_bounded_buffer_tensor(tensor, num_records_bytes):  # num_records_bytes: fx.Int64
    elem_ty = tensor.element_type
    ptr     = fx.get_iter(tensor)
    layout  = fx.get_layout(tensor)
    buf_ptr_ty = fx.PointerType.get(
        elem_ty=elem_ty.ir_type,
        address_space=TargetAddressSpace.BufferDesc,
        alignment=ptr.alignment,
    )
    buf_ptr = fx.make_ptr(buf_ptr_ty, [
        ptr, fx.Int16(0).ir_value(),
        num_records_bytes.ir_value(),
        fx.Int32(_get_buffer_flags()).ir_value(),
    ])
    return fx.make_view(buf_ptr, layout)

# bound BOTH A and C to the group's real rows:
A_buf = make_bounded_buffer_tensor(A_g, fx.Int64(fx.Int32(M_b) * fx.Int32(K) * fx.Int32(2)))
C_buf = make_bounded_buffer_tensor(C_g, fx.Int64(fx.Int32(M_b) * fx.Int32(N) * fx.Int32(2)))
```

**The fault this prevents is data-dependent and intermittent.** An unbounded
descriptor (`max_size=True`) over an exactly-sized allocation reads unmapped pages
on the last partial tile.

## 5. MFMA K-layout for PLAIN (non-preshuffled) B

Use example 04's K-layout `make_layout((4,4,2),(1,8,4))` **unchanged**, even
without preshuffle. That stride is the natural MFMA fragment K-ordering (ex.04
applies it to A, which is never shuffled). The B *un-shuffle* in ex.04 lives in
its `preshuffle_layout_B` **view** (omit that), **not** the K-layout. The nested
`(2,2)` K-axis the layout produces is **required** for the
`[None, None, (None, block_k_iter)]` inner-loop slice — changing the stride to an
"identity" breaks that slice.

## 6. bf16 epilogue + broadcast bias

```python
# truncate fp32 accumulators + bias to bf16
mma_frag_C_bf16.store(fx.arith.trunc_f(
    fx.T.VectorType.get([C_FRAG_LEN], fx.T.bf16()),
    fx.arith.addf(mma_frag_C.load(), bias_f32)))
# extend bias to fp32 first (extf is the raw op; trunc_f is the wrapper)
bias_f32 = fx.arith.ExtFOp(fx.T.VectorType.get([C_FRAG_LEN], fx.T.f32()), bias_frag.load()).result
```

Broadcast bias = a **stride-0 M view** of the group's `(N,)` slice, partitioned
with the *same* `make_tiled_copy_C` slice as the output so per-thread N-coords
align:

```python
gBias2d = fx.make_view(fx.get_iter(BIAS_buf), fx.make_layout((BLOCK_M, N), (0, 1)))
```

## scf.for inside a persistent loop (gotchas)

When the tile body lives inside an `scf.for` persistent loop:

- **Build `make_tiled_mma` / copies *inside* the loop body.** If built before and
  merely read inside, the DSL auto-promotes them to `scf.for` iter_args (block
  args), and the gemm-op lowering's `getDefiningOp<MakeTiledMma>` then fails. They
  are loop-invariant, so LICM can still hoist the cheap setup later.
- **Reset `SmemPtr._view_cache = None` each iteration** so the 2nd+ iteration's
  `fx.get_dyn_shared` views don't reuse the previous iteration's SSA values (MLIR
  dominance error).

---
Related: [measurement](../../kernel-optimization/00-method/measurement.md) · [hot-loop-scheduling](hot-loop-scheduling.md) · [work-decomposition](../../kernel-optimization/10-levers/work-decomposition.md)
