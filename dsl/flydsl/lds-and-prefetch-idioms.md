---
name: lds-and-prefetch-idioms
description: "How to express LDS staging and load prefetch in FlyDSL — the stage-as-a-layout-mode ping-pong, constexpr stage indices by unrolling, swizzle/padding as layouts, sync vs async G→LDS atoms, and the loop-carried register prefetch (init=/yield) with its ISA check."
source: https://github.com/ROCm/FlyDSL (examples/04-preshuffle_gemm.py, docs/kernel_tuning_guide.md, .claude/skills/lds-optimization/SKILL.md, .claude/skills/prefetch-data-load/SKILL.md, python/flydsl/expr/rocdl/universal.py, cdna4.py)
---

# LDS and prefetch idioms in FlyDSL

The theory (bank math, XOR vs padding per arch, double buffering, async copy) is
DSL-neutral and lives in [memory-hierarchy](../../kernel-optimization/levers/memory-hierarchy.md)
and [pipelining-scheduling](../../kernel-optimization/levers/pipelining-scheduling.md).
This note shows how to write it in FlyDSL.

## 1. First, name the LDS problem

Triage the stall before picking a fix (bank conflict vs exposed latency vs cross-wave
serialization): [memory-hierarchy](../../kernel-optimization/levers/memory-hierarchy.md),
lessons section.

## 2. Ping-pong = one extra layout mode

Example 04 puts the stage index in the LDS layout and in the B fragment:

```python
sA = fx.make_view(fx.get_dyn_shared(fx.Float16),
       fx.make_composed_layout(fx.static(fx.SwizzleType.get(3, 3, 3)),
         fx.make_ordered_layout((BM, BK, STAGES), (1, 0, 2))))   # (BM, BK, stage)
frag_B = thr_mma.make_fragment_B(gB_k, stages=2)                 # (V, N, K, stage)
```

One stage, in order:

```
 run_stage(read, next_k):          write = read ^ 1
   ① issue next global loads   A(next_k) → regs ;  B(next_k) → frag_B[write]   (soffset = next_k·stride)
   ② for k in range_constexpr(BK/32):  LDS→regs sA[..,k,read] ;  gemm(A[k], B[k,read])
   ③ regs → sA[..,write]
   ④ gpu.barrier()                  one barrier per stage
   ⑤ sched_* hints                  see hot-loop-scheduling
```

```
 prologue: load k=0 → sA[0], frag_B[0]; barrier
 loop k += 2:  run_stage(read=0, next=k+1);  run_stage(read=1, next=k+2)
 tail:         run_stage(0, last);           run_stage(1, None, read_next=False)
```

- **Unroll by the stage count.** Then `read` / `write` are Python ints. Every LDS address is
  a constant offset from the base, with no runtime select between buffers.
- The loads in ① are issued before the MFMAs in ②. Source order is not schedule order:
  the `sched_*` hints place them inside the MFMA stream
  ([hot-loop-scheduling](hot-loop-scheduling.md)).
- LDS bytes = `cosize(sA layout) · elem_bytes`. Pass it as `smem=` at launch.

## 3. Swizzle or pad = choose the layout

```python
swz = fx.make_composed_layout(fx.static(fx.SwizzleType.get(B, M, S)), base)   # XOR, 0 extra bytes
pad = fx.make_layout((BM, BK), (BK + PAD, 1))                                  # padding, cosize grows
```

- Because the swizzle is part of the view, writes and reads use the same transform.
- Direct global→LDS DMA writes lane-contiguous addresses. It can break a swizzled layout
  that a register-staged store respects. Test each tile depth.
- Layout rules: [layout-algebra §5](layout-algebra.md).

## 4. Write→read distance in DSL terms

FlyDSL (LLVM) inserts `s_waitcnt`. You cannot set `vmcnt(N)` from the sync atoms. You
control **distance** by where you put code:

```python
sA_store(data)                     # ds_write: returns at once
nxt = next_offsets(); g = buffer_load(next)  # independent: address math, next global load
gpu.barrier()                      # by now the write has landed
v = sA_load(...)
```

Good fillers, best first: next-phase global loads → address math → independent MFMA →
scalar arg loads. Bad fillers: anything that reads the just-written LDS, more LDS traffic,
and anything that pushes VGPRs past the occupancy step.

## 5. G→LDS copy atoms

| Atom | Wait handling | Width |
|---|---|---|
| `BufferCopy128b` → regs → `UniversalCopy128b` → LDS | compiler | 16 B/lane, costs staging VGPRs |
| `rocdl.BufferCopyLDS{32,128}b` (DMA) | compiler inserts `vmcnt` before the LDS read | 4 B; 16 B on gfx950 |
| `rocdl.cdna4.BufferLoadAsyncLDS` / `GlobalLoadAsyncLDS` | you: `asyncmark()` … `wait_asyncmark(n)` | gfx950 |
| `rocdl.cdna4.LDSReadTrans(...)` (read side) | — | `ds_read_*_tr_*`; MFMA operand without a transpose pass |

DMA removes the staging registers (≈ 4 VGPRs per 16-B load). That is worth it when it lets
you deepen the tile without losing occupancy. The sync DMA atom waits right away, so it
hides little latency on its own.

## 6. Register prefetch with loop-carried state

For loops with no LDS stage (paged-KV decode, gathers), prefetch into registers with
`range(..., init=[...])` carried state. The pattern (prologue, loads-first body, `yield`,
epilogue) is in [flydsl-kernel-authoring](https://github.com/ROCm/FlyDSL/tree/main/.claude/skills/flydsl-kernel-authoring) §3 "Runtime Loops with Loop-Carried Values". An unrolled
`data = next_data` swap does not work ([pitfalls §1](pitfalls.md)). What the skill does not say:

- Carry **everything** the next iteration needs: payload, block-table entries, page ids,
  scales, running max/sum. A load that depends on another load (table → page) stays in
  order inside the prefetch block.
- If the scale loads are conditional (quant modes), the prefetch block repeats the same
  `const_expr` conditions.
- **ISA check:** the next-iteration `buffer_load_dwordx4`s appear at the *end* of the loop body
  (after the MFMAs). The prologue has one set and the epilogue has none. If they are at the top
  next to their use, the prefetch did not take effect.
- Cost: each carried 16-B load = 4 VGPRs. Check the occupancy step first
  ([gfx942-registers-occupancy](../../hardware/amd-cdna/gfx942-registers-occupancy.md)). Do not
  cap registers to make room (forcing AGPRs off made MFMA kernels much slower).
- Skip it when the loop is all loads with little compute, when MFMA use is already above 90 %,
  or when the kernel is at 1 wave and spilling.

## 7. Verify

Correctness on the same shapes → the targeted stall moved (ATT) → LDS bytes fit the arch
(64 / 160 KB) → VGPR count did not cross an occupancy step.

Sources: [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) — `examples/04-preshuffle_gemm.py`, `docs/kernel_tuning_guide.md` (§2–§4), `.claude/skills/lds-optimization/SKILL.md`, `.claude/skills/prefetch-data-load/SKILL.md`, `python/flydsl/expr/rocdl/universal.py` (`BufferCopyLDS`), `python/flydsl/expr/rocdl/cdna4.py` (`BufferLoadAsyncLDS`, `LDSReadTrans`); [AMD CDNA3 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) (`s_waitcnt`, LDS); [AMD CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) (`ds_read_*_tr_*`, LDS DMA widths).
Paths drift; verify before use. Verify specifics against the primary source.

---
Related: [layout-algebra](layout-algebra.md) · [hot-loop-scheduling](hot-loop-scheduling.md) · [operator-recipes](operator-recipes.md) · [memory-hierarchy](../../kernel-optimization/levers/memory-hierarchy.md)
