---
name: pitfalls
description: FlyDSL traps that compile but run wrong, and how to debug them — trace-time vs runtime values, range/loop-carry rules, side-effect branches, element-unit offsets, MFMA operand order, stale caches, hangs, and an isolation workflow.
source: https://github.com/ROCm/FlyDSL (.claude/skills/debug-flydsl-kernel/SKILL.md, .claude/skills/flydsl-kernel-authoring/SKILL.md, docs/kernel_tuning_guide.md, python/flydsl/compiler/ast_rewriter.py)
---

# FlyDSL pitfalls

Branch-local definitions, early exit, and persistent-loop view caching are in
[varlen-idioms](varlen-idioms.md). This note covers the rest.

## 1. Trace time vs run time

```
 Python value  (int, bool, list, Constexpr)    → decided while tracing → no IR
 SSA value     (tid, bid, loads, fx.Int32(rt)) → decided on the GPU    → scf / arith IR
```

| Trap | Symptom | Fix |
|---|---|---|
| `if fx.const_expr(lane == 0)` | trace error, or a branch taken for all lanes | `const_expr` only on Python values. `tid`, `lane`, `wave_id` are SSA, even with `known_block_size` |
| `for i in range(4): lst[i]` | `i` is SSA, cannot index a Python list | `fx.range_constexpr(4)` |
| a helper writes a captured outer variable | wrong / stale value after the call | pass it in, return it out |
| runtime `if` with side effects or loop state inside a helper | result-type errors | split the bodies into local functions; wrap the `if` in a local `@flyc.jit` dispatcher |
| a variable initialized to `None` before a runtime loop that assigns it | `TypeError` from the loop dispatcher | initialize with a typed value (`fx.Float32(0.0)`) |

## 2. Loops

| Form | Lowers to |
|---|---|
| `range(a, b, s, init=[...])` | `scf.for` (bounds cast to index) |
| `range(a, b, s)`, no `init` | `scf.for`; variables assigned in the body become loop-carried automatically |
| `fx.range_constexpr(n)` | unrolled |

Use **typed DSL bounds** (`fx.Int32(..)` / `fx.Index(..)`)
for runtime loops, and `range_constexpr` for unrolling. Then confirm `scf.for` in the
first IR dump. Bounds must be i32 (an index type is cast, with a warning).

Loop-carried state (explicit `init=`):
- Keep the slot order the same across `init`, `state`, `yield`, and `results`. Each slot keeps
  one meaning and one MLIR type.
- Supported slot types: scalar f32/i32/i64/index and vectors. Unwrap to raw `ir.Value` only
  where a low-level helper needs it.
- Python `data = next_data` inside an *unrolled* loop is invisible to MLIR. Both names alias
  one SSA value, and LLVM hoists the "prefetch" as loop-invariant. Use real carried state
  ([lds-and-prefetch-idioms](lds-and-prefetch-idioms.md)).

## 3. Addressing and data

- `buffer_load(rsrc, off, vec_width, dtype)`: `off` is in **elements of `dtype`**. To read
  packed fp8 bytes as `i32`, use `byte_off // 4`. Recompute offsets every time you change
  dtype, packing, or vector width.
- `Vector.store` needs a vector. Make a 1-element vector with `Vec.from_elements([x], T)`.
- **MFMA operand order:** `mfma(A, B, acc)` — A gives the M rows, B gives the N columns. If
  you swap them, you get a transposed tile with no error. The accumulator lane→(row, col)
  map depends on which tensor you put in A. Derive it again for every kernel. Do not copy the
  forward-pass mapping into a backward pass.
- `.transpose().clone()` in torch keeps the strided layout. The first contiguous use then
  does a hidden full copy. Use `.contiguous()` or `clone(memory_format=torch.contiguous_format)`.

## 4. Stale results

The first suspect for "my fix had no effect" is the cache
([compilation-pipeline §3](compilation-pipeline.md)). Helpers imported from another directory
and rebuilt C++ passes do not invalidate it. When in doubt, run with
`FLYDSL_RUNTIME_ENABLE_CACHE=0`.

## 5. Hangs

- A `gpu.barrier()` inside a branch that only some threads take will deadlock. Barriers
  must be reached by the whole workgroup.
- A runtime loop with `step = 0`, or with `stop < start` under an unsigned compare, spins
  forever. Print the bounds on the host before launch.

## 6. Isolation workflow (wrong numbers)

```
 1  clear cache / disable it                           → rules out stale code
 2  dump IR: did the loop / layout / atom lower as intended?
 3  all-ones inputs                                    → output must be uniform;
      a non-uniform result = addressing / layout bug   (blind to operand mis-pairing)
 4  one tile / one partition / one K-step              → splits main-loop bugs from reduce bugs
 5  fx.printf from tid 0 of block 0                    → trace one thread's addresses by hand
 6  compare intermediates with a torch reference       → find the first wrong stage
```

Clues from how the output looks:

| Looks like | Usually |
|---|---|
| all zeros | wrong output stride/slot, a store predicated off, or reduce inputs never initialized |
| > 50 % wrong | missing tiles/partitions (grid smaller than work), or a layout mismatch |
| 1–5 % wrong | quantization / scale mismatch, or a scale applied twice |
| NaN only on masked rows | `exp(-inf - (-inf))`. Guard the max with `select(max > -inf, diff, 0)` and the sum with `select(sum > 0, sum, 1)` |
| "close but low SNR" softmax | log-sum-exp in a different domain (natural log vs scaled log2) from the reference |

Sources: [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) — `.claude/skills/debug-flydsl-kernel/SKILL.md`, `.claude/skills/flydsl-kernel-authoring/SKILL.md`, `docs/kernel_tuning_guide.md`, `docs/architecture_guide.md`, `python/flydsl/compiler/ast_rewriter.py` (`scf_range`, `_transform_for_auto`, `scf_for_dispatch`); [PyTorch `Tensor.clone`](https://pytorch.org/docs/stable/generated/torch.clone.html) (`memory_format=torch.preserve_format`).
Paths drift; verify before use.

---
Related: [varlen-idioms](varlen-idioms.md) · [compilation-pipeline](compilation-pipeline.md) · [failure-modes](../../kernel-optimization/failure-modes/failure-modes.md)
