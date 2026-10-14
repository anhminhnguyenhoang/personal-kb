---
name: pitfalls
description: "FlyDSL traps: unrolled-alias prefetch, SSA list indexing, MFMA operand order, strided torch views, stale cache, hangs, and an isolation workflow for wrong numbers."
source: https://github.com/ROCm/FlyDSL (.claude/skills/debug-flydsl-kernel/SKILL.md, docs/kernel_tuning_guide.md, python/flydsl/compiler/ast_rewriter.py)
---

# FlyDSL pitfalls

## 1. Trace time vs run time, and loops

Two traps:

- `for i in range(4): lst[i]` makes `i` SSA, which cannot index a Python list. Use
  `fx.range_constexpr(4)`.
- `data = next_data` inside an *unrolled* loop is invisible to MLIR. Both names alias one SSA
  value, and LLVM hoists the "prefetch" as loop-invariant. Use real carried state
  ([lds-and-prefetch-idioms](lds-and-prefetch-idioms.md)).

## 2. Operands and torch views

- **MFMA operand order:** `mfma(A, B, acc)` — A gives the M rows, B gives the N columns. If
  you swap them, you get a transposed tile with no error. The accumulator lane→(row, col)
  map depends on which tensor you put in A. Derive it again for every kernel. Do not copy the
  forward-pass mapping into a backward pass.
- `.transpose().clone()` in torch keeps the strided layout. The first contiguous use then
  does a hidden full copy. Use `.contiguous()` or `clone(memory_format=torch.contiguous_format)`.

## 3. Stale results

The first suspect for "my fix had no effect" is the cache. What the key misses is in
[compilation-pipeline §2](compilation-pipeline.md).

## 4. Hangs

- A `gpu.barrier()` inside a branch that only some threads take will deadlock. Barriers
  must be reached by the whole workgroup.
- A runtime loop with `step = 0`, or with `stop < start` under an unsigned compare, spins
  forever. Print the bounds on the host before launch.

## 5. Isolation workflow (wrong numbers)

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

Sources: [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) — `.claude/skills/debug-flydsl-kernel/SKILL.md`, `docs/kernel_tuning_guide.md`, `docs/architecture_guide.md`, `python/flydsl/compiler/ast_rewriter.py` (`scf_range`, `_transform_for_auto`, `scf_for_dispatch`); [PyTorch `Tensor.clone`](https://pytorch.org/docs/stable/generated/torch.clone.html) (`memory_format=torch.preserve_format`).
Paths drift; verify before use.

---
Related: [varlen-idioms](varlen-idioms.md) · [compilation-pipeline](compilation-pipeline.md) · [failure-modes](../../kernel-optimization/failure-modes/failure-modes.md)
