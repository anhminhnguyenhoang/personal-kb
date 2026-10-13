---
name: row-ops-reduce-norm
description: "CK-Tile row-shaped ops (reduce, softmax, rmsnorm/layernorm, add+rmsnorm+quant, smoothquant, topk) all built on one thread→warp→block reduction primitive; the pass-count, Welford, cross-warp routing and trait-flag idioms they share."
source: https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/reduce ; https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/rmsnorm2d ; https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/add_rmsnorm2d_rdquant
---

# CK-Tile row ops: reduce, softmax, norm, quant

## 1. One reduction primitive

```
 thread: sweep own Y elements  ─►  warp: xor-shuffle tree (log2 steps)
                                   ─►  block: lane 0 of each warp writes LDS, all reload + reduce
                                   ─►  optional broadcast back to every lane
```
- Parameterized by `(reduce_op, distribution)`. The distribution's P/R dims say which axes
  are spread over lanes, so the same code reduces rows or columns.
- Power-of-two lane counts give a compile-time unrolled tree.
- One-warp problems skip LDS entirely (`kNeedCrossWarpSync = false`). This is the main
  routing knob: warp-per-row for short rows, block-per-row for long rows.
- Argmax/argmin carry the index through the same tree as a packed `(value, index)`.

Softmax, norms, topk, smoothquant and the FMHA row max/sum all supply lambdas to it.

## 2. Pass count is the design choice

| Op | Passes over the row | Notes |
|---|---|---|
| softmax | 2: max, then `exp(x − max)` and sum | `1/sum` computed once per row; second pass replays registers, no reload |
| rmsnorm | 2: `Σx²`, then `x · rsqrt(mean + ε) · γ` | |
| layernorm | 1 with Welford | running mean / M2; partial stats merge with the M2 correction |
| add + rmsnorm + quant | 3 in one kernel: add + `Σx²`, normalize, absmax + quantize | no intermediate writes; saves two HBM round trips |
| smoothquant | 1 (or 2 for precision) | `x · smooth[c]`, absmax, scale, cast |

## 3. Shared idioms

- **Fast reciprocal option** (`kFastFDiv`): `rcp` instead of IEEE divide in Welford. About
  2× cheaper, slightly less accurate; make it a trait.
- **`v_max3_f32`** computes `max(a, |b|, |c|)` in one instruction for absmax.
- **Trait flags compile out unused paths**: `kHasGamma`, `kPadN`, `kSaveInvRms`,
  `kFusedAdd`. One template, many shipped variants, no runtime branches.
- **Save `1/rms` for backward** (`kSaveInvRms`): 4 bytes per row to skip a recompute.
- **Ragged rows** use padded views and masked stores, never a scalar tail loop.
- **Merge dims to 1-D** with a merge transform so N-D elementwise ops run as one sweep.

Sources: [ck_tile reduce](https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/reduce) ·
[ck_tile rmsnorm2d](https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/rmsnorm2d) ·
[ck_tile add_rmsnorm2d_rdquant](https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/add_rmsnorm2d_rdquant) ·
[ck_tile softmax](https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/softmax)
Paths drift; verify before use.

---
Related: [tile-distribution](tile-distribution.md) · [algorithmic-mapping](../../kernel-optimization/levers/algorithmic-mapping.md)
