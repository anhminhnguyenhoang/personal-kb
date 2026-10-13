---
name: build-codegen-and-debug
description: "Working with CK C++ in practice: scoping the build (targets, dtypes, instance groups), instance library vs Python codegen and trait pruning, profiler sweeps and why pinned instances drift, the runtime dispatcher, and a symptom → cause table for wrong, missing or slow CK kernels."
source: https://github.com/ROCm/composable_kernel ; https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html ; https://github.com/ROCm/composable_kernel/blob/develop/CHANGELOG.md
---

# CK build, codegen, tuning loop, debugging

## 1. How kernels get into a binary

```
 classic GEMM / conv          CK-Tile FMHA / MoE / GEMM
 hand-listed instance .cpp    codegen/generate.py  (tile × pipeline × epilogue × traits)
 (each = tuple of 8–12        validation prunes illegal combos
  tile variants)              emits one .cpp per surviving trait
          └──────────────┬──────────────┘
           compiled per GPU target, filtered by dtype
                         │
      instance factory / registry  ──►  profiler or dispatcher picks one at runtime
```

- **Most decisions are compile-time.** Target, dtype family (FNUZ fp8, OCP fp8, MX) and
  pipeline are fixed at configure/codegen; runtime only selects among survivors.
- **Scope the build**: set `GPU_TARGETS` to your arch, filter `DTYPES`, and build only the
  instance group or example you need. A full build is hours and tens of GB of RAM per
  compile job; limit parallel jobs.
- **Prune generator traits** (head dims, dtypes, masks, paged/non-paged) to what you serve.
  Uncapped FMHA codegen emits hundreds of files.
- New-arch types (fp4, MX) only appear if their dtype gate is enabled at configure time.

## 2. Tuning loop

1. Enumerate instances for the op signature.
2. Drop those where `IsSupportedArgument` is false.
3. Time the rest at the **exact** production shape (warmup, repeats, rotating buffers so
   the next run is not L2-hot).
4. Re-time the winner with more iterations; record it per shape.

- `ckProfiler` does steps 1–4 for classic ops and reports TFLOP/s and GB/s. CK-Tile
  examples carry their own bench + `-v=1` reference check.
- **A pinned winner is build-specific.** Instance lists and pipeline code change between
  releases. Re-sweep after every CK/ROCm bump; never ship a copied table as portable.
- The CK-Tile dispatcher holds a registry with priorities and a pluggable heuristic
  (`Problem → ranked kernel ids`), plus an arch filter. A heuristic is just a function, so
  rule tables or learned selectors both fit.

## 3. Symptom → cause

| Symptom | Likely cause | Check / fix |
|---|---|---|
| Output wrong, no error | instance run past `IsSupportedArgument == false` | always gate; use a padding spec |
| Wrong by exactly 2× (fp8) | FNUZ vs OCP dialect mismatch | match quantizer to target ([epilogues-and-quant-gemm](epilogues-and-quant-gemm.md)) |
| Wrong only with MX / block scale | scale tensor not in Policy layout | let the Policy own the scale distribution; check with the matrix calculator |
| "does not support this problem" at runtime | no compiled instance covers the shape | add instance / pad dim / route elsewhere |
| Instance or type missing | arch or dtype not enabled at configure | fix `GPU_TARGETS` / `DTYPES` |
| CK-Tile dense GEMM slower than classic | front-end gap on square GEMM | benchmark both ([architecture-classic-vs-tile](architecture-classic-vs-tile.md)) |
| Slow, MFMA-bound | tile too large → AGPR/VGPR spills; or 32×32 MFMA clock loss | read ISA for `scratch_` and `v_accvgpr` moves; try 16×16 |
| Slow, memory-bound | load vector < 128 bit; Intrawave on skinny M | fix `AK1/BK1` and alignment; try Interwave / split-K |
| Pinned config regressed after upgrade | instance drift | re-sweep |

ISA check for any CK GEMM hot loop: `buffer_load_dwordx4`, `ds_read_b128`, dense
`v_mfma_*`, an `s_waitcnt lgkmcnt(N)` just before dependent MFMAs, no scratch.

## 4. Correctness gates

- Compare against an fp32-accumulate reference; CK examples ship one (`-v=1`).
- bf16/fp16 attention: relative error ~1e-2 / ~1e-3 at the output is normal.
- fp8 / MX: gate on task accuracy, not elementwise parity.

Sources: [CK repo](https://github.com/ROCm/composable_kernel) ·
[CK CHANGELOG](https://github.com/ROCm/composable_kernel/blob/develop/CHANGELOG.md) ·
[Optimizing with CK (ROCm how-to)](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html) ·
[01_fmha codegen](https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/01_fmha) ·
[LLVM issue #131954 (large MFMA tiles spill)](https://github.com/llvm/llvm-project/issues/131954) ·
[MI300X workload optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)
Paths drift; verify before use. Verify specifics against the primary source.

---
Related: [architecture-classic-vs-tile](architecture-classic-vs-tile.md) · [measurement](../../kernel-optimization/method/measurement.md) · [failure-modes](../../kernel-optimization/failure-modes/failure-modes.md)
