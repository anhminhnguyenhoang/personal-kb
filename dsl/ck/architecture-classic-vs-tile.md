---
name: architecture-classic-vs-tile
description: How Composable Kernel C++ is built — the classic device-op stack (Device→Grid→Block→Warp→Thread, five-call lifecycle, instance library) versus CK-Tile (Kernel = Partitioner + Pipeline + Epilogue, Problem/Policy split) — and how to pick between them.
source: https://github.com/ROCm/composable_kernel ; https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/ ; https://github.com/ROCm/composable_kernel/issues/1727
---

# CK architecture: classic device ops vs CK-Tile

Composable Kernel (CK) is AMD's C++ template library for GPU kernels.
It has two front-ends in one repo. Both sit on the same idea:
describe data movement as a chain of compile-time coordinate transforms,
so all index math folds into the load/store address.

```
            include/ck  (classic)                 include/ck_tile  (CK-Tile)
 ┌───────────────────────────────┐     ┌──────────────────────────────────────┐
 │ DeviceOp   (host API, gate)   │     │ Kernel<Partitioner, Pipeline, Epi>   │
 │  └ GridwiseGemm  (one kernel) │     │   Partitioner : (M,N,K) → grid       │
 │     └ BlockwiseGemm pipeline  │     │   Pipeline    : K-loop schedule      │
 │        └ XdlopsGemm (MFMA)    │     │     Problem = shapes + dtypes        │
 │           └ Threadwise copy   │     │     Policy  = layouts + WarpGemm     │
 │ descriptors written by hand   │     │   Epilogue    : write-back + fusion  │
 └───────────────────────────────┘     │ tile_window / tile_distribution      │
                                       └──────────────────────────────────────┘
```

## Classic CK

- **Six levels**: Device → Grid → Pipeline → Block → Warp → Thread. Each level is a
  struct whose `Run()` calls the level below.
- **One host lifecycle for every op family** (GEMM, batched, grouped, multi-D, conv,
  reduce):

```
op   = DeviceOp{}                      // one compiled instance (tile, pipeline, dtypes)
arg  = op.MakeArgument(ptrs, M,N,K, strides, KBatch, elementwise ops)
if (!op.IsSupportedArgument(arg)) skip // capability gate
inv  = op.MakeInvoker()
ms   = inv.Run(arg, StreamConfig{stream, time_kernel})
```

- **`IsSupportedArgument` is a correctness gate, not a hint.** It checks divisibility of
  M/N/K by the tile, K vs `KPerBlock × KBatch`, pointer alignment vs the load vector, and
  layout. An instance run past `false` writes wrong data without an error. Use a padding
  specialization (`GemmSpecialization::MNKPadding`) for odd shapes.
- **Tuning = enumerate + filter + time.** An instance factory returns every compiled
  instance for a signature; the profiler (`ckProfiler`) drops unsupported ones and sorts
  by time. There is no JIT. Callers cache the winner per shape.
- **Elementwise ops are stateless functors** (`AElementwiseOperation`, …) inlined at
  instantiation, so bias/activation/scale cost no extra pass.
- **Layout shorthand**: `R`/`C` per operand. RCR (A row, B col, C row) is `Y = X·Wᵀ`,
  the linear layer, and the most-tuned layout.

## CK-Tile

- **Five nouns**: `tensor_view`, `tile_distribution`, `tile_window`, distributed tensor
  (registers), and Pipeline/Policy/Epilogue. See [tile-distribution](tile-distribution.md).
- **Problem vs Policy split.** The Problem type carries shapes, dtypes and traits. The
  Policy type answers layout questions: DRAM load distribution, LDS descriptor (with
  swizzle), which WarpGemm (MFMA) to use. Swap the Policy to retune without touching the
  pipeline body.
- **Golden rule**: making a view or window only declares addresses. Loads and stores
  happen inside the pipeline and epilogue. A window is a cursor, not a copy.
- **Kernel families are codegen-driven.** FMHA, fused-MoE and many GEMM variants come from
  Python generators that emit one `.cpp` per trait combination
  ([build-codegen-and-debug](build-codegen-and-debug.md)).
- **Same warp-GEMM core everywhere.** Quant GEMM, MoE, flatmm, Stream-K and FMHA all call
  one WarpGemm layer; scale and routing logic wrap around it.

## Which front-end

| Workload | Start with | Why |
|---|---|---|
| Dense bf16/fp16 square GEMM | Benchmark **both**; classic `DeviceGemm_Xdl_CShuffleV3` / universal v3 was the reference | Issue #1727: ck_tile 359 vs classic 615 TFLOP/s at 4096³, same 256×256×64 tile; a later fix (K tile 64) reached ~80 % of classic |
| Attention (prefill, decode, paged, bwd) | CK-Tile FMHA | Classic batched softmax-GEMM is legacy; the shipped FMHA library is generated from the CK-Tile example |
| Fused MoE, block-scale / MX GEMM, small-M flatmm | CK-Tile | New features land here first |
| Fusion the library cannot express | CK-Tile | Pipeline + Policy + Epilogue compose |

Rules:
- The gap between front-ends is version-dependent. Re-measure after each CK bump.
- If a library (hipBLASLt, aiter) already dispatches a good CK instance for your shape,
  tune its selection first. Author templates only for missing shapes or fusions.
- Standalone `ROCm/composable_kernel` is a read-only mirror. Development moved to
  `ROCm/rocm-libraries` (`projects/composablekernel`). Paths inside are the same.

The Python DSL in the same monorepo is a different tool: see [CK DSL](../ck-dsl/README.md).

Sources: [CK repo](https://github.com/ROCm/composable_kernel) ·
[rocm-libraries/composablekernel](https://github.com/ROCm/rocm-libraries/tree/develop/projects/composablekernel) ·
[CK-Tile concept docs](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/) ·
[Optimizing with CK (ROCm how-to)](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html) ·
[Issue #1727](https://github.com/ROCm/composable_kernel/issues/1727) ·
[Hands-on CK-Tile GEMM blog (source)](https://github.com/ROCm/rocm-blogs/blob/release/blogs/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.md)
Paths drift; verify before use. Verify specifics against the primary source.

---
Related: [tile-distribution](tile-distribution.md) · [gemm-pipelines-and-schedulers](gemm-pipelines-and-schedulers.md) · [build-codegen-and-debug](build-codegen-and-debug.md)
