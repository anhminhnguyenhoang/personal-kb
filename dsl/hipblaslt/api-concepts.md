---
name: api-concepts
description: The hipBLASLt mental model (descriptors, heuristic query, matmul), epilogue bit flags, FP8/MX scale modes, and the C++ ext API with per-call split-K and WGM tuning.
source: https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt (library/include/hipblaslt/hipblaslt.h, hipblaslt-ext.hpp), https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/
---

# hipBLASLt API concepts

hipBLASLt is AMD's "lightweight" GEMM library. Its API follows cuBLASLt. A precompiled TensileLite kernel library backs it. See [tensilelite-kernels-and-selection](tensilelite-kernels-and-selection.md).

```
D = Epilogue( alpha · op(A)·op(B) + beta · C  [+ bias] )      C and D: same type and shape
```

## Lifecycle

```
Create handle ─► MatmulDesc(computeType, scaleType) ─► MatrixLayout ×4 (A,B,C,D) ─► Preference(max workspace)
      │                set TRANSA/B, EPILOGUE, BIAS, SCALE ptrs         set BATCH_COUNT / STRIDE
      ▼
AlgoGetHeuristic(desc, layouts, pref, N) ─► results[] (sorted fastest-first by estimate)
      ▼
Matmul(handle, desc, α, A, B, β, C, D, &results[i].algo, workspace, wsSize, stream)
      ▼
Destroy in reverse order
```

- **Handle.** Use one handle per stream. The handle stores the current device when you create it. `hipblasLtDestroy` synchronizes the device, so do not create and destroy handles in a hot loop.
- **Shape inference.** `m` and `n` come from **D**. `k` comes from A and `opA`. The library does not fully check that the four layouts agree. `ld < rows` is not rejected and gives memory corruption.
- **Heuristic results.** Each result holds an opaque `algo` (the solution index is inside it), `workspaceSize`, `state`, and `wavesCount`. `wavesCount` estimates how many waves of workgroups the problem fills: 1.0 means one full wave over the GPU.
- **Workspace.** The default `MAX_WORKSPACE_BYTES` is 0. That removes every solution that needs workspace, which includes most split-K and Stream-K kernels. Set a real limit, for example the bench default of 128 MB. Align the buffer to 16 bytes.
- **`algo = NULL`** in `hipblasLtMatmul` makes the library run its default heuristic itself.

## Epilogue flags

The epilogue values are bit flags. You can add them together:

```
RELU=2  BIAS=4  BGRAD=16  GELU=32  AUX=128      e.g. GELU_BIAS = 32+4 = 36,  RELU_AUX = 2+128 = 130
backward: DRELU=136  DGELU=192  (+16 → *_BGRAD)      ext: SWISH, CLAMP (act args 0/1 = min/max), SIGMOID
```

- `*_AUX` (forward) **writes** the pre-activation value to the aux tensor E. `D*` (backward) **reads** it. You must set the E pointer, `ld`, and batch stride.
- The bias vector has length = rows of D and stride 1. You can query the heuristic before you set the bias pointer, but you must set it before `Matmul`.

## FP8 / MX scaling

| Scale mode | Meaning |
| --- | --- |
| `SCALAR_32F` | one fp32 per tensor (default for fp8) |
| `OUTER_VEC_32F` | per-row of A (length M) and per-column of B (length N): `D[i,j] *= sA[i]·sB[j]` |
| `VEC32_UE8M0` | MX block scale: one E8M0 exponent per 32 elements along K |
| `BLK32_UE8M0_32_8_EXT` | the same block scale, pre-swizzled into the layout the kernel reads |

- The scale modes for A and B must be the same kind: both scalar, both vector, or both block.
- `AMAX_D_POINTER` returns max|D|, used for delayed or dynamic scaling.
- gfx942 uses the FNUZ FP8 types (`HIP_R_8F_E4M3_FNUZ`). gfx950 uses OCP (`HIP_R_8F_E4M3`). Your API code must pass the type that matches the GPU.

## C++ ext API (`hipblaslt_ext`)

```cpp
hipblaslt_ext::Gemm g(handle, opA, opB, tA, tB, tC, tD, computeType);
g.setProblem(m, n, k, batch, epilogue, inputs);   // GemmEpilogue + GemmInputs
g.algoGetHeuristic(N, pref, results);
hipblaslt_ext::GemmTuning t; t.setSplitK(4); t.setWgm(8);   // 0 = use the solution default
g.initialize(results[0].algo, t, workspace);
g.run(stream);
```

- `GroupedGemm` runs many problems in one launch. They share op and type configuration, and each problem has its own M/N/K. That fits MoE. The device-side `UserArguments` array lets a GPU kernel write the problem list without a host round trip.
- `getAllAlgos(...)` lists every solution for a type configuration. Filter it with `isAlgoSupported` for your problem. Use it for exhaustive tuning.
- `getIndexFromAlgo` / `getAlgosFromIndex` save a solution as an integer. **The index is valid only for the same library build.**

Sources: [hipBLASLt API reference](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/api-reference.html), [hipBLASLt headers](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt/library/include/hipblaslt), [OCP MX spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
Enum values are from the public headers. Verify specifics against the primary source.

---
Related: [tensilelite-kernels-and-selection](tensilelite-kernels-and-selection.md) · [benchmarking-and-tuning](benchmarking-and-tuning.md) · [pitfalls](pitfalls.md)
