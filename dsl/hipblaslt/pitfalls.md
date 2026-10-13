---
name: pitfalls
description: hipBLASLt traps that cause zero algorithms, silent wrong results, or slow kernels (type rules, workspace, layouts and swizzle, epilogue pointers, alpha handling, grouped GEMM, tuning portability).
source: https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt (library/src/amd_detail/rocblaslt/, clients/), https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/
---

# hipBLASLt pitfalls

## "Heuristic returned 0 algorithms"

```
0 results ─┬─ workspace limit = 0?                → set MAX_WORKSPACE_BYTES (split-K/Stream-K need it)
           ├─ type / epilogue combo not in library → it still returns SUCCESS with count 0
           ├─ C type != D type?                    → rejected (type mismatch)
           ├─ scale kinds differ for A and B?       → rejected (both scalar / vector / block)
           └─ wrong FP8 flavor for the arch?        → FNUZ on gfx942, OCP on gfx950
```

A count of 0 with status SUCCESS means "no kernel in the shipped library for this case". It does not mean the arguments are wrong. Try `getAllAlgos` for the type configuration to see if any kernel exists.

## Types

- `compute_f32_fast_xf32` (TF32) needs all of A, B, C, D in fp32. `compute_i32` needs i8 inputs and i32 or i8 outputs.
- Setting `COMPUTE_INPUT_TYPE_A/B_EXT` **changes the effective compute type** (for example to f32 with fast f8). The order in which you set attributes matters. Set the compute-input types last, and read the descriptor back if you are not sure.
- `scaleType` at descriptor creation accepts only fp32, fp64, or i32. alpha and beta are read as the **compute type** (`float*` for every fp compute, including fp16). Passing a `half*` alpha silently gives the wrong scalar.

## Layouts

- `ld < rows` is **not** checked. It gives silent corruption. Negative batch strides and unequal batch counts are checked.
- Swizzled (tiled-order) operands need `opA=T` and `opB=N`, and each dtype has a fixed order: fp16/bf16 → `COL16_4R8`, fp8 → `COL16_4R16`, fp4 → `COL16_4R32`. Any `lda`/`ldb` you pass is **ignored**, with only a warning.
- `k == 0` does not return early. D must still equal beta·C.

## Epilogues and pointers

- The `*_AUX`, `D*`, and `SIGMOID` epilogues need the E (aux) pointer. The `*BIAS*` and `*BGRAD*` epilogues need the bias pointer. A null pointer gives `invalid_pointer` at matmul time, not at heuristic time.
- If `lde` is 0 or negative, it defaults to m. A positive `lde < m` is an error.
- **Alpha-vector mode** (`ALPHA_DEVICE_VECTOR_BETA_HOST`, or setting a scale-alpha vector) replaces the scalar alpha with 1. The vector scales each row. The alpha you passed is discarded with no warning.

## Workspace and handles

- The workspace you allocate is limited to the algo's `max_workspace_bytes`, which comes from the preference at query time. A large buffer does not help if the preference was small.
- A handle stores its device at creation. If you call `hipSetDevice` later and reuse the handle, kernels can be chosen for the wrong device. `HIPBLASLT_USE_ROCROLLER` is read once, when the handle is created.

## Grouped GEMM

- op and types come from the **first** problem and apply to all of them. Groups with mixed types must be split.
- Problems of size zero are **skipped silently** (n = 0 is allowed). Indices in your result bookkeeping can then shift.
- `getAllAlgos` is preferred over the heuristic for grouped GEMM. Workspace scales with the group count.

## Tuning portability

Solution indices, override files, and caches belong to one library build and one arch. A shape that goes slow again after an upgrade usually has a stale index. The index then fails validation and the shape falls back to the heuristic.

## Testing patterns worth copying

- **Integer-exact inputs.** Use small integers, for example A, C ∈ {0, 1, 2}, B ∈ {−2..2}, and alpha 2. The GPU result must then equal the CPU result **bit for bit**. This finds wrong accumulation order and dropped tiles that tolerance checks hide.
- **Accumulator-precision probe.** Use near-max fp16 values in A and pairs of ±2 in B, arranged so the result is exact only when the kernel accumulates in fp32.
- **alpha = 0 with NaN inputs.** Check that NaN in A or B does not reach D when alpha = 0.
- Use three checks together: element-wise near-equality, relative Frobenius norm, and allclose (`|a−b| ≤ atol + rtol·|b|`). Scale the tolerance with K and the dtype.

Sources: [hipBLASLt source](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt), [hipBLASLt docs](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/).
Behavior comes from reading the source of one release. Verify specifics against the primary source.

---
Related: [api-concepts](api-concepts.md) · [benchmarking-and-tuning](benchmarking-and-tuning.md) · [establish-baselines](../../kernel-optimization/method/establish-baselines.md)
