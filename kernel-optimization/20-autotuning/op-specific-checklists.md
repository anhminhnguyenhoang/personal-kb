---
name: op-specific-checklists
description: per-operation checklists for GEMM, convolution, attention, reduction, and fused ops — the questions to ask before and during tuning each family.
source: CK DSL Optimization Runbook (ROCm/rocm-libraries@1efeadd, projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
---

## 13. Operation-Specific Checklists

### 13.1 GEMM Checklist

- Is layout transposed or not?
- Are operands contiguous along the vectorized dimension?
- Does tile shape match the MFMA atom shape?
- Is K loop long enough?
- Are A/B staged through LDS?
- Are loads vectorized? (the intrinsic-count probe:
  `global.load`/`buffer.load` count vs MFMA count)
- Is epilogue vectorized? (the ISA-inspection probe:
  `buffer_store_dwordx2`/`_x4` vs `buffer_store_short`)
- Is split-K needed?
- Is Stream-K needed?
- Is persistent scheduling helpful?
- Are scales / bias fused efficiently? (`FusedEpilogue`)

### 13.2 Convolution Checklist

- Direct vs implicit GEMM decision (see Section 4.2).
- `N_gemm`, `K_gemm`, reduction length.
- Channel count small or large.
- Filter size (3×3 vs 5×5 vs 7×7).
- Padding / tails.
- Row streaming.
- Circular accumulators.
- Weight residency (registers vs LDS).
- Input LDS reuse.
- Q / spatial tiling.
- Group tiling.
- Output store mapping.
- Swizzle on `(spatial, C8)`.

### 13.3 Attention Checklist

- Tile Q / K / V.
- Online softmax (`OnlineSoftmaxState`).
- Mask handling (`causal_mask`, `sliding_window_mask`).
- Page table overhead (`transforms.indirect + unmerge`).
- KV cache layout.
- Split sequence / head (2D vs 3D).
- Register pressure from accumulators.
- Shared memory capacity.
- Coalesced V/O stores.
- LSE output.
- 2D vs 3D selection heuristic (`select_2d_config`, `use_2d_kernel`).
- ALiBi / QQ-bias paths exercised (per
  `notes/ATTENTION_PARITY_REPORT.md` 2D path was failing C9/C10/C11
  until the score-block fix landed).

### 13.4 Reduction Checklist

- Axis length.
- Rows per block.
- Vector width.
- Warp-level vs block-level (LDS block reduction vs warp-only).
- Multi-pass.
- Numerical stability.
- Atomic or no atomic.
- Store width.

### 13.5 Fused Op Checklist

- Does fusion reduce memory traffic?
- Does fusion increase VGPR too much? (the occupancy probe)
- Can the epilogue be fused without scalarizing stores?
- Are scale / bias loads coalesced?
- Are activation approximations acceptable?
- Is intermediate precision visible?
- Do fusion in a dedicated fusion pipeline that legalizes the fused
  graph: check LDS budget, dtype, vector width, and atomic-region
  rules before lowering.

ck_dsl specifics: [primitive-map](../../dsl/ck-dsl/primitive-map.md#op-specific-checklists).

---
Family instance tables: [algorithmic-mapping](../10-levers/algorithmic-mapping.md). Full knob list: [knob-catalog-and-sweep](../../dsl/ck-dsl/knob-catalog-and-sweep.md). Failure modes per op: [failure-modes](../30-failure-modes/failure-modes.md).
