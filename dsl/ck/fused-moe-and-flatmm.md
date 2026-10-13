---
name: fused-moe-and-flatmm
description: "CK MoE and small-M GEMM idioms: token sorting into expert-aligned blocks, the fused gate-up + down pipeline with two accumulators and an atomic-add epilogue, one-stage vs two-stage MoE, routed-weight placement, instance coverage, topk-softmax routing, and flatmm with preshuffled weights for decode."
source: https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/fused_moe ; https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/flatmm ; https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/topk_softmax
---

# CK fused MoE and flatmm

## 1. Data flow

```
 router logits ─► topk_softmax ─► (weights, expert ids) per token
                                   │
                         moe_sorting: group token slots by expert,
                         pad each expert to a multiple of block_m
                                   │
          ┌────────────────────────┴───────────────────────┐
     one-stage (fused_moegemm)                  two-stage (separate launches)
     gate-up GEMM → act → down GEMM            stage 1: gate-up (+act) → buffer
     in one kernel                             stage 2: down GEMM (grouped)
          └────────────── atomic-add / scatter to output ×topk weight ┘
```

## 2. Idioms

- **Sort, then GEMM.** Sorting makes every block read rows of a single expert, so weight
  loads coalesce and each block runs one plain GEMM. Padding per expert to `block_m` keeps
  tiles full; empty padded tiles should exit early.
- **Two accumulators** in the fused kernel: one for gate-up, a ping-pong pair for down.
  The down GEMM starts while gate-up results are being written to the LDS bridge buffer.
- **Atomic-add epilogue**: each (token, expert) slot adds its partial output into the
  token row. top-k > 1 means several adds per row; output must start zeroed.
- **Routed weight is applied once.** It can be multiplied in stage 1 (before down) or in
  stage 2 (in the epilogue). Check the flag on both stages; applying it twice is a silent
  error.
- **Quant scales live in the epilogue**: per-token A scale × per-channel or per-block B
  scale × routed weight, then cast. Smooth-quant variants pre-scale activations per
  channel.
- **Fixed occupancy target** (2 blocks/CU in the fused pipeline) balances variable expert
  load better than one large block.
- **Instance coverage is a crash risk.** Library MoE dispatch picks a compiled instance per
  shape. An expert/inter-dim combo with no supporting instance fails at runtime; it does
  not fall back. Check coverage before deploying a new model shape.

## 3. topk-softmax routing kernel

- Warp per row: argmax → record → set to `−inf` → repeat k times. No sort.
- Carry `(value, index)` packed so the warp reduce stays one operation.
- Optional persistent grid (`num_cu × n` blocks) amortizes the tail when rows are few.
- Row length must fit one warp's span; pad experts with `pad_tensor_view`.

## 4. flatmm (small M, decode)

- Batch × sequence is flattened into M, weights are **preshuffled on the host** into the
  per-lane MFMA operand order, so each lane loads B with one `dwordx4` and no LDS.
- C stays in VGPRs (no C staging in LDS); LDS goes to A double buffering.
- A per-M-iteration schedule interleaves `ds_read`, `ds_write`, `mfma`, `buffer_load`
  explicitly; A is read two stages ahead.
- Mixed-precision and MX variants: fp16/bf16 A × fp4 + E8M0 B on CDNA4.
- MoE flatmm variants reuse the same structure for per-expert decode GEMMs.

## 5. Grouped GEMM (classic)

`DeviceGroupedGemm*` takes arrays of pointers/strides/M per group. Use it when experts
have very different M and a sorted single launch is not available.

Sources: [ck_tile fused_moe](https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/fused_moe) ·
[ck_tile flatmm](https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/flatmm) ·
[ck_tile topk_softmax](https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/topk_softmax) ·
[CK CHANGELOG (flatmm, preshuffle B, MX)](https://github.com/ROCm/composable_kernel/blob/develop/CHANGELOG.md) ·
[aiter (CK MoE consumer)](https://github.com/ROCm/aiter)
Paths drift; verify before use. Verify specifics against the primary source.

---
Related: [epilogues-and-quant-gemm](epilogues-and-quant-gemm.md) · [gemm-pipelines-and-schedulers](gemm-pipelines-and-schedulers.md) · [work-decomposition](../../kernel-optimization/levers/work-decomposition.md)
