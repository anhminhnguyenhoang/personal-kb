---
name: moe-dispatch-on-rocm
description: The sglang MoE call chain on ROCm (router GEMM, top-k, expert runner) and how the all-to-all backend chooses between the TP and EP aiter paths.
source: https://github.com/sgl-project/sglang (python/sglang/srt/layers/moe/), https://github.com/ROCm/aiter (aiter/fused_moe.py)
---

# sglang MoE dispatch on ROCm

## Call chain of one sparse MoE layer

```
hidden ─► router GEMM ─► top-k ─► experts(hidden, topk) ─► (× routed_scale?) ─► + shared ─► all-reduce
           │               │          │
           │               │          └─ TP path  : FusedMoE → registered fused func ("none","aiter")
           │               │             EP path  : DeepEP/Mori MoE → forward_aiter
           │               └─ aiter biased_grouped_topk / fused_topk   (else Triton/torch topk)
           └─ aiter tuned GEMM (tgemm.mm) when _use_aiter             (else F.linear / vendor GEMM)
```

Both expert paths end in **one** call to `aiter.fused_moe.fused_moe`. That call fuses GEMM0 → activation → GEMM1 → weighted combine. The `quant_type` enum chooses the kernel family: `No`, `per_Token`, `per_128x128` (fp8 block scale), and `per_1x32` (MX fp4).

## Decision tree at call time

1. Read the all-to-all backend from server args (`moe_a2a_backend`).
2. If it is `none`, sglang takes the TP path. `FusedMoE.forward_impl` calls dispatch, then `quant_method.apply`, then combine, then the optional all-reduce. On HIP with aiter, `apply` resolves to the fused function registered for the `("none", "aiter")` pair.
3. Otherwise (`deepep`, `mori`, …), sglang takes the EP path. `run_moe_core` checks `_use_aiter` **first** and calls `forward_aiter`. The Mori EP class asserts that aiter is on.
4. If `SGLANG_USE_AITER` is off, all MoE goes to the Triton fused-MoE runner.

## Reusable adapter tricks

- **Invalid expert ids.** DeepEP marks dropped slots with `topk_id == -1`. The aiter kernel does not accept `-1`. The adapter maps `-1` to `num_local_experts`, which is one past the last real expert, and passes an `expert_mask`. Use this pattern whenever an EP dispatcher and an expert kernel disagree on how to mark a sentinel.
- **Scale folded into top-k.** On the aiter path, the aiter top-k applies `routed_scaling_factor` itself, so the model skips the explicit multiply. If you swap one side (top-k or experts) for a custom kernel, you can apply the scale twice or not at all. Check this first when MoE output is off by a constant factor.
- **Activation mapping.** The adapter maps sglang activation strings to aiter `ActivationType`. An unknown string falls back to a default (Gelu in one version). A typo can therefore run silently with the wrong activation.
- **ASM "tkw1" variant.** A separate custom op wraps aiter's `asm_moe_tkw1`. It is used only for per-channel fp8 w8a8 with the router weight applied on the input. Check those three conditions before you expect it in a trace.

## Router GEMM

On CUDA, small-batch DeepSeek-style router GEMMs go to a dedicated kernel when the shape matches. On HIP with aiter, the router goes to aiter's tuned GEMM dispatcher. Its kernel choice depends on aiter's tuned-shape CSVs. An untuned shape gets the default solution, which can be slow.

Sources: [sglang moe layers](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/moe), [ROCm/aiter fused_moe](https://github.com/ROCm/aiter), [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP).
Paths drift; verify before use.

---
Related: [rocm-dispatch-and-gates](rocm-dispatch-and-gates.md) · [vllm moe-and-quant-selection](../vllm/moe-and-quant-selection.md)
