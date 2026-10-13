---
name: moe-and-quant-selection
description: How vLLM on ROCm chooses between Triton fused-MoE, aiter fused-MoE, and aiter ASM MoE, how FP8 linear layers reach aiter, and where scale-layout bugs hide.
source: https://github.com/vllm-project/vllm (vllm/model_executor/layers/fused_moe/, vllm/model_executor/layers/quantization/, vllm/_aiter_ops.py)
---

# vLLM MoE and quant kernel selection on ROCm

## MoE path choice

```
FusedMoE.__init__:
  rocm_aiter_fmoe_enabled = is_fused_moe_enabled()          # AITER master AND AITER_MOE
                            AND is_act_and_mul               # gated activation only
        │
   ┌────┴──────────────────────────────┬──────────────────────────────┐
   no                                  yes                            special schemes
   ▼                                   ▼                              ▼
 Triton fused_experts            torch.ops.vllm.rocm_aiter_fused_moe   asm_moe_tkw1 (per-channel w8a8,
 (in/out-of-place custom ops)    → aiter.fused_moe.fused_moe           router weight on input),
 + per-shape JSON configs                                              dedicated MX/w4a8 expert classes
```

- A non-gated activation (`is_act_and_mul=False`) **forces the Triton path**, because the aiter fused MoE supports only gated forms (silu, gelu, swiglu).
- A shared-expert fusion flag (`VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS`) can fold the shared expert into the aiter call. When it is on, the trace shows no separate shared-MLP kernel.
- The Triton path reads tuned configs from `fused_moe/configs/E=<E>,N=<N>,device_name=<NAME>[,dtype=...].json`. If the name or shape does not match, it uses a default config **silently**. Before you blame the kernel, check that a file exists for your `(E, N, device, dtype)`.
- vLLM converts its string enums to aiter `ActivationType` and `QuantType` (`no`, `per_tensor`, `per_token`, `per_1x32`, `per_1x128`, `per_128x128`) in one place, inside `_aiter_ops`.

## FP8 linear

FP8 projections in attention and MLP go through a scaled-mm kernel layer. The aiter backend is `kernels/linear/scaled_mm/aiter.py`, gated by `VLLM_ROCM_USE_AITER_LINEAR` (still ANDed with the master switch). Block-scale GEMMs have two registered variants, aiter HIP/ASM and aiter Triton, and a flag chooses between them. When aiter is off, dense GEMMs go to hipBLASLt through PyTorch. See [hipBLASLt benchmarking-and-tuning](../hipblaslt/benchmarking-and-tuning.md).

## Scale-layout trap

Scales can be reshaped in three places, and none of them validates the others:

1. **The weight loader** writes the final layout to the device once, for example a preshuffle or padding of the K tail.
2. **The quant-scheme module** decides the per-scheme layout (quark, compressed-tensors, mxfp4, …).
3. **The `_aiter_ops` wrappers** take scales as plain tensors and **assume** the layout the loader wrote.

A mismatch does not crash. It gives a quiet accuracy loss. When you swap a kernel, compare against a reference GEMM on real loaded weights, not on freshly made random tensors.

## Trace bucketing hints

| Bucket | Symbols to expect |
| --- | --- |
| MoE | `rocm_aiter_fused_moe`, Triton `fused_moe_kernel` |
| top-k | `rocm_aiter_topk_softmax`, `rocm_aiter_(biased_)grouped_topk` |
| norm + quant | aiter `rmsnorm2d_fwd_with_add[_dynamicquant]` |
| dense GEMM | hipBLASLt `Cijk_*` kernels, aiter `gemm_a8w8*` |
| all-reduce | custom AR, quick-reduce, aiter fused AR+RMSNorm |

Sources: [vllm fused_moe](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe), [vllm quantization](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization), [ROCm/aiter](https://github.com/ROCm/aiter).
Paths drift; verify before use.

---
Related: [rocm-code-map](rocm-code-map.md) · [sglang moe-dispatch-on-rocm](../sglang/moe-dispatch-on-rocm.md) · [tensilelite-kernels-and-selection](../hipblaslt/tensilelite-kernels-and-selection.md)
