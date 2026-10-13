---
name: operator-catalog
description: "aiter operators by family (GEMM, MoE pipeline, attention, norm/quant/activation) with the entry point, the backends behind it, and the contract each stage expects from the one before."
source: https://github.com/ROCm/aiter ; https://github.com/ROCm/rocm-blogs/blob/release/blogs/software-tools-optimization/aiter-mla/README.md ; https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang ; https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
---

# Operator catalog by family

Entry points are the Python names under `aiter.ops.*` (or the package root for
`tuned_gemm`, `fused_moe`, `mla`). Most ops have several backends with different shape
envelopes. Pick the family entry, then let the dispatcher pick the kernel.

## GEMM / linear

| Op | Entry | Backends | Notes |
|---|---|---|---|
| Dense bf16/fp16 | `tuned_gemm.gemm_a16w16` (`tgemm.mm`) | hipblaslt, asm, skinny, triton, flydsl, opus, torch | The live serving path; see [tuned-config-dispatch](tuned-config-dispatch.md) |
| Decode GEMV | (selected by the same dispatcher) | skinny `wvSpltK` | Small M. Bandwidth-bound, splits K over all CUs. Not usable with preshuffled B |
| fp8 / int8 | `gemm_a8w8_ck`, `gemm_a8w8_asm`, `*_blockscale_*`, `flatmm_a8w8_blockscale_asm` | CK, asm, CK-tile | Per-token / per-tensor / block (1x128, 128x128) scales |
| fp4 (MXFP4) | `gemm_a4w4`, `gemm_a4w4_asm` | asm, CK | e8m0 scale per 1x32 block. Needs gfx950 hardware |
| Batched | `batched_gemm_bf16`, `batched_gemm_a8w8` | CK | Own tuned CSVs |
| opus split-K | `gemm_a16w16_opus` (or via the dispatcher) | opus | fp32 partials in a per-stream workspace, then a reduce kernel that also adds bias |
| Weight prep | `shuffle_weight`, `shuffle_weight_NK`, `shuffle_scale` | host permute | Rewrites B into MFMA lane order; sets `is_shuffled` |

**Preshuffle ("bpreshuffle")** permutes `[N, K]` weights offline so each lane's MFMA
operand is contiguous. The kernel then loads it straight into registers. The default
`layout=(16, 16)` assumes a 16x16 MFMA tile. The tag says only *that* a tensor was
shuffled, not *which* layout. Block scales (e8m0) must be shuffled to match.

## MoE pipeline

```
 router logits [T, E]
   │  topk_softmax / topk_sigmoid / biased_grouped_topk / moe_fused_gate
   ▼
 topk_ids [T, k], topk_weights [T, k] (fp32)
   │  (EP only) MoRI dispatch: tokens → ranks owning their experts
   ▼
 moe_sorting  (superset of moe_align_block_size)
   │  sorted_token_ids  (expert-contiguous; each expert padded to block_m)
   │  sorted_expert_ids (expert id per block)   num_valid_ids   sorted_weights
   │  moe_buf zero-filled (stage 2 accumulates into it)
   ▼
 fused_moe: stage 1  [gate+up (g1u1) GEMM → activation]   often asm
            stage 2  [down GEMM × routed weight → moe_buf]  often CK
   │  (EP only) MoRI combine: weighted outputs → source ranks
   ▼
 hidden [T, H]
```

| Stage | Contract | Trap |
|---|---|---|
| Routing | Scoring function must match the model | The biased grouped path is **sigmoid-only** (DeepSeek-style gating); route softmax-gated models through `topk_softmax` or the non-biased grouped op. Group count limited to a small set (check the launcher) |
| Align/sort | `block_m` = the grouped-GEMM tile M | Pad slots hold an out-of-range sentinel; the caller pre-fills it for the align op. Ids must be in `[0, E)` (no bounds check) |
| Sort backends | CK, opus (default), FlyDSL, adaptive | Picked by env and flags; FlyDSL is skipped for several output modes |
| Stage 1/2 | Picked per shape and quant signature from `tuned_fmoe.csv` | `doweight_stage1` decides where the routed weight multiplies. Keep it consistent or it is applied twice or never |
| Quant routing | `quant_type` + `q_dtype_a/w` select the kernel family | bf16, fp8 per-token, fp8 block (1x128), int8, MXFP4 (1x32). A4W4 uses FlyDSL if installed, else CK |
| EP dispatch/combine | aiter delegates to MoRI (`MoriAll2AllManager`) | See [MoRI EP dispatch/combine](../mori/ep-dispatch-combine.md) |

Two general facts behind MoE perf:
- Weight bytes moved scale with **activated experts** = `min(T·k, E)` per rank, not with T.
  Past the point where every expert is active, more tokens add FLOPs but no weight bytes.
- Align/sort is a small, memory-bound kernel. A multi-block rewrite that sizes the grid to
  the chiplet count (and keeps small grids on one die) gave a large speedup on MI300X.

## Attention

| Op | Entry | Notes |
|---|---|---|
| MHA prefill | `flash_attn_func`, `mha_fwd`, `fmha_v3_fwd`, `mha_batch_prefill` | asm/CK with a Triton fallback. GQA-native: pass KV at `num_kv_heads`, never `repeat_kv` first |
| Paged decode | `paged_attention_v1`, `paged_attention_ragged`, `pa_fwd_asm` | Split-KV partitions. Supported head sizes and GQA ratios are a fixed list; outside it falls back to Triton |
| MLA decode | `mla.mla_decode_fwd` | Matrix absorption turns MLA into MQA over the latent (1 KV head); asm kernel. `num_kv_splits=None` lets a heuristic pick split-KV from batch and CU count |
| MLA prefill | `mla_prefill_fwd`, `mla_prefill_ps_fwd` (persistent) | |
| Sparse MLA | `mla_decode_fwd_v4_nm`, Triton/Gluon sparse paths | Arch-specific; check which path ran. Needs an attention-sink tensor |
| Sparse indexer | `deepgemm_fp8_paged_mqa_logits` | Scores KV tokens for top-k selection; not the same op as the `deepgemm` grouped GEMM |
| KV store | `reshape_and_cache*`, `concat_and_cache_mla`, `fused_qk_norm_rope_cache_quant_*` | Store layout, fp8 dialect, and scale granularity must match the attention read |

## Norm, quant, activation, misc

All of these are bandwidth-bound. The lever is **fusion**, not tiling: on the serving
path you call the fused entry, never the plain one.

| Family | Fused entry to prefer | Note |
|---|---|---|
| RMSNorm | `rmsnorm2d_fwd_with_add`, `*_with_dynamicquant`, `add_rmsnorm_quant` | HIP fast path for 2-D bf16/fp16 up to a hidden-size cap, a generic kernel otherwise. `fused_add_rms_norm_cu` writes both inputs in place |
| LayerNorm | `layernorm2d_fwd_with_add`, `*_with_smoothquant` | |
| fp8 quant | `per_token_quant_hip`, `per_tensor_quant_hip`, `per_group_quant_hip` | `scale = amax / dtype_max`. The torch dtype picks FNUZ (gfx942) vs OCP |
| MXFP4/MXFP8 | `per_1x32_f4_quant`, `per_1x32_f8_scale_f8_quant` | `pack_dim=-1` for activations, `0` for weights. Scale shuffle must match the consumer |
| Gated act | `silu_and_mul`, `gelu_and_mul`, `gelu_tanh_and_mul` | Input is packed `[..., 2, d]`. Usually fused into MoE stage 1 or a GEMM epilogue |
| RoPE | `rope_*_fwd_impl` (cached, positions, 2-channel, thd) | The fused QK-norm + RoPE + KV-store + quant entry removes a Q/K round-trip |
| Sampling | `top_k_top_p_sampling_from_probs` and friends | Sort-free rejection sampling; fp32 probs |
| TP collectives | custom all-reduce, quick (quantized) all-reduce | Intra-node, bypasses RCCL. One-shot for small payloads, two-shot for large; fused residual + RMSNorm (+ quant) variant |

Sources: [ROCm/aiter](https://github.com/ROCm/aiter) (`aiter/ops/`, `aiter/fused_moe.py`, `aiter/mla.py`, `aiter/ops/shuffle.py`) ·
[AITER MLA decode blog](https://github.com/ROCm/rocm-blogs/blob/release/blogs/software-tools-optimization/aiter-mla/README.md) ·
[`mla_decode_fwd` notebook](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.html) ·
[MoE align/sort design (SGLang)](https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang) ·
[Wide-EP DeepSeek on MI300X](https://github.com/ROCm/rocm-blogs/blob/release/blogs/software-tools-optimization/wide-ep-deepseek/README.md) ·
[grouped top-k kernel (biased path is sigmoid-only)](https://github.com/ROCm/aiter/blob/main/csrc/kernels/topk_softmax_kernels_group.cu)
Paths drift; verify before use.

---
Related: [architecture](architecture.md) · [pitfalls](pitfalls.md) · [MoRI EP dispatch/combine](../mori/ep-dispatch-combine.md)
