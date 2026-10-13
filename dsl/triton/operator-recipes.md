---
name: operator-recipes
description: General AMD Triton recipes for four operator families (prefill attention, fused MoE grouped GEMM, fp8/MXFP4 quant, RMSNorm), drawn from public aiter, vLLM, and upstream Triton kernels.
source: https://github.com/ROCm/aiter/tree/main/aiter/ops/triton ; https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py ; https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# Triton on AMD: operator recipes

Four families. Each section: shape of the kernel, the knobs that matter, the traps.

```
                 bound by          num_stages   num_warps   key lever
 attention fwd   MFMA + softmax    1            4 (8 big d)  BLOCK_M, GQA packing
 fused MoE       MFMA / launch     1-2          4            per-shape table, BLOCK_M = align block
 quant fp8/fp4   HBM               1-2          1-4          fuse into producer
 RMSNorm         HBM               1-2          2-4          128-bit loads, persistent grid
```

## 1. Prefill attention (FlashAttention forward)

Two chained dots around an online softmax: `S = Q·Kᵀ`, then `O += P·V`. Keep running max and sum
in registers. Never write S to memory.

- `num_stages=1`. AMD docs: two fused GEMMs → 1. Two dots plus softmax state already fill LDS and VGPRs.
- `BLOCK_M` (query rows) is the main occupancy lever: O accumulator + stats scale with it.
  128 on CDNA for long prefill; lower for big head dims.
- **GQA packing.** Put the query heads that share one KV head into the M tile:
  `BLOCK_M = max(16, next_pow2(q_per_kv))`, `BLOCK_Q = BLOCK_M // q_per_kv` query positions per
  program. Each K/V tile is loaded once for all heads in the group (aiter `unified_attention`).
- **Use `exp2`.** Fold `log2(e)` into the QK scale and call `tl.math.exp2`. Softcap as
  `x * tanh(S/x)` can be built from two `exp2` calls.
- **fp8 attention:** fold `q_descale * k_descale` into the QK scale before softmax; fold `v_descale`
  into the final `1/l` division. No extra pass over the tile.
- **Causal:** specialize the masked diagonal tiles; skip fully masked KV tiles. Do not mask every tile.
- Grid: `batch × heads × ceil(sq / BLOCK_M)`; target ≥ 1024 workgroups. Short sequences with few
  heads starve the GPU: split KV (flash-decoding style) helps even in prefill.

## 2. Fused MoE grouped GEMM

```
 router topk ─► align & sort (pad each expert's tokens to BLOCK_M)
             ─► sorted_token_ids, expert_ids per M-block
             ─► one Triton GEMM: grid (num_M_blocks, N/BLOCK_N)
                 program reads expert_ids[pid_m] -> picks weight slice
                 gathers its tokens via sorted_token_ids, masks pad rows
```

- The GEMM `BLOCK_M` must equal the align block size. They are one parameter.
- The align & sort step is on the critical path for every backend. A slow sort makes the GEMM look slow.
- Decode: few tokens per expert → most of each M-block is padding. Use a small `BLOCK_M`, or a
  masked grouped GEMM that skips empty experts.
- Ship a per-shape config table (keyed by experts, N, dtype, device), made offline.
  A missing entry falls back to a default config without failing. Watch the log for a
  "default config" warning.
- Weighted top-k combine: multiply by the routing weight in one GEMM epilogue only (gate-up or down).

## 3. Quant kernels (fp8 per-token / per-tensor, MXFP4)

- Memory-bound, no MFMA. A standalone quant kernel moves the tensor twice (read and write).
  The real win is to **fuse** quant into the producer: RMSNorm+quant, SiLU·mul+quant, MoE sort+quant.
- Per-token quant uses one program per row, `BLOCK = next_pow2(cols)`, row `amax` in fp32,
  `scale = amax / FP8_MAX`, clamp, cast.
- MXFP4: group size 32 is fixed by the format. Pick tiles by regime: few rows (decode) → 1 warp,
  1 stage; many rows → 4 warps, 2 stages, tile N in multiples of 32.
- Write scales in the layout the consumer GEMM reads (see [low-precision-dtypes](low-precision-dtypes.md)).

## 4. RMSNorm (and fused add + RMSNorm)

```python
for row in tl.range(pid, n_rows, NUM_PROGRAMS, num_stages=...):     # persistent loop
    p = tl.multiple_of(x_ptr + row * stride + cols, (16,))           # -> dwordx4
    x = tl.load(p, mask=m, other=0., cache_modifier=".cg").to(tl.float32)
    r = tl.math.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    tl.store(o_ptr + row * stride + cols, (x * r * g).to(o_ptr.dtype.element_ty), mask=m)
```

- Persistent grid: `min(rows, num_CUs)` programs that stride over rows. Saves launch and
  prologue cost when rows ≫ CUs.
- `BLOCK = next_pow2(cols)`, capped (aiter caps at 64 KB of row). Wider rows → two-pass blocked
  path: pass 1 sums `x²` block by block, pass 2 normalizes. This is for register fit, not accuracy.
- Upcast to fp32 **before** squaring. bf16 `Σx²` loses bits fast for large hidden sizes.
- `tl.multiple_of(ptr, 16)` lets the compiler emit `dwordx4`. Without it, loads can fall back to
  narrow ones (more instructions).
- `.cg` on the read-once input avoids polluting cache for data that is not reused.
- Tall-skinny case (many short rows): 2D tile with several rows per program, or lanes sit idle.
- A reduced dim < 64 wastes lanes in the wave reduce. Round up to a power of 2.
- Fused quant output: compute the normalized row in fp32, take its `amax`, write scale, cast.
  In the two-pass path keep the fp32 row (or recompute it identically) so quant matches the one-pass result.

## Cross-cutting

- Reduction order differs between backends → greedy argmax can flip on ties. Gate a backend swap
  on task accuracy, not bit-equality.
- Confirm the Triton kernel actually runs: its name must appear in the profiler's top kernels.

Verify specifics against the primary source.

Sources: [aiter Triton kernels (attention, moe, quant, normalization)](https://github.com/ROCm/aiter/tree/main/aiter/ops/triton) ·
[vLLM fused_moe.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py) ·
[ROCm workload optimization (num_stages rules, ≥1024 grid)](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html) ·
[Triton tutorial 05: layer norm](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) ·
[Triton tutorial 06: fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) ·
[FlashAttention-2 (arXiv 2307.08691)](https://arxiv.org/abs/2307.08691)
Paths drift; verify before use.

---
Related: [amd-knobs](amd-knobs.md) · [low-precision-dtypes](low-precision-dtypes.md) · [op-specific-checklists](../../kernel-optimization/autotuning/op-specific-checklists.md)
