---
name: fmha-forward
description: "CK-Tile FlashAttention-2 forward as reusable idioms: the FA-2 → tile mapping, the qr/qs/async/whole-K/split-KV/paged/append-KV pipeline family, online-softmax edge cases (validated max, exp2 with folded scale, zero-sum rows, LSE), tile-range mask elision, sinks, fp8 descale folding, and occupancy by head dim."
source: https://github.com/ROCm/rocm-blogs/blob/release/blogs/software-tools-optimization/ck-tile-flash/README.md ; https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/fmha ; https://arxiv.org/abs/2307.08691
---

# CK-Tile FMHA forward

The CK-Tile FMHA example is the FlashAttention-2 forward that FlashAttention-ROCm's `ck`
backend and several serving stacks build from. Fwd supports head dim ≤ 256.

## 1. FA-2 → CK-Tile

```
 per workgroup: one Q tile [kM0 × d]  (kept in registers for the whole loop)
 for each KV tile j:                                  CK-Tile mechanism
   S  = Q·K_jᵀ            (gemm0, K via LDS)          BlockGemm pipeline
   m' = max(m, rowmax S)                              block_tile_reduce(max) + _sync
   P  = exp(S − m')                                   sweep_tile lambda
   c  = exp(m − m');  l = c·l + rowsum P              block_tile_reduce(sum)
   O  = c·O + P·V_j        (gemm1, V via LDS)         rescale O BEFORE adding PV
 O /= l ; LSE = m + log l ; store                     epilogue
```
Tile names: `kM0` (Q rows), `kN0` (KV rows per step), `kK0` (QK contraction slice),
`kN1` (O columns), `kK1` (PV contraction slice). The blog's example: 128/128/32/128/32.
Softmax state is fp32; P is cast to the input type for gemm1.

## 2. Pipeline family (`ops/fmha/pipeline`)

| Pipeline | Q | K/V | Use |
|---|---|---|---|
| `qr_ks_vs` | registers | streamed through LDS | general prefill |
| `qs_ks_vs` | re-read per K tile | LDS | lower VGPR pressure |
| `qr_ks_vs_async` | registers | async global→LDS, double buffer | latency-hidden prefill |
| `*_whole_k_prefetch` | registers | all of K loaded up front | small head dim |
| `*_trload` | | transpose-on-LDS-read for V | targets with `ds_read_tr*` |
| `fwd_splitkv` + `splitkv_combine` | | KV range split across WGs | long context / decode |
| `fwd_pagedkv`, `batch_prefill` | | KV through a page table | paged KV cache |
| `fwd_appendkv` | | writes new K/V (+rotary) into the cache | no softmax |
| `qr_ks_vs_fp8` | | fp8 inputs | quantized attention |

Pipelines are swapped by trait; the kernel shell is the same.

## 3. Online-softmax edge cases (copy these)

- **Validated max**: if a row's running max is still `−inf` (all masked so far), use 0 in
  `exp(s − m)`. Prevents `exp(−inf − (−inf)) = NaN`.
- **exp2 fast path**: fold `log2(e)·scale` into one multiplier and compute
  `exp2(scale·s − m)`. LSE is then `m / log2(e) + log(l)`.
- **Zero-sum rows**: at the end, if `l == 0` write zeros, not `0/0`.
- **Fully masked rows** emit `LSE = −inf`.
- **Rescale-then-add**: multiply O by `exp(m_old − m)` before adding the new `P·V`.
- **Logit soft-cap** and other score transforms run as a variant hook on S before the max.
- **Attention sinks**: initialize `m` and `l` from the sink logit as a virtual prefix token.
  No branch in the inner loop.

## 4. Masks

- **Tile-range elision**: per Q tile compute `[kv_start, kv_end)` from the mask (causal,
  sliding window, local). Fully masked KV tiles are never loaded.
- **Per-element mask only on boundary tiles**: `set_tile_if(s, -inf, pred)`. Interior tiles
  run mask-free.
- **ALiBi / bias** is added to S during or right after gemm0; a `sched_barrier(0)` keeps the
  bias load from being scheduled into the GEMM.
- Under a causal mask, work per Q tile grows with its index. The block-sparse variant maps
  block `i` to Q tile `n − 1 − i` so the heaviest tiles start first (tail balance).
- **Block-sparse attention**: iterate a precomputed list of valid KV blocks per Q tile
  instead of the dense range; no per-element check.

## 5. Split-KV and paged KV

```
 split i:  (O_i, LSE_i) over KV range i           combine:
                                                   L   = max_i LSE_i
                                                   LSE = L + log Σ exp(LSE_i − L)
                                                   O   = Σ exp(LSE_i − LSE) · O_i
```
- Uneven last split is handled by masking, not a separate kernel.
- **Paged KV at window level**: a page navigator maps a logical KV tile origin to a
  physical page and swaps the buffer pointer when the window crosses a page boundary. The
  rest of the pipeline is unchanged from contiguous KV.

## 6. Quantized attention

- **Per-tensor fp8**: fold `q_descale · k_descale` into the softmax scale once. No
  per-iteration multiply.
- **Per-block KV scales**: index the scale by `kv_position / block_size`, shared by K and V.
- **fp8 P for gemm1**: compute `P = exp2(s − (m − k))` so the row max of P is `2^k`
  instead of 1 and P uses the fp8 range (k = 8 for OCP e4m3, 7 for FNUZ, whose max is
  smaller). Undo the shift in LSE.
- **MX P for gemm1** (CDNA4): cast P to fp4 + E8M0 scale per lane group and feed the scaled
  MFMA.

## 7. Occupancy and tuning

- `kBlockPerCu` is chosen by head dim: small d → 2–3 WGs/CU, large d → 1 WG/CU to avoid
  spills. `kM0` is the main occupancy vs reuse knob.
- Grid = `batch × heads × ceil(sq / kM0)`. Few heads and short sequences starve the GPU;
  split-KV helps even in prefill then.
- Specialize causal and non-causal separately; prune the generator's head-dim, dtype and
  mask lists to the shapes you serve.
- Cross-backend outputs differ in the last bits (KV reduction order). Gate on tolerance and
  task parity, not bit equality.

Sources: [CK-Tile FlashAttention-v2 blog (source)](https://github.com/ROCm/rocm-blogs/blob/release/blogs/software-tools-optimization/ck-tile-flash/README.md) ·
[ck_tile/ops/fmha](https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/fmha) ·
[qr_ks_vs pipeline](https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp) ·
[01_fmha example + codegen](https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/01_fmha) ·
[FlashAttention-2 (arXiv 2307.08691)](https://arxiv.org/abs/2307.08691) ·
[flash-attention (ROCm ck backend)](https://github.com/Dao-AILab/flash-attention)
Paths drift; verify before use. Verify specifics against the primary source.

---
Related: [fmha-backward](fmha-backward.md) · [tile-distribution](tile-distribution.md) · [conditional-softmax-rescaling](../../papers/flashattention-4/02-algorithm/conditional-softmax-rescaling.md)
