---
name: epilogues-and-quant-gemm
description: "CK epilogue toolbox (CShuffle with set/update, cast and scale inside the LDS hop, dynamic per-row quant, PermuteN, multi-D fusion) and the low-precision GEMM idioms: group-quant scale folding, preshuffled scales, MX/E8M0 scaled MFMA, MXFP4 scale math, fp8 FNUZ vs OCP."
source: https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/epilogue ; https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/gemm_quant ; https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
---

# CK epilogues and quantized GEMM

## 1. Epilogue choices

```
 MFMA acc (fp32, scattered per lane)
     │
     ├─ Default2D ─────────────── cast → direct per-lane store     (no LDS)
     ├─ PermuteN ──────────────── in-register N permute → store    (no LDS, no D tensors)
     ├─ DynamicQuant ──────────── row absmax → scale → cast        (per-token quant)
     └─ CShuffle ── VGPR → LDS ── sync ── LDS → VGPR (coalesced) ── D-fusion ── store
                                   ▲ cast / row-col scale applied here
```

CShuffle specifics worth reusing:
- **Shuffle granularity** (`MXdlPerWavePerShuffle`, `NXdlPerWavePerShuffle`) caps the LDS
  slice per round. It auto-shrinks to 1×1 when LDS would overflow or the mainloop holds a
  double LDS buffer.
- **Cast and scale inside the LDS hop**: fp32→bf16/fp8 conversion and per-row
  (`scale_m`) / per-column (`scale_n`) scaling happen between LDS write and DRAM store.
  Scalar scales skip window creation.
- **D tensors** (bias, residual, gate) are loaded in the output distribution and fused
  before the store (`NumDTensor`). Classic CK: `DeviceGemmMultipleD*`.
- **Memory op `set` vs `update`**: one epilogue path serves overwrite and atomic-add
  (split-K, MoE scatter).
- **Element-op placement** (classic `DoElementwiseBeforeCShuffle`): apply the op on the
  VGPR→LDS side (more VGPR pressure) or the LDS→DRAM side.
- **Padding is word-aware on 64-bank LDS** (gfx950): add one 4-byte word to the row
  stride only when the base stride in words is even.

Dynamic quant epilogue: optional smooth scale per channel (`o *= smooth[n]`) → row absmax
(warp reduce + cross-warp LDS, `v_max3_f32` packs three abs-max ops) →
`y_scale = absmax / max(OutT)` → divide, cast, store values and `y_scale`.

## 2. Group / block-scale GEMM (`gemm_quant`)

Families: `aquant`, `bquant`, `abquant` (scales on A, B or both), and `microscale` (MX).
Group size per operand is a template (`AQuantGroupSize`, `BQuantGroupSize`).

Idioms:
- **Loop over K in quant groups.** Each group iteration loads its A/B scales, runs the
  warp GEMM into a temporary, then folds `c += c_tmp · sa · sb`. Dequantized A or B never
  exists in memory.
- **Preshuffle the weights and their scales on the host** so each warp K-iteration fetches
  exactly one scale per lane with one wide load. A lane that needs another lane's scale
  gets it with `ds_bpermute` instead of a second LDS read.
- **Packed sub-byte types** (int4, fp4) carry a `PackedSize`; LDS sizing and vector width
  must use packed elements, not logical ones.
- fp8/bf8 scales are converted in-register with the hardware cvt builtins.

## 3. MX (microscaling) GEMM on CDNA4

```
 32 consecutive K-elements ──share──► one E8M0 scale  (value = 2^(e − 127), e = 255 is NaN)
 v_mfma_scale_f32_{16x16x128, 32x32x64}_f8f6f4 (A_fp4/6/8, scaleA, B_fp4/6/8, scaleB)
           └─ hardware applies both scales; accumulator is already de-scaled fp32
```

- **No post-GEMM descale.** The scaled MFMA takes packed E8M0 bytes; an op-select field
  picks which byte of a packed 32-bit register applies to this MFMA. Pack 4 scales per
  dword and walk the byte index instead of reloading.
- A and B may use different element types (e.g. fp4 weights × fp8 activations).
- Scale layout must match what the Policy expects. A mis-shuffled scale tensor gives
  silently wrong results. Check register layout with the AMD matrix instruction calculator.
- gfx942 has no FP4/FP6/MX MFMA; MX GEMMs there are emulation only (no speedup).

MXFP4 element and scale math (OCP MX v1.0):
- E2M1 values: `0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6`. Block size 32 is fixed by the spec.
- Scale per block: pick the power of two that puts the block absmax at the largest
  element exponent. Implementations divide absmax by 4.0 (= 2^floor(log2 6)), round to a
  power of two, then cast `x / scale`.
- Storage ≈ 4.25 bits/element (4 + 8/32).
- On CDNA4, FP6 runs at the FP4 rate, so MXFP6 buys accuracy for free when MXFP4 loses
  too much.

## 4. fp8 dialect

| Target | fp8 MFMA encoding | e4m3 max |
|---|---|---|
| gfx942 (CDNA3) | FNUZ (`e4m3fnuz`, `e5m2fnuz`) | 240 |
| gfx950 (CDNA4) | OCP (`e4m3fn`, `e5m2`) | 448 |

FNUZ exponent bias is one higher than OCP, so reading bytes in the wrong dialect is off by
exactly 2×. CK gates FNUZ vs OCP code per target at configure time; match the producer's
quantizer to the consumer's target.

Sources: [ck_tile epilogue](https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/epilogue) ·
[ck_tile gemm_quant](https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/gemm_quant) ·
[OCP MX spec v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) ·
[Microscaling formats paper (arXiv 2310.10537)](https://arxiv.org/abs/2310.10537) ·
[Matrix cores on CDNA3/CDNA4 blog (source)](https://github.com/ROCm/rocm-blogs/blob/release/blogs/software-tools-optimization/matrix-cores-cdna/README.md) ·
[FP8/MXFP GEMM on CDNA4 blog (source)](https://github.com/ROCm/rocm-blogs/blob/release/blogs/software-tools-optimization/cdna4-gemm-kernels/README.md) ·
[AMD matrix instruction calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) ·
[AMD CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
Paths drift; verify before use. Verify specifics against the primary source.

---
Related: [gemm-pipelines-and-schedulers](gemm-pipelines-and-schedulers.md) · [fused-moe-and-flatmm](fused-moe-and-flatmm.md) · [epilogue](../../kernel-optimization/levers/epilogue.md) · [gfx950](../../hardware/amd-cdna/gfx950.md)
