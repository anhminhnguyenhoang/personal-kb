---
name: matrix-core-generations
description: MFMA / SMFMAC / scaled-MFMA across CDNA1–CDNA4 — capability matrix, cycle counts, the per-instruction peak formula, why legacy-K shapes run at half CDNA4 peak, the F8F6F4 cycle rule, and per-lane register footprint.
source: AMD CDNA4 ISA reference (Table 28, MFMA opcodes) ; AMD CDNA3 ISA reference ; https://github.com/ROCm/amd_matrix_instruction_calculator ; https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# Matrix cores across CDNA generations

## 1. What one MFMA is

```
 one wave (64 lanes) issues  v_mfma_<D>_<M>x<N>x<K>_<AB>   D = A·B + C
 A: M×K  spread over 64 lanes → M·K/64 values per lane
 B: K×N  spread over 64 lanes → K·N/64 values per lane
 C/D: M×N spread over 64 lanes → M·N/64 accumulators per lane  (VGPR or AGPR)
 1 matrix core per SIMD → 4 per CU
```

The lane↔element map is fixed per instruction. Get it from the matrix instruction calculator
(`--get-register`), never by guess.

## 2. Capability matrix

| Capability | CDNA1 gfx908 | CDNA2 gfx90a | CDNA3 gfx942 | CDNA4 gfx950 |
|---|---|---|---|---|
| FP16 / BF16 dense | ✓ | ✓ | ✓ K=16 | ✓ K=16 and **K=32** |
| FP32 / FP64 matrix | FP32 only | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ (FP64 rate halved) |
| INT8 | ✓ | ✓ | ✓ K=32 | ✓ K=32 and **K=64** |
| TF32 | — | — | ✓ | **removed** |
| FP8 / BF8 (A, B independent) | — | — | ✓ FNUZ | ✓ OCP |
| F8F6F4 (FP8 / FP6 / FP4 per operand) | — | — | — | ✓ 16×16×128, 32×32×64 |
| Block-scaled (E8M0 per 32 K) | — | — | — | ✓ `v_mfma_scale_*_f8f6f4` |
| SMFMAC 2:4 sparse | — | — | ✓ | ✓ wider K |
| FP16/BF16 subnormals in MFMA | see MMA-Sim | flushed | kept | kept |

## 3. Cycle table (CDNA4, one matrix core)

| Shape | Type | Cycles | FLOPs / cycle / core |
|---|---|---|---|
| 16×16×16, 32×32×8 | FP16 / BF16 (legacy) | 16, 32 | 512 |
| **16×16×32, 32×32×16** | FP16 / BF16 (new) | 16, 32 | **1024** |
| 16×16×32, 32×32×16 | FP8 / BF8 (legacy) | 16, 32 | 1024 |
| 16×16×32, 32×32×16 | INT8 (legacy) | 16, 32 | 1024 |
| **16×16×64, 32×32×32** | INT8 (new) | 16, 32 | **2048** |
| **16×16×128** | F8F6F4, either operand FP8 | 32 | **2048** |
| **16×16×128** | F8F6F4, both FP6/FP4 | 16 | **4096** |
| **32×32×64** | F8F6F4, either FP8 / neither | 64 / 32 | 2048 / 4096 |
| 16×16×4, 32×32×2 | FP32 | 32, 64 | 64 |
| 16×16×4 | FP64 | 64 | 32 |

```
 peak = 2·M·N·K / cycles × matrix_cores × clock
 MI355X FP16 new:     2·16·16·32 / 16 × 1024 × 2.4e9 = 2.5 PF
 MI355X FP16 legacy:  2·16·16·16 / 16 × 1024 × 2.4e9 = 1.26 PF   ← half
 MI355X FP8  legacy:  2·16·16·32 / 16 × 1024 × 2.4e9 = 2.5 PF    ← half of 5 PF
 MI355X FP8  f8f6f4:  2·16·16·128/ 32 × 1024 × 2.4e9 = 5.0 PF
```

Key rule: **on CDNA4 the 2× comes only from the new shapes.** A CDNA3 kernel recompiled for gfx950
keeps K=16 FP16 or K=32 FP8 atoms and sees CDNA3-like per-CU matrix speed on 16% fewer CUs.
FP8 at full rate needs the F8F6F4 instruction with both types set to FP8.

F8F6F4 rule: the larger cycle count applies if A or B is FP8. FP6 and FP4 share one rate.
Mixing FP4 weights with FP8 activations runs at the FP8 rate.

## 4. Same peak, different shape: how to choose

| | 16×16×K | 32×32×K |
|---|---|---|
| Accumulators per lane | 4 | 16 |
| A+B operand loads per FLOP | 2× more | baseline |
| Output layout matches next A input | no | yes (chained GEMM without re-pack) |
| Power efficiency (AMD tuning guide, MI300X) | better | worse |

- Register pressure or power-bound GEMM → 16×16 usually wins on MI300X.
- LDS-read-bound loop or chained QK→PV → 32×32 can win. Lane-layout detail:
  [matrix-instructions](../../kernel-optimization/levers/matrix-instructions.md).
- On CDNA4, re-benchmark the choice. AMD notes the doubled ≤16-bit rate can change it.

## 5. Sparse and scaled variants

- SMFMAC: A is 2:4 structured sparse; a compressed A plus a sparsity index (in `src2`). Doubles the
  dense rate. Pays only with weights pruned to 2:4.
- Scaled MFMA: see [low-precision-formats](low-precision-formats.md) for the E8M0 scale operand,
  OPSEL byte select, and layout.

## 6. Accumulators

- Accumulate in FP32 or INT32. Never down-convert inside the K loop.
- C/D can live in AGPRs to spare architected VGPRs; reading them out costs `v_accvgpr_read_b32` in
  the epilogue. Allocator caveats: [hipcc-agpr-pinning](../../papers/hipkittens/01-paper/hipcc-agpr-pinning.md).
- Back-to-back MFMAs that read the previous result need the documented wait states; they scale
  with pass count. Check the ISA hazard table before hand-scheduling.

Verify specifics against the primary source.

Sources:
- [AMD CDNA4 ISA reference](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) (§7.1 MFMA opcode table, §7.2 block-scaled, sparse MFMA, hazards)
- [AMD CDNA3 ISA reference](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD matrix instruction calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) (`--detail-instruction`, `--get-register`)
- [ROCm: MI300 Series / MI350 Series workload optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html) (16×16 vs 32×32, doubled ≤16-bit rate)
- [MMA-Sim, arXiv 2511.10909](https://arxiv.org/abs/2511.10909) (CDNA2 subnormal flush, CDNA3 rounding)

---
Related: [gfx950](gfx950.md) · [cdna3-cdna4-deltas](cdna3-cdna4-deltas.md) · [low-precision-formats](low-precision-formats.md) · [matrix-instructions](../../kernel-optimization/levers/matrix-instructions.md)
