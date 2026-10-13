---
name: low-precision-formats
description: CDNA low-precision numerics — FP8 FNUZ (CDNA3) vs OCP (CDNA4) bit-level differences, FP6/FP4 ranges, MX block scaling with E8M0 and the scaled-MFMA operand, scaled converts with saturation and stochastic rounding, and MFMA rounding/subnormal behavior per generation.
source: AMD CDNA4 ISA reference (§7.2 block-scaled, §7.3 small float formats) ; AMD CDNA3 ISA reference ; OCP Microscaling Formats v1.0 spec ; arXiv 2511.10909 ; https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# Low-precision formats on CDNA (FP8, FP6, FP4, MX)

## 1. FP8: FNUZ on CDNA3, OCP on CDNA4

| | E4M3 FNUZ (gfx942) | E4M3 OCP (gfx950) | E5M2 FNUZ (gfx942) | E5M2 OCP (gfx950) |
|---|---|---|---|---|
| Exponent bias | 8 | 7 | 16 | 15 |
| Max finite | ±240 | ±448 | ±57344 | ±57344 |
| −0 | no (0x80 = NaN) | yes (0x80) | no (0x80 = NaN) | yes (0x80) |
| Inf | no | no | no | yes (0x7C / 0xFC) |
| NaN | 0x80 only | 0x7F, 0xFF | 0x80 only | 0x7D–0x7F, 0xFD–0xFF |

```
 same byte, different value (normal numbers):
   value_OCP = 2 × value_FNUZ          (bias differs by 1)

 byte 0x80 :  OCP → −0        FNUZ → NaN
 byte 0x7F :  OCP → NaN       FNUZ → 240   (E4M3)
 OCP 448.0 :  no FNUZ encoding (max 240) → saturate or overflow
```

Rule: convert FP8 data between generations with a real cast. A bit-copy gives values off by 2×,
turns −0 into NaN, and clips the top of the E4M3 range. Check checkpoints, KV caches, and any
pre-quantized weights.

## 2. FP6 and FP4 (CDNA4)

| Format | Bias | Max | Min normal | Min subnormal | Inf / NaN |
|---|---|---|---|---|---|
| FP6 E2M3 | 1 | ±7.5 | 1.0 | 0.125 | none |
| BF6 E3M2 | 3 | ±28 | 0.25 | 0.0625 | none |
| FP4 E2M1 | 1 | ±6 | 1.0 | 0.5 | none |

No Inf, no NaN. Out-of-range values saturate to max on convert; tiny values go to zero.
FP4 values: `0, 0.5, 1, 1.5, 2, 3, 4, 6` (and negatives). Two FP4 per byte.

## 3. MX block scaling (OCP MX, E8M0)

```
 K dimension ─────────────────────────────────────────────►
 │ e0 e1 … e31 │ e32 … e63 │ e64 … e95 │ e96 … e127 │     one row of A
 └──── s0 ─────┘└─── s1 ───┘└─── s2 ───┘└─── s3 ────┘     one E8M0 scale per 32 K values

 E8M0 scale s:  value = 2^(s − 127)     s = 127 → ×1     s = 255 → NaN
 dot product per block, then × scale_A × scale_B, then accumulate in FP32
```

- Cost: 8 bits per 32 elements = +0.25 bit/element. MXFP4 ≈ 4.25 bit/element.
- Why it helps: each 32-block gets its own range. A per-tensor scale is set by the largest
  outlier and flushes small values.
- Scaled MFMA (`v_mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4`): A and B types are set
  independently (FP8, BF8, FP6, BF6, FP4).

### The scale operand

```
 16×16×128:  16 rows × (128/32) = 64 scales per matrix  → 1 byte per lane
 32×32×64 :  32 rows × (64/32)  = 64 scales per matrix  → 1 byte per lane

 scale VGPR (32 bit):  [ byte3 | byte2 | byte1 | byte0 ]
                          ▲ OPSEL code 11  …  00 ▲      selects which byte this MFMA uses
```

- One VGPR holds the scales for 4 MFMAs. Step OPSEL instead of loading a new VGPR.
- The scale can be an inline constant (exponent part used). Use it for an unscaled operand.
- The scale applies to that one instruction only. It does not carry into later non-scale MFMAs.
- Exact lane-to-scale map: matrix instruction calculator, or the ISA layout figures.

## 4. Converts with scale (CDNA4)

- `v_cvt_scalef32_*`: convert FP32/FP16/BF16 ↔ FP8/BF8/FP6/BF6/FP4 with an E8M0-style scale in
  one VALU op. Packed forms (`pk`, `pk32`, `2xpk16`) convert many values at once.
- `_sr_` variants use stochastic rounding. Use them when quantizing gradients or long
  accumulations where round-to-nearest bias matters.
- To FP4/FP6: saturate at max, flush below min subnormal.

## 5. MFMA arithmetic per generation

| Gen | Behavior | Effect |
|---|---|---|
| CDNA2 | FP16/BF16 MFMA flush subnormals | Smaller dynamic range; can hurt training |
| CDNA3 | TF32/FP16/BF16 MFMA: fused sum rounded with round-down (asymmetric) | Small bias; grows over long K or many steps |
| CDNA3 | FP8 MFMA uses a modified rounding | Bias mitigated |
| CDNA4 | MFMA keeps input/output denormals (ignores MODE denorm bits) | — |
| CDNA4 | No TF32 | Emulate with BF16 or use FP32 |

All: accumulate in FP32 or INT32. For bit-exact reference outputs of CDNA1–3 MFMAs, use MMA-Sim.
Compare quantized kernels against a task-accuracy gate, not bit parity.

Verify specifics against the primary source.

Sources:
- [AMD CDNA4 ISA reference](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) (§7.2 block-scaled matrices, §7.3 small float formats and conversions, VALU convert list)
- [AMD CDNA3 ISA reference](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) (FNUZ FP8/BF8)
- [OCP Microscaling Formats (MX) v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [MMA-Sim, arXiv 2511.10909](https://arxiv.org/abs/2511.10909) · [microsoft/MMA-Sim](https://github.com/microsoft/MMA-Sim)
- [ROCm: MI300 Series / MI350 Series workload optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html) (FNUZ vs OCP, TF32 emulation)

---
Related: [matrix-core-generations](matrix-core-generations.md) · [cdna3-cdna4-deltas](cdna3-cdna4-deltas.md) · [gfx950](gfx950.md)
