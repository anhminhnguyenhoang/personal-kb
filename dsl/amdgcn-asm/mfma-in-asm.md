---
name: mfma-in-asm
description: Writing v_mfma by hand on gfx942/gfx950 — operand register sizes and alignment, lane layout formulas for A/B/D, cbsz/abid/blgp meanings (broadcast vs format select), scaled MFMA scale selection, forced RNE/FTZ, and the FNUZ-vs-OCP fp8 trap.
source: AMD CDNA3 and CDNA4 ISA reference guides (chapter 7, matrix arithmetic) ; https://github.com/ROCm/amd_matrix_instruction_calculator ; https://github.com/ROCm/rocm-blogs/tree/release/blogs/software-tools-optimization/matrix-cores-cdna ; https://rocm.docs.amd.com/en/latest/reference/precision-support.html
---

# MFMA in hand-written asm

Atom choice and throughput: [matrix-instructions](../../kernel-optimization/levers/matrix-instructions.md),
[gfx950](../../hardware/amd-cdna/gfx950.md). This note is the asm-level detail.

## 1. Anatomy

```
 v_mfma_f32_16x16x32_bf16  a[0:3],  v[40:43], v[44:47], a[0:3]   cbsz:0 abid:0 blgp:0
                           └─ D ─┘  └── A ──┘ └── B ──┘ └─ C ─┘
   D, C : VGPR or AGPR (ACC_CD bit)    A, B : VGPR or AGPR (ACC bits)    C may be literal 0 to start fresh
```

| Instruction (examples) | Cycles | A, B regs / lane | D regs / lane |
|---|---|---|---|
| `16x16x16_bf16` | 16 | 2 + 2 | 4 |
| `16x16x32_bf16` (gfx950) | 16 | 4 + 4 | 4 |
| `32x32x16_bf16` (gfx950) | 32 | 4 + 4 | 16 |
| `16x16x32_fp8_fp8` | 16 | 2 + 2 | 4 |
| `16x16x128_f8f6f4` (gfx950) | 16 (fp4/fp6) / 32 (any fp8) | 8 + 8 | 4 |
| `32x32x64_f8f6f4` (gfx950) | 32 / 64 | 8 + 8 | 16 |

Pass count = cycles / 4. It sets the wait states before anything else reads D
([waitcnt-and-hazards](waitcnt-and-hazards.md)). Back-to-back MFMAs of the same opcode that
accumulate into the exact same D need no NOPs.
Multi-register operands must start on an even register. The f8f6f4 forms always take
8-register (256-bit) A/B tuples. fp4 data fills half of it; keep the rest zero.

## 2. Lane layouts (single-block MFMA, wave64)

```
 K_L = K / (64 / M)            # K values held by one lane

 A[i][k] → lane  i + M·(k / K_L),          item  k mod K_L
 B[k][j] → lane  j + N·(k / K_L),          item  k mod K_L
 D[i][j] → lane  j + N·((i/4) mod M_I),    item  (i mod 4) + 4·(i / (4·M_I))      M_I = 64 / N
```

| Shape | D: lane | D: item | Reading it |
|---|---|---|---|
| 16×16 | `j + 16·(i/4)` | `i mod 4` | lane owns column `j`; its 4 regs are 4 consecutive **rows** |
| 32×32 | `j + 32·((i/4) mod 2)` | `(i mod 4) + 4·(i/8)` | 4 groups of 4 rows, alternating lane halves |

- The output is "column per lane". Storing row-major C needs a stride of `ldc` between the
  4 values a lane holds, or a pack/shuffle first.
- If you swap the roles of A and B (feed K as A, Q as B), every score of one query row lands
  in one lane. Row max / row sum in softmax then become lane-local `v_max3_f32` / `v_add_f32`
  chains instead of cross-lane shuffles.
- Do not assume a layout from another shape. Print it:
  `matrix_calculator.py --architecture cdna3 --instruction v_mfma_f32_16x16x16_bf16 --register-layout --D-matrix`.

## 3. `cbsz`, `abid`, `blgp` mean different things per family

| Family | `cbsz` | `abid` | `blgp` |
|---|---|---|---|
| Dense multi-block MFMA (4x4, 32x32x1_2b ...) | A broadcast block size | which A block broadcasts | B lane-group permute/broadcast |
| `*_f8f6f4` (gfx950) | **A format** | — | **B format** |
| `v_mfma_scale_*_f8f6f4` | A format | scale enable | B format |
| `v_smfmac_*` (2:4 sparse A) | index-set select mode | which index set in the SrcC register | — |
| F64 MFMA | — | — | negate A / B / C |

f8f6f4 format codes: 0 = E4M3 (fp8), 1 = E5M2 (bf8), 2 = E2M3 (fp6), 3 = E3M2 (bf6),
4 = E2M1 (fp4). A and B may differ. Single-block dense shapes take 0 for all three.

Scaled MFMA (`v_mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4`) applies one E8M0 scale per
32 K values per row/column of A and B. The scales sit in a VGPR. `op_sel`/`op_sel_hi` pick
which byte of that register is used. The scale exponents add to the product exponent
**inside** the MFMA, so there is no post-MFMA `v_mul_f32` chain.
SMFMAC replaces C with a compression-index operand; print its layout with the calculator's
`--compression` flag.

## 4. Numerics you cannot change

- Every MFMA ignores `MODE`: round-to-nearest-even, denormals flushed (F64 keeps them).
  A reference built from `v_fma_f32` with other modes will differ in the last bits.
- **fp8 dialect follows the chip.** gfx942 MFMA reads **FNUZ** fp8/bf8 (E4M3 bias 8, max
  240, no −0, `0x80` = NaN). gfx950 reads **OCP** (E4M3 bias 7, max 448, `0x7F/0xFF` = NaN).
  The same bits decode to **2×** the FNUZ value under OCP for every finite value
  (bias differs by 1; bf8 too). Feed FNUZ
  data to a gfx950 MFMA and the product is 4× too big (both operands). Fix by converting
  the data, or fold 0.25 into the output scale if no special values occur.
- `v_cvt_pk_bf16_f32 d, lo, hi` (gfx950) rounds RNE and puts the **first** source in bits
  [15:0]. It replaces the software rounding sequence gfx942 code needs
  (extract the rounding bit, add the bias, select NaN, shift, pack).
- `v_exp_f32` computes 2ˣ. Pre-multiply by log2(e) (fold it into the softmax scale) to get eˣ.

## 5. Shape choice reminders

- 16×16 needs 4 D registers per lane per tile; 32×32 needs 16. That decides VGPR pressure
  long before throughput does.
- On MI300X, 16×16 shapes can reach more sustained FLOP/s than 32×32 because the larger
  shape draws more power and lowers clocks. Measure both.

Sources: [AMD CDNA3 ISA — ch. 7 matrix arithmetic, dependency table](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) ·
[AMD CDNA4 ISA — §7.1.4 layouts, §7.1.5 f8f6f4 formats, §7.2 scaled MFMA, §7.5 SMFMAC](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) ·
[amd_matrix_instruction_calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) ·
[ROCm blog — Matrix cores on CDNA3/CDNA4](https://github.com/ROCm/rocm-blogs/tree/release/blogs/software-tools-optimization/matrix-cores-cdna) ·
[ROCm blog — Measuring max-achievable FLOPs, part 2](https://github.com/ROCm/rocm-blogs/tree/release/blogs/software-tools-optimization/measuring-max-achievable-flops-part2) ·
[ROCm precision support (FNUZ vs OCP fp8)](https://rocm.docs.amd.com/en/latest/reference/precision-support.html)
Verify specifics against the primary source.

---
Related: [waitcnt-and-hazards](waitcnt-and-hazards.md) · [registers-agpr-liveness](registers-agpr-liveness.md) · [lds-patterns-asm](lds-patterns-asm.md) · [matrix-instructions](../../kernel-optimization/levers/matrix-instructions.md)
