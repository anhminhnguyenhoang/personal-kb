---
name: matrix-instructions
description: MFMA/WMMA atom selection, operand layout (lane-layout matching across chained atoms — the 32x32 C-out matches A-in, 16x16 does not), accumulator strategy, and reducing MFMA count via K-pack.
source: CK DSL Optimization Runbook (ROCm/rocm-libraries@1efeadd, projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
---

## 7. Matrix Instructions And Compute

### 7.1 MFMA/WMMA Selection

Typical atom catalog:

```text
f16: 16x16x16, 16x16x32, 32x32x8, 32x32x16, 4x4x4
bf16: 16x16x16, 16x16x32, 32x32x16
fp8 / bf8: native CDNA4 fp8 / bf8 MFMA + scaled / MX variants
```

See [gfx950](../../hardware/amd-cdna/gfx950.md) for the full per-arch atom catalog, including CDNA4-
only atoms (fp8 / bf8 / scaled / MX) and the lane-layout-match
property of the 32×32 atoms.

- `16x16x16` for fp16/bf16 standard tiles.
- `16x16x32` for fp16 on newer AMD architectures when K packing is
  valid.
- `4x4x4` for many independent small-channel computations
  (used by small-channel direct convolution).
- Scaled MFMA for fp8 / fp6 / fp4 where available.
- WMMA for RDNA / wave32 architectures — not currently exposed.

Always confirm the emitted intrinsic with the intrinsic-count probe.
For example, switching from the default to `mfma_f32_32x32x16_f16` is
visible immediately: the `mfma.f32.32x32x16.f16` line goes from 0 to
N, and the `mfma.f32.16x16x32.f16` line drops correspondingly.

### 7.2 Operand Layout

- Verify lane-to-output mapping for each atom (write a lane→element
  table and compare).
- Verify packed vector element order.
- Verify A/B orientation.
- Verify K packing.
- Verify output accumulator lane mapping.
- Write a tiny kernel or test if unsure.
- Compare against reference before optimizing performance.

For `mfma_f32_16x16x32_f16` on AMD CDNA, lane `(c4 = lane / 16)` holds
K elements `[c4*8 : c4*8 + 8]` (not a flat concatenation of two
4-element halves) — see Section 5.3 for the canonical bug signature.

Lane-layout matching between chained atoms matters as much as atom
shape. The `mfma_f32_16x16x32` atom's C-output lane layout does not
match its A-input lane layout, so a QK→PV chain using two 16×16 atoms
needs an LDS round-trip or `~288` cross-lane permute ops per iter to
re-pack between them. The `mfma_f32_32x32x16` atom's C-output natively
matches the A-input, so the same chain runs with zero re-pack. A unified-attention
worked example quantifies this: the 32×32 atom (with a transposed QK
layout) is structurally cheaper *because of* this
layout match, independent of the atom's compute throughput.

### 7.3 Accumulator Strategy

- Use enough accumulators to hide latency but not so many that VGPR
  pressure collapses occupancy.
- Circular accumulators can avoid shifting data.
- Reset accumulator slots at the correct lifecycle point.
- For convolution, reset phantom-row accumulators even when the
  output row is OOB. The CK direct-conv "unconditional slot reset"
  trick is non-obvious and easy to get wrong. Symptom of the bug:
  error roughly proportional to one filter coefficient times one row
  of input (we measured `max_abs ~ 1e-2`, mean ~ 1.4e-3, with ~50 %
  of elements above `1e-3`).
- For split-K, define accumulation and reduction strategy.
- For attention, keep max / sum / output accumulators consistent.
  An online-softmax state object that carries these is the usual pattern.

### 7.4 Reducing MFMA Count

- Fold small dimensions into K if operand layout supports it.
- Use larger K MFMA variants when valid (`16x16x32` over `16x16x16`,
  `32x32x16` over `32x32x8`).
- Fuse filter positions into K for convolution when contiguous.
- Use Toeplitz-like packing for convolution only when correct and
  worth the complexity.
- Avoid K32/K64 if it changes the result beyond tolerance.
- Check whether MFMA reduction count reduction causes memory / layout
  overhead elsewhere.

## Lessons from worked examples

- **Prefer the K=32 bf16 atom on CDNA4.** The 16x16x32 bf16 atom keeps the same accumulators per lane as 16x16x16 but halves the MFMA issue count, and the result is bit-exact. It is an issue-efficiency win, not an occupancy win. CDNA3 lacks it and falls back to 16x16x16. It needs a K-permute change in the operand layout so each lane holds 8 contiguous K elements (e.g. +4-7% on a memory-bound grouped GEMM).
- **Preshuffle block scales to match the lane layout.** A scaled MFMA lane needs several scale bytes that are strided in a row-major scale tensor, which gives narrow loads and extra waits. Store the scales offline in lane order so each lane reads them in one load. ([Triton block-scaled matmul tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/10-block-scaled-matmul.py))
- **MFMA result wait states scale with pass count.** Before any consumer other than a same-accumulator MFMA reads D, the ISA requires about 5/8/12/20 wait states for 2/4/8/16-pass MFMAs. Back-to-back accumulation into the same D needs none. Fill the gap with independent work instead of blanket `s_nop`. ([CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf), MFMA dependency table) Verify specifics against the primary source.

ck_dsl specifics: [primitive-map](../../dsl/ck-dsl/primitive-map.md#matrix-instructions).

---
Full atom catalog: [target-architecture-gfx950](../../hardware/amd-cdna/gfx950.md). Atom/K-pack knobs: [knob-catalog-and-sweep](../../dsl/ck-dsl/knob-catalog-and-sweep.md).
