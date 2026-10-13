---
name: lds-patterns-asm
description: LDS at the instruction level on gfx942/gfx950 — DS offset fields, read2/st64 forms, transpose reads, ping-pong buffer toggles, barrier counts, per-wave regions, and cross-wave reductions written by hand.
source: AMD CDNA3 and CDNA4 ISA reference guides (LDS chapter, DS encoding, MFMA transpose loads) ; https://llvm.org/docs/AMDGPUUsage.html ; https://github.com/ROCm/aiter
---

# LDS patterns in hand-written asm

Bank-conflict theory and swizzle choice live in [memory-hierarchy](../../kernel-optimization/levers/memory-hierarchy.md)
and the per-arch table in [gfx950](../../hardware/amd-cdna/gfx950.md). This note is about
the instructions and the protocols around them.

## 1. Use the offset fields, not VALU

| Form | Offset field | Reach |
|---|---|---|
| `ds_read_b{32,64,96,128}`, `ds_write_*` | 16-bit unsigned byte offset | 0..65535 from one base VGPR |
| `ds_read2_b32` / `ds_write2_b32` | two 8-bit offsets × 4 B | two dwords up to 1020 B apart |
| `ds_read2_b64` / `ds_write2_b64` | two 8-bit offsets × 8 B | up to 2040 B |
| `ds_read2st64_*` / `ds_write2st64_*` | two 8-bit offsets × 64 elements | strides of 256 B (b32) or 512 B (b64) |

One base VGPR per wave-role plus immediate offsets can address a whole double-buffered
tile. Every `v_add` you remove from the inner loop is issue slots back for MFMA.

Compute LDS bases once in the prologue and keep them in registers for the whole loop. Advance
only a wave-uniform offset (SGPR) or toggle a bit.

## 2. Feeding MFMA

- `ds_read_b128` gives 4 VGPRs = 8 bf16 or 16 fp8 per lane. Check how many K values
  per lane your MFMA wants: one read may feed one MFMA (16x16x32 bf16) or two (16x16x16 bf16).
- Changing MFMA shape changes the lane-to-K mapping. LDS read addresses must be redone, not
  just the opcode.
- Load the operand that is shared by more MFMAs **once** and reuse it across the other
  operand's blocks (B reuse across A rows). Fewer reads per K step = less LDS bandwidth.
- gfx950 has MFMA transpose loads: `ds_read_b64_tr_b4`, `ds_read_b64_tr_b8`,
  `ds_read_b64_tr_b16`, `ds_read_b96_tr_b6`. Two instructions fill one operand, already in
  MFMA input order. Rules: EXEC all ones, address aligned to the element, even-aligned
  destination (except the b96 form). They replace `ds_read` + `v_perm_b32` shuffles for
  `Aᵀ·B` shapes.
- On gfx942 (no transpose loads), shuffle in registers before the write: `v_perm_b32` with
  selectors such as `0x01000504` / `0x03020706` interleaves 16-bit halves of two dwords.
  A few perms + `ds_write2_b32` beat a separate transpose pass. Compute perms right before
  use to keep register pressure flat.

## 3. Ping-pong buffers

```
 iter k   : MFMA reads buf[k&1]        loads fill buf[(k+1)&1]
 toggle   : v_xor_b32 v_base, 0x4000, v_base   (one XOR per base register)
 or       : unroll by 2 → buf0 / buf1 offsets are immediates, no toggle at all
```

Barrier count per K step, from [waitcnt-and-hazards](waitcnt-and-hazards.md):

| Scheme | Barriers per K step | Why |
|---|---|---|
| Single buffer, VGPR-staged | 2 | readers done before write; write landed before read |
| Double buffer, VGPR-staged | 1–2 | the write targets the idle buffer, but slow waves may still read it |
| Separate A and B phases | up to 4 | each operand has its own read-done / write-done pair |
| Direct-to-LDS double buffer | 2–3 | wait is `vmcnt`; extra barrier so next DMA does not overwrite a slow reader |

Every barrier in these schemes guards one hand-off. Remove one and you get rare, small
errors that grow with the number of waves and the trip count. Remove a buffer to save
LDS and you may lose a workgroup per CU instead. Barriers usually cost less than that.

## 4. Regions per wave

```
 LDS (per workgroup)
 ┌───────────────┬───────────────┬──────┬──────┬──────┬──────┐
 │ shared K tile │ shared V tile │ Q w0 │ Q w1 │ Q w2 │ Q w3 │   ← shared + per-wave slices
 └───────────────┴───────────────┴──────┴──────┴──────┴──────┘
```

- Shared tiles need barriers. Per-wave scratch (`base + wave_id * size`) does not.
- Direct-to-LDS writes lane-contiguous data from a wave-uniform M0. Give each wave its
  own M0 base.
- Read-only broadcast data (scales) in LDS is cheap: same-address reads in a phase broadcast.

## 5. Cross-wave reduction

```asm
; per-wave partial in v36 (already reduced inside the wave with DPP / v_max3)
ds_write_b32   v_slot, v36 offset:1024    ; slot = wave_id * 4
s_waitcnt      lgkmcnt(0)
s_barrier
ds_read_b128   v[40:43], v_zero offset:1024   ; all 4 waves' partials
s_waitcnt      lgkmcnt(0)
v_max3_f32     v44, v40, v41, v42
v_max_f32      v44, v44, v43
```

Reduce inside the wave first (DPP `row_shr`/`row_bcast`, `v_max3_f32`, `ds_swizzle_b32`,
`ds_bpermute_b32`). `ds_bpermute_b32` / `ds_swizzle_b32` use the LDS crossbar but no LDS
storage, and they count on `lgkmcnt`. Only one value per wave should cross waves through LDS.

Sources: [AMD CDNA3 ISA — LDS, DS instruction formats](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) ·
[AMD CDNA4 ISA — §11 LDS incl. MFMA transpose loads, §12.12 DS opcodes](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) ·
[LLVM AMDGPUUsage — GFX942 memory model, LDS ordering](https://llvm.org/docs/AMDGPUUsage.html) ·
[ROCm/aiter `hsa/` kernels (`v_perm_b32` packing, LDS reductions)](https://github.com/ROCm/aiter)
Verify specifics against the primary source.

---
Related: [waitcnt-and-hazards](waitcnt-and-hazards.md) · [memory-instruction-choice](memory-instruction-choice.md) · [mfma-in-asm](mfma-in-asm.md) · [memory-hierarchy](../../kernel-optimization/levers/memory-hierarchy.md)
