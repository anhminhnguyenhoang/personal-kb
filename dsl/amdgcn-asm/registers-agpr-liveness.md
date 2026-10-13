---
name: registers-agpr-liveness
description: Hand-allocating the gfx942/gfx950 unified VGPR+AGPR file — accum_offset aliasing, VGPR vs AGPR accumulators, async-load write timing, the clobber bug classes, and a liveness method that finds free registers to cross an occupancy step.
source: https://llvm.org/docs/AMDGPUUsage.html ; AMD CDNA3 and CDNA4 ISA reference guides ; arXiv 2511.08083 (HipKittens)
---

# Registers, AGPRs and liveness in hand-written asm

In asm you are the register allocator. Most wrong-result bugs in hand-written CDNA kernels
are a live register overwritten by something you forgot was in flight.

## 1. The unified file and `accum_offset`

On gfx90a, gfx942 and gfx950 each lane has one 512-entry vector file. `accum_offset`
splits it:

```
 v0 ............ v[accum_offset-1] │ v[accum_offset] ............ v511
 architectural VGPRs               │ AGPRs: a0 == v[accum_offset], a1 == v[accum_offset+1], ...
 ◄──────── align(arch, 4) ────────►│◄──── #AGPRs ────►│  (unallocated)
                                     next_free_vgpr = accum_offset + #AGPRs
```

- `accum_offset` is a multiple of 4 in 4..256. Encoded as `accum_offset/4 - 1` in
  `compute_pgm_rsrc3[5:0]`.
- `next_free_vgpr` must count the AGPRs. If it equals `accum_offset`, zero AGPRs are
  allocated, yet `a[N]` still assembles. Signature: MFMA output is garbage while its
  inputs check out; two MFMAs into the same accumulator give the same answer.
- Any VGPR number at or above `accum_offset` names AGPR storage. A `ds_read_b128 v[64:67]`
  with `accum_offset = 64` overwrites `a[0:3]`. Keep every non-accumulator VGPR below
  `accum_offset`.
- Occupancy uses the combined count: `waves/SIMD = floor(512 / roundup(next_free_vgpr, 8))`,
  capped at 8. Steps at 64 / 128 / 168 / 256. One register over a step costs a whole wave.
  Arithmetic and the gfx942 table: [gfx942-registers-occupancy](../../hardware/amd-cdna/gfx942-registers-occupancy.md).

## 2. AGPR or VGPR accumulators?

The MFMA encoding has per-operand bits (`ACC`, `ACC_CD`) that pick VGPR or AGPR for
A/B and for C/D. Both are legal in asm. On gfx90a and later, loads can also target AGPRs.

| Accumulators in | Gain | Cost |
|---|---|---|
| AGPRs | Frees architectural VGPRs for addresses, operands, prefetch | Epilogue needs `v_accvgpr_read_b32`, a VALU read that must wait for the last MFMA (see [waitcnt-and-hazards](waitcnt-and-hazards.md)) |
| VGPRs (`accum_offset = next_free_vgpr`, 0 AGPRs) | Softmax / scale math reads results directly | Competes with everything else for the low VGPRs |

Attention kernels that touch the score tile every iteration usually keep it in VGPRs.
GEMMs that only read the tile in the epilogue often keep it in AGPRs. Compilers may refuse
AGPRs as MFMA *inputs* even though hardware allows it. Hand asm does not have that limit
(see [hipcc-agpr-pinning](../../papers/hipkittens/01-paper/hipcc-agpr-pinning.md)).

## 3. When does a load write its destination?

```
 issue ─────────────── ~hundreds of cycles ─────────────── data returns ─── s_waitcnt
   │ address VGPRs read HERE                               │ dest VGPRs written HERE
   └── dest must hold nothing live from here ──────────────┴──────────────► until the wait
```

- A VMEM or DS load writes its destination when data **returns**, not at the `s_waitcnt`.
  Anything written to those registers in between is silently overwritten.
- The address is read at issue. So `global_load_dwordx4 v[4:7], v[4:5], off` (dest overlaps
  address) is safe, and compilers emit it.
- Multi-dword ops on gfx90a and later need even-aligned VGPR/AGPR tuples. SGPR descriptors
  need 4-aligned `s[N:N+3]`.

## 4. The clobber bug classes

| Class | Example | Usual symptom |
|---|---|---|
| Wide destination | `ds_read_b128 v[16:19]` overwrites a pointer in `v17` | Right for 1 loop trip, wrong or faulting for 2+ |
| Cross-phase live value | Scale set in the prologue, read in the epilogue, reused as a temp in the loop | Correct accumulation, wrong magnitude |
| Async destination in flight | Prefetch into `v[88:91]` while an MFMA still reads them | Non-deterministic error |
| Load into an address register | `global_load v[36:39]` lands on `v38` = LDS address | NaN in some lanes, only under load |
| Phase dual-use | Zero-init of next accumulator wipes operands still needed | Error in one GEMM of several |
| Reordering | Moving a `ds_read` above the MFMA that consumes the old value | Appears after a "pure scheduling" edit |
| SGPR descriptor reuse | Tile math reuses `s6/s7` of a buffer descriptor | Persistent kernels: wrong after first tile |

## 5. Liveness method (to free registers)

1. List DEF and USE per instruction. Multi-dword ops DEF every register in the range;
   read-modify-write ops (`v_and_b32 v0, 0x3ff, v0`) are both.
2. Cover **every path**: prologue, loop body, loop back-edge (values used next iteration),
   loop exit, epilogue, persistent outer loop, each branch target.
3. Count multi-dword destinations as groups. A 4-wide load keeps 4 registers busy even if
   only one is read soon. Peak pressure estimates rise a lot from naive interval → def-use
   → group-aware → per-register with scheduling constraints. Trust only the last.
4. Look for long dead windows: values recomputable from SGPRs, redundant copies of
   loop-invariant addresses, temps live only in one phase.
5. Remap one range, rebuild, re-run correctness at **several loop trip counts**. A clobber
   often hides at one trip.

Typical payoff: remap a handful of short-lived temps into dead windows and drop below a step
(e.g. 136 → 128 = 3 → 4 waves/SIMD). Freed dead registers can also hold a hoisted prefetch.

Sources: [LLVM AMDGPUUsage — compute_pgm_rsrc1/rsrc3, `.amdhsa_accum_offset`, `.amdhsa_next_free_vgpr`](https://llvm.org/docs/AMDGPUUsage.html) ·
[AMD CDNA3 ISA (register files, MFMA ACC bits)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) ·
[AMD CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) ·
[HipKittens, arXiv 2511.08083](https://arxiv.org/abs/2511.08083)
Verify specifics against the primary source.

---
Related: [kernel-abi-and-prologue](kernel-abi-and-prologue.md) · [waitcnt-and-hazards](waitcnt-and-hazards.md) · [debugging-playbook](debugging-playbook.md) · [isa-resource-inspection](../../kernel-optimization/levers/isa-resource-inspection.md)
