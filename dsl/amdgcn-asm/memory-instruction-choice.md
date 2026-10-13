---
name: memory-instruction-choice
description: Choosing between buffer, global, flat, scalar, and direct-to-LDS loads in gfx942/gfx950 asm — address forms, bounds behavior, counters, register cost, and when each one fits.
source: AMD CDNA3 and CDNA4 ISA reference guides (MUBUF, FLAT/global, SMEM chapters) ; https://llvm.org/docs/AMDGPUUsage.html ; https://github.com/ROCm/aiter
---

# Buffer vs global vs flat vs scalar vs LDS-DMA

The same bytes can reach a kernel by five instruction families. They differ in address
registers, bounds behavior, which counter they use, and where data lands.

## 1. Decision table

| Family | Address | Out of bounds | Counter | Lands in | Use for |
|---|---|---|---|---|---|
| `buffer_load/store_*` (MUBUF) | V# (4 SGPRs) + SGPR offset + optional VGPR offset/index + 12-bit imm | Load returns 0, store dropped, no fault | vmcnt | VGPR or AGPR | Tiles with ragged edges, any wave-uniform base |
| `global_load/store_*` | 64-bit VGPR pair, or SGPR base + 32-bit VGPR offset, + small signed imm | Faults | vmcnt | VGPR or AGPR | Arbitrary per-lane pointers, >4 GB spans |
| `flat_*` | 64-bit generic address (may be LDS or global) | Faults | vmcnt **and** lgkmcnt, out of order | VGPR | Only when the pointer may point into LDS |
| `s_load_*`, `s_buffer_load_*` | SGPR base (+ imm or SGPR offset) | — | lgkmcnt (out of order) | SGPR | Wave-uniform data: kernargs, per-block scales, expert ids, page tables |
| `buffer_load_* ... lds`, `global_load_lds_*` | as buffer/global; LDS dest from M0 | as buffer/global | vmcnt | LDS, no VGPR | Staging tiles without VGPR round trip |

## 2. Buffer (MUBUF) details

```
 address = V#.base + SOFFSET + (OFFEN ? vgpr_offset : 0) + (IDXEN ? vgpr_index * V#.stride : 0) + inst_offset
 bounds  : checked against V#.num_records  →  OOB load = 0, OOB store = dropped
```

- The V# holds base (48 bits), stride, `num_records`, swizzle bits, format fields, type.
  Copy the format/word-3 constants from compiler output for your target instead of guessing.
- Per-lane cost is one VGPR (offset). Global needs two for a 64-bit address.
- OOB-zero is a free mask: point dead lanes past `num_records` (e.g. offset `0x80000000`)
  instead of `s_and_saveexec` around the load. Tails, padded K, and paged gaps come free.
- Offsets are 32-bit, so one descriptor spans at most 4 GB. Rebase the descriptor per
  tile or page, or use global loads, beyond that.
- Several descriptors per kernel (A, B, scales, output) each take 4 aligned SGPRs. Keep
  them out of the SGPRs you reuse for tile math. Persistent kernels that reuse `s6/s7`
  for arithmetic corrupt the descriptor after the first tile.

## 3. Global and flat

- The `global_*` form with an SGPR base (`global_load_dword v1, v0, s[2:3]`) needs only a
  32-bit VGPR offset. It halves address VGPRs compared with the 64-bit VGPR form.
- There is no bounds check. Mask lanes with EXEC or clamp the address.
- `flat_*` resolves the address space per lane at run time. Because it may touch LDS it
  counts on both counters and can finish out of order against both. In hot loops use
  `global_*` or `ds_*` and keep `flat_*` for generic pointers.

## 4. Scalar loads

- One load per wave, result in SGPRs, no VGPR cost. Ideal for anything uniform across the
  wave: kernargs, one fp32 scale per KV block, expert id for the tile, a page-table entry.
- The scalar cache is **not coherent** with the vector caches. Use SMEM only for data that
  does not change during the dispatch. Do not poll a flag with `s_load`.
- Wait with `lgkmcnt(0)` when several scalar loads are in flight (they return out of order).

## 5. Direct-to-LDS (LDS-DMA)

```
 s_mov_b32 m0, s_lds_base        ; wave-uniform LDS base
 s_nop     0                     ; SALU-writes-M0 hazard
 buffer_load_dword v_off, s[8:11], 0 offen lds
 ; LDS[m0 + inst_offset + lane * 4] = mem[...]    (lane-contiguous, per wave)
 s_waitcnt vmcnt(0)              ; completion is vmcnt, not lgkmcnt
 s_barrier                       ; before other waves read it
```

- gfx942 supports byte, short and dword widths. gfx950 adds `dwordx3` and `dwordx4`.
- The LDS layout is fixed: lane-contiguous from the M0 base. An XOR-swizzled or transposed
  layout must come from the **global** addressing side, not the LDS side.
- Give each wave its own M0 base, or waves overwrite each other.
- No VGPR destination means nothing to hoist or double-buffer in registers. Pipelining
  moves to LDS buffers and barriers.

## 6. Width and destination

- Use 128-bit ops (`*_dwordx4`, `ds_read_b128`) for tiles. Fewer instructions, fewer
  counter slots, better coalescing.
- On gfx90a and later, VMEM loads can write AGPRs directly. Public aiter flatmm kernels
  load MFMA operands straight into `a[...]` and keep accumulators in VGPRs.

Sources: [AMD CDNA3 ISA — MUBUF buffer addressing, OOB, buffer-load-to-LDS, FLAT/global, SMEM](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) ·
[AMD CDNA4 ISA — §9 buffer, §10 flat](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) ·
[LLVM AMDGPUUsage — scalar cache coherence, memory spaces](https://llvm.org/docs/AMDGPUUsage.html) ·
[ROCm/aiter `hsa/` code objects](https://github.com/ROCm/aiter)
Verify specifics against the primary source.

---
Related: [memory-coherence-bits](memory-coherence-bits.md) · [lds-patterns-asm](lds-patterns-asm.md) · [waitcnt-and-hazards](waitcnt-and-hazards.md) · [memory-hierarchy](../../kernel-optimization/levers/memory-hierarchy.md)
