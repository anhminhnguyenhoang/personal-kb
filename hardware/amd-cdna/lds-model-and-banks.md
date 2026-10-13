---
name: lds-model-and-banks
description: CDNA LDS model — 32 vs 64 banks, the bank formula, conflict cost, padding and XOR math per access width, LDS bandwidth per CU, direct global→LDS address contract, transpose reads, and the LDS occupancy limit.
source: AMD CDNA4 ISA reference (LDS chapter, buffer-load-to-LDS section) ; AMD CDNA3 ISA reference ; https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# LDS model and bank rules (CDNA3 / CDNA4)

## 1. Geometry

```
 CDNA3 (gfx942)  64 KB = 32 banks × 512 dwords      bank = (byte_addr / 4) mod 32
 CDNA4 (gfx950) 160 KB = 64 banks × 640 dwords      bank = (byte_addr / 4) mod 64

 dword address:   0   1   2  ...  31 | 32  33 ...      (CDNA3: wraps every 32 dwords = 128 B)
 bank:            0   1   2  ...  31 |  0   1 ...
 dword address:   0   1   2  ...  63 | 64  65 ...      (CDNA4: wraps every 64 dwords = 256 B)
 bank:            0   1   2  ...  63 |  0   1 ...
```

| Item | CDNA3 | CDNA4 |
|---|---|---|
| Capacity per CU | 64 KB | 160 KB |
| Max per workgroup | 64 KB | 160 KB |
| Banks × width | 32 × 4 B | 64 × 4 B |
| Read bandwidth per CU | 128 B/clk | doubled (256 B/clk) |
| Transposing reads | — | `ds_read_b64_tr_b4`, `ds_read_b64_tr_b8`, `ds_read_b64_tr_b16`, `ds_read_b96_tr_b6` |
| Direct buffer→LDS sizes | 1, 2, 4 B/lane | 1, 2, 4, **12, 16** B/lane |

CDNA3 bandwidth detail: 4 SIMDs form 2 SIMD pairs. Each pair has a 128 B/clk LDS bus. The two
pairs cannot read at once, so the CU sees 128 B/clk.

## 2. Conflict cost

- An LDS op takes from ~2 cycles (no conflict) up to 64 cycles (full serialization).
- Conflict = two lanes in the same issue group hit the same bank at different dwords.
- The issue group size depends on the opcode width. On gfx950 `ds_read_b128` and
  `ds_write_b128` group differently: see the per-opcode table in [gfx950](gfx950.md).
- Profile with `SQ_LDS_BANK_CONFLICT` vs `SQ_LDS_IDX_ACTIVE`. Do not trust a paper analysis alone.

## 3. Padding and XOR, by access width

Let `NB` = banks, `W` = dwords per lane access (1 for b32, 2 for b64, 4 for b128),
`S` = row stride in dwords. Lanes read one column across rows.

```
 conflict-free column read needs   S mod NB  to move each row by ≥ W banks
 and   NB / W   rows to fill all banks once

   b32, NB=32: S = 32 → all rows in bank 0  (32-way)     S = 33 → rows in banks 0,1,2… ✓
   b128,NB=64: S = 64 → all rows in banks 0–3            S = 68 → rows in banks 0,4,8… ✓
```

| Fix | Formula | Cost |
|---|---|---|
| Padding | `S = row_dwords + W` | wastes `W / row_dwords` of LDS; 1–2 ALU ops |
| XOR swizzle | `col_slot' = col_slot XOR (row mod (NB / W))` | no waste; a few more ALU ops |

- An XOR sized for 32 banks does not clear 64 banks. Example: b128 reads, row = 64 dwords.
  CDNA3 needs `row mod 8` (8 slots × 4 banks = 32). On gfx950 `ds_read_b128` spans 64 banks, so
  `row mod 8` lets rows `r` and `r+8` collide (2-way). Use `row mod 16` there. Re-derive per arch
  and per opcode.
- Write and read sides must agree on one layout. A swizzle that fixes the read can create conflicts
  on the write if the two group lanes differently.
- Which to pick per arch (XOR on 64 KB parts, padding when LDS is abundant):
  [memory-hierarchy](../../kernel-optimization/levers/memory-hierarchy.md).

## 4. Direct global→LDS: the address contract

```
 LDS_addr(lane) = M0_base + inst_offset + lane_id × size        size = 1, 2, 4 B
 LDS_addr(lane) = M0_base + inst_offset + lane_id × 16          for 12 B and 16 B (CDNA4)

 lane:     0      1      2      3   ...
 16 B:   [####] [####] [####] [####]       contiguous
 12 B:   [###.] [###.] [###.] [###.]       12 B written, 4 B hole per lane
```

- The destination is lane-contiguous from a wave-uniform base. A per-lane LDS address is not
  possible. Express any swizzle through the global (source) address or through the base.
- There is no 8 B (dwordx2) variant. Choose 16, 12, or 4 B.
- The data skips VGPRs. This frees the staging registers and removes the `ds_write`.
- Clamping: writes must stay inside the wave's LDS allocation; mask lanes that would fall outside.

## 5. LDS limits occupancy

```
 workgroups_per_CU (LDS) = floor(LDS_per_CU / LDS_per_workgroup)
   CDNA3: floor(65536 / L)        CDNA4: floor(163840 / L)
   e.g. L = 48 KB → CDNA3: 1 WG/CU   CDNA4: 3 WG/CU
```

Occupancy is the minimum of this, the register limit, and the wave-slot limit
([gfx942-registers-occupancy](gfx942-registers-occupancy.md)). Attention kernels often hit the LDS
limit first on CDNA3.

Verify specifics against the primary source.

Sources:
- [AMD CDNA4 ISA reference](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) (ch. 11 LDS; §9.1.9 buffer load to LDS; DS opcode list)
- [AMD CDNA3 ISA reference](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [ROCm: MI300 Series / MI350 Series workload optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html) (SIMD-pair LDS bus, doubled CDNA4 read bandwidth)
- [LLVM AMDGPU usage](https://llvm.org/docs/AMDGPUUsage.html)

## Lessons from worked examples

- **M0 supplies the LDS address for load-to-LDS.** Buffer and global loads that write straight to LDS take their LDS base from the M0 register, not from a VGPR. Set or advance M0 before each such load, or consecutive loads overwrite the same LDS region. ([CDNA3 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf), MUBUF LDS loads)

---
Related: [gfx950](gfx950.md) · [memory-hierarchy](../../kernel-optimization/levers/memory-hierarchy.md) · [memory-ladder-xcd-hbm](memory-ladder-xcd-hbm.md) · [matrix-core-generations](matrix-core-generations.md)
