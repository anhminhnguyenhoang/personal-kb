---
name: waitcnt-and-hazards
description: The two things hardware does not check for you in gfx942/gfx950 asm — memory completion (vmcnt/lgkmcnt ordering, barriers) and pipeline hazards (required wait states for MFMA, DPP, M0, trans ops) — plus inline-asm rules.
source: AMD CDNA3 and CDNA4 ISA reference guides (s_waitcnt, manually inserted wait states, MFMA dependency table) ; https://llvm.org/docs/AMDGPUUsage.html (memory model) ; https://github.com/ROCm/HIP/issues/3333
---

# `s_waitcnt`, barriers and hazards

Two separate mechanisms, both your job in asm:

```
 data not back yet from memory   ──► counters  ──► s_waitcnt vmcnt(N) / lgkmcnt(N)
 result not yet in register file ──► hazards   ──► s_nop / independent instructions
```

A wait satisfies only the first. The assembler inserts nothing for either.

## 1. Counters (gfx9 / CDNA layout)

| Counter | Bits | Counts | Returns in order? |
|---|---|---|---|
| `vmcnt` | 6 (≤63) | MUBUF / global / scratch loads, stores, atomics; also `buffer_load ... lds` and `global_load_lds_*` | Yes, loads and stores mixed |
| `lgkmcnt` | 4 (≤15) | LDS (`ds_*`), SMEM (`s_load_*`, `s_memtime`), `s_sendmsg`, FLAT | LDS in order; **SMEM can return out of order** |
| `expcnt` | 3 | exports (not used by compute) | — |

- `s_waitcnt vmcnt(N)` means "wait until at most N remain". The oldest finish first.
- There is no separate store counter on CDNA (no `vscnt`). Stores count in `vmcnt`.
- SMEM out of order: when scalar loads are outstanding, only `lgkmcnt(0)` is safe.
- FLAT ops bump both counters and may complete out of order against both queues. Wait for 0.
- Encoding: `simm16[3:0]` = vmcnt low, `[6:4]` = expcnt, `[11:8]` = lgkmcnt,
  `[15:14]` = vmcnt high. Omitted fields mean "do not wait".

### The reorder trap

`vmcnt(N)` counts by **issue order**, not by register. After any hoist or reorder, recount
every wait below it.

```asm
global_load_dwordx4 v[80:83], ...   ; prefetch  (oldest)
global_load_dwordx4 v[48:51], ...   ; data A
s_waitcnt vmcnt(1)                  ; finishes the PREFETCH; v[48:51] may still be in flight
```

Issue what you need first before you issue prefetches, or account for them in N.
Same rule for `lgkmcnt` and the `ds_read` stream: `lgkmcnt(1)` after three reads means the
newest is still pending.

### Pipelined waits

```asm
ds_read_b128 v[40:43], v14          ; operand for MFMA k
ds_read_b128 v[44:47], v15          ; operand for MFMA k+1
s_waitcnt    lgkmcnt(1)             ; only the older one
v_mfma_...   a[0:3], v[40:43], ...  ; runs while the newer read lands
s_waitcnt    lgkmcnt(0)
v_mfma_...   a[0:3], v[44:47], ...
```

Gains only show when LDS latency is a real share of the MFMA window. In long MFMA chains,
`lgkmcnt(0)` vs `(1)` often measures the same.

## 2. Barriers and LDS visibility

- `s_barrier` waits for waves. It does **not** wait for memory.
- LDS ops of different waves can be reordered across the CU's queues. Before other waves
  read what this wave wrote: `s_waitcnt lgkmcnt(0)` then `s_barrier`.
- A buffer that all waves read and then rewrite needs two barriers per round:

```
 all waves read buf ─► lgkmcnt(0) + s_barrier ─► write buf ─► lgkmcnt(0) + s_barrier ─► read buf
                       (nobody still reading)                 (everybody's write landed)
```

- Every wave must execute the same number of `s_barrier`s. An early-exit wave that skips one
  hangs the workgroup.
- With direct-to-LDS loads, the "writes landed" wait is `vmcnt`, not `lgkmcnt`.
- Global-memory visibility across workgroups needs cache control, not barriers:
  [memory-coherence-bits](memory-coherence-bits.md).

## 3. Manually inserted wait states (selected rows)

`s_nop N` gives N+1 wait states (N = 0..15). Independent instructions count too.

| First | Then | Waits |
|---|---|---|
| VALU writes VGPR | MFMA reads it | 2 |
| VALU writes VGPR | DPP op reads it | 2 |
| VALU writes EXEC | DPP op | 5 |
| VALU writes SGPR / VCC | `v_readlane` / `v_writelane` lane select | 4 |
| VALU writes SGPR | VMEM reads that SGPR | 5 |
| VALU writes VCC | `v_div_fmas` | 4 |
| `v_cmpx` writes EXEC | MFMA | 4 |
| SALU writes M0 | LDS add-TID op, `buffer_store_lds_dword`, global/scratch with LDS=1 (treat any LDS-DMA load the same) | 1 |
| x3/x4 store or cmpswap_x2 (VGPR offset) | any instruction overwrites its write-data VGPRs | 1 (2 if VALU) |
| SALU writes M0 | `s_sendmsg`, `s_movrel` | 1 |
| Trans op (`v_exp_f32`, `v_log_f32`, `v_rcp_f32`, `v_rsq_f32`, `v_sqrt_f32`, `v_sin/cos_f32`) | non-trans VALU uses result | 1 |
| `s_setreg` | `s_getreg` same reg | 2 |

### MFMA results (CDNA4 table 38, by pass count 2 / 4 / 8 / 16)

| MFMA (XDL) writes D | Then | Waits |
|---|---|---|
| | next MFMA, same opcode, SrcC == D exactly | 0 (back-to-back accumulate is forwarded) |
| | MFMA reads overlapping SrcC | 4 / 6 / 10 / 18 |
| | MFMA reads D as SrcA/SrcB | 5 / 8 / 12 / 20 |
| | VALU read/write, `v_accvgpr_read`, VMEM/LDS reads D | 5 / 8 / 12 / 20 |
| MFMA reads SrcC | VALU overwrites that SrcC (WAR) | 1 / 3 / 7 / 15 |

One pass = 4 cycles. Get the pass count from the ISA opcode table or the matrix-instruction
calculator. Padding every epilogue with 4× `s_nop 15` is far above the table and costs real
cycles in short-K or many-group kernels. Size the gap to the pass count, or fill it with
independent work (address math, the next tile's loads). An `s_waitcnt` before an MFMA that
reads load results needs no extra NOPs: the hazard rows are for VALU producers.

## 4. Inline asm in HIP (when not writing a whole `.s`)

- Put an ordered sequence in **one** `asm volatile` block. Separate blocks can be reordered.
- Use early-clobber outputs (`"=&v"`) when an output must not share a register with an input.
- Add a `"memory"` clobber around timing or sync code, or `-O2` may move or delete it.
- The compiler's hazard recognizer and `s_waitcnt` insertion do not look inside your block.
  Close each block with the waits and NOPs it needs.
- An MFMA written in inline asm is invisible to `sched_group_barrier` pipelining. Keep MFMA
  as the builtin and hand-schedule only the loads around it.

Sources: [AMD CDNA3 ISA — s_waitcnt, "Manually Inserted Wait States", MFMA dependency resolution](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) ·
[AMD CDNA4 ISA — §4 wait states, §7.6 table 38](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) ·
[LLVM AMDGPUUsage — GFX942 memory model (LDS ordering across waves)](https://llvm.org/docs/AMDGPUUsage.html) ·
[HIP issue 3333 — inline asm clobber pitfalls](https://github.com/ROCm/HIP/issues/3333) ·
[amd_matrix_instruction_calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
Verify specifics against the primary source.

---
Related: [registers-agpr-liveness](registers-agpr-liveness.md) · [lds-patterns-asm](lds-patterns-asm.md) · [memory-coherence-bits](memory-coherence-bits.md) · [pipelining-scheduling](../../kernel-optimization/levers/pipelining-scheduling.md)
