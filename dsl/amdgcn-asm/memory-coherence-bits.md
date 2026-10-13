---
name: memory-coherence-bits
description: What the sc0/sc1/nt bits mean on gfx942/gfx950 loads, stores and atomics, the release/acquire sequences (buffer_wbl2, buffer_inv), and a correct cross-workgroup spin barrier in asm.
source: https://llvm.org/docs/AMDGPUUsage.html (Memory Model GFX942 and its code-sequence table) ; AMD CDNA3 and CDNA4 ISA reference guides (memory scope tables) ; https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html
---

# Coherence bits: sc0, sc1, nt

On gfx942 and gfx950, `sc0`, `sc1` and `nt` are not "bypass L1 / bypass L2" switches. They pick a
**scope**, and the meaning of `sc0` changes for atomics.

## 1. The bits

| On loads and stores | Scope | Stays coherent with |
|---|---|---|
| none | wavefront | this wave |
| `sc0` | workgroup | waves of the same workgroup |
| `sc1` | agent | every CU on this GPU |
| `sc0 sc1` | system | host and peer GPUs |

| Bit | Other meaning |
|---|---|
| `nt` | Non-temporal / streaming hint: do not keep the line. Use for data touched once |
| `sc0` on an **atomic** | Return the old value (the RMW always bypasses L1, so it is already agent-coherent) |
| `sc1` on an **atomic** | System scope (host / peer GPUs) for the RMW |

Code the LLVM memory model emits (GFX942 table):

| Operation | Agent scope | System scope |
|---|---|---|
| monotonic load | `global_load ... sc1` | `global_load ... sc0 sc1` |
| monotonic store | `global_store ... sc1` | `global_store ... sc0 sc1` |
| monotonic atomic RMW | `global_atomic_*` (add `sc0` to get the old value) | `global_atomic_* ... sc1` |
| release (before the flag store/atomic) | `buffer_wbl2 sc1` ; `s_waitcnt vmcnt(0) lgkmcnt(0)` | `buffer_wbl2 sc0 sc1` ; same wait |
| acquire (after the flag load/atomic) | `s_waitcnt vmcnt(0)` ; `buffer_inv sc1` | `s_waitcnt vmcnt(0)` ; `buffer_inv sc0 sc1` |
| volatile | `sc0 sc1` + `s_waitcnt vmcnt(0)` after each access | — |

Inside one workgroup, waves share the CU's L1, so no cache action is needed. Only LDS
ordering (`lgkmcnt(0)` + barrier) applies. Between workgroups the L1 may be stale, which
is why acquire needs `buffer_inv`.

## 2. Cross-workgroup spin barrier

```
 every WG                         wave 0 only                        all waves
 ─────────┐   release   ┌─ atomic add (+1) ─┐   spin   ┌─ load flag ─┐ acquire  ┌──────────┐
 stores   ├────────────►│ sc0 (return old)  ├─────────►│ sc1 / sc0sc1├─────────►│ s_barrier│
 done     │ wbl2+vmcnt  └───────────────────┘ s_sleep  └─────────────┘ buf_inv  └──────────┘
```

```asm
    ; ---- wave 0 of each workgroup ----
    buffer_wbl2      sc1                         ; release this WG's prior global stores
    s_waitcnt        vmcnt(0) lgkmcnt(0)
    global_atomic_add v_old, v_off, v_one, s[cnt:cnt+1] sc0   ; return old; add sc1 across GPUs
    s_waitcnt        vmcnt(0)
.Lspin:
    s_sleep          2                           ; back off; do not hammer L2
    global_load_dword v_val, v_off, s[cnt:cnt+1] sc1   ; agent-scope poll (sc0 sc1 for peers)
    s_waitcnt        vmcnt(0)
    v_cmp_ge_u32     vcc, v_val, s_target
    s_cbranch_vccz   .Lspin
    buffer_inv       sc1                         ; acquire: drop stale L1 lines
    ; ---- all waves ----
    s_barrier
```

- Poll with a **scope that reaches L2** (`sc1`). A `sc0`-only load is workgroup scope and
  can spin forever on a stale L1 line.
- Never poll with `s_load`: the scalar cache is not coherent with vector stores.
- Let one wave per workgroup spin, then `s_barrier`. Every lane of every wave spinning
  wastes memory bandwidth.
- All participating workgroups must be **resident at once**. Size the grid to what fits
  (or use a cooperative launch). Otherwise unscheduled workgroups never arrive and the
  barrier deadlocks.
- Reuse counters without a host reset: make the target grow per use
  (`target = round * num_wgs`).
- The flag orders nothing by itself. The producer releases (`buffer_wbl2` + `vmcnt(0)`)
  before it signals; the consumer acquires (`buffer_inv`) after it sees the flag. Skip
  either side and the data can be stale while the flag is fresh.

## 3. Across kernel launches

Kernel boundaries on one stream already include the release/acquire the runtime needs.
Extra `buffer_inv` at kernel entry, or mixing `flat_*` and `global_*`, is not a fix for
stale data between launches. Look for a missing wait or a real race first.

Sources: [LLVM AMDGPUUsage — "Memory Model GFX942" and "AMDHSA Memory Model Code Sequences GFX942"](https://llvm.org/docs/AMDGPUUsage.html) ·
[AMD CDNA3 ISA — memory scope and temporal hint tables](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) ·
[AMD CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) ·
[HIP cooperative groups (co-residency for grid sync)](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html)
Verify specifics against the primary source.

---
Related: [memory-instruction-choice](memory-instruction-choice.md) · [waitcnt-and-hazards](waitcnt-and-hazards.md) · [aiter-asm-kernel-patterns](aiter-asm-kernel-patterns.md)
