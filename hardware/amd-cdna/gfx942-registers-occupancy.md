---
name: gfx942-registers-occupancy
description: gfx942/CDNA3 (MI300X) per-SIMD register files (VGPR/AGPR/SGPR) and wave occupancy — the 512-total = 256 VGPR + 256 AGPR *flexible* split that differs from the gfx950 reference's separate-files model, the per-wave vs per-SIMD allocation arithmetic, and a worked occupancy example.
source: AMD Instinct MI300 CDNA3 ISA Reference Guide (2025-08-05); gfx950.md and papers/hipkittens/01-paper/hipcc-agpr-pinning.md in this KB
---

## gfx942 / CDNA3 register files & occupancy

The on-chip register storage that bounds occupancy. **All four register
families live inside the SIMD** (4 SIMDs per CU). Confusing their capacities
or the unit they're allocated in (per-wave vs per-SIMD) leads to mistuned
occupancy targets.

### The three register classes (per SIMD)

| Class | Width | Allocated per | Purpose |
|---|---|---|---|
| **VGPR** (vector GPR) | 32-bit × 64 lanes | wave (from a per-SIMD pool) | per-lane values; MFMA inputs; VALU operands |
| **AGPR** (accumulator GPR) | 32-bit × 64 lanes | wave (from a per-SIMD pool) | MFMA *output* accumulators; can also be loaded from memory |
| **SGPR** (scalar GPR) | 32-bit (uniform/wave) | wave | scalar/uniform values, addresses, predicates |

### gfx942 capacity — the key fact (and the cross-arch trap)

Per the **CDNA3 ISA reference**, on gfx942 a SIMD holds **512 total 32-bit
vector registers per wave, split into up to 256 VGPR + up to 256 AGPR, with a
*flexible* division when fewer than 512 total are needed**. VGPRs are allocated
in granules (ISA: groups of 8 dwords).

> ⚠️ **This differs from how the existing KB gfx950 reference models it.**
> [gfx950](gfx950.md) lists
> `VGPR/SIMD = 512` *and* `AGPR/SIMD = 256 (separate file)` — i.e. 512 + 256 as
> two independent budgets. The CDNA3 ISA describes gfx942 as **512 total,
> shared/flexible between the two pools (256 max each)**. For a *non-MFMA*
> kernel (AGPR usage = 0), both models give the same VGPR headroom (512), so the
> distinction only bites MFMA kernels that co-allocate VGPR+AGPR. Treat 512 as
> the **combined** vector-register budget on gfx942 until verified otherwise on
> the specific part with an occupancy probe or the `--save-temps` ISA.

| Quantity | gfx942 (MI300X) value |
|---|---|
| SIMDs / CU | 4 |
| Vector regs / SIMD (per wave) | **512 total = ≤256 VGPR + ≤256 AGPR, flexible** |
| SGPRs / CU | 800 (hardware); ~104 usable/wave |
| VGPR allocation granularity | 8 dwords (ISA) — gfx950 ref cites 16; verify per target |
| Waves / SIMD (hardware max) | 8 (⇒ 32 waves/CU) |
| Wave size | 64 (wave64) |
| LDS / CU | 64 KB (32 banks) |

### Allocation arithmetic — per **wave**, not per workgroup

VGPRs are handed out **per wavefront** from the SIMD's pool:

```
waves_per_SIMD (VGPR-limited) = floor( 512 / VGPRs_per_wave )     # rounded to granule
```

Occupancy = the **minimum** over all limiters:
- `__launch_bounds__(_, minBlocksPerCU)` request (compiler caps VGPRs to honor it),
- VGPR budget (above),
- LDS budget: `64 KB / LDS_per_workgroup`,
- SGPR budget,
- hardware cap: **8 waves/SIMD = 32 waves/CU**.

### Worked example — from block geometry to waves/SIMD

A workgroup of 128 threads = **2 waves/WG** (128 ÷ 64), launched with
`__launch_bounds__(128, N)` requesting **N workgroups/CU**. For N = 8:

```
2 waves/WG × 8 WG/CU = 16 waves/CU ÷ 4 SIMDs = 4 waves/SIMD  (achieved)
```

4 ≤ 8 (hw cap); if VGPR/LDS footprint also clears their budgets, the request is
honored. The CU *could* hold up to 32 waves — running at 16 is a deliberate
occupancy target, not a limit. For a memory-bound kernel this is often enough
workgroup-level parallelism to hide HBM latency without spending VGPRs to push
occupancy higher.

### Distinctions worth not confusing

- **2 waves/WG** (block geometry, = `kNumThreads/64`) ≠ **waves/SIMD** (per-SIMD
  occupancy) ≠ **waves/CU**. They relate by `waves/CU = WG/CU × waves/WG` and
  `waves/SIMD = waves/CU ÷ 4`.
- `__launch_bounds__(threads, N)`: 2nd arg = **workgroups per CU**, not waves.
- A wavefront is pinned to **one SIMD** for its lifetime.
- "512" on gfx942 is the **combined** vector-register count (VGPR+AGPR), not
  VGPR-only and not V+A+S.

Related: [gfx950](gfx950.md)
(gfx950/CDNA4 caps), [memory-hierarchy](../../kernel-optimization/10-levers/memory-hierarchy.md)
(storage taxonomy), [hipcc-agpr-pinning](../../papers/hipkittens/01-paper/hipcc-agpr-pinning.md) (AGPR-as-MFMA-input
allocator refusal).

Sources:
- [AMD Instinct MI300 (CDNA3) ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD CDNA 3 Architecture White Paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)
- [HIP — Hardware implementation](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html)
</content>
