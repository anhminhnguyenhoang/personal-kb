---
name: kernel-abi-and-prologue
description: How a hand-written gfx942/gfx950 kernel starts — kernel descriptor directives, the initial SGPR/VGPR state, kernarg loading or preload, the wave-id prologue, and how a .co is built, inspected, and launched.
source: https://llvm.org/docs/AMDGPUUsage.html ; AMD CDNA3 and CDNA4 ISA reference guides ; https://github.com/ROCm/aiter
---

# Kernel ABI and prologue for hand-written AMDGCN

A raw `.s` kernel has no compiler to set up its state. The packet processor (CP) and the
wave launcher (SPI) fill some registers, and the kernel descriptor says which ones.
Get one field wrong and the kernel reads garbage or fails to launch.

## 1. The kernel descriptor (`.amdhsa_kernel` block)

| Directive | Meaning | Trap |
|---|---|---|
| `.amdhsa_next_free_vgpr` | Highest VGPR used + 1. On gfx90a/gfx942/gfx950 this is the **unified** file: `align(arch_vgprs, 4) + agprs` | Leaving out the AGPRs allocates too few registers |
| `.amdhsa_accum_offset` | First AGPR index in the unified file. Multiple of 4, range 4..256 | See [registers-agpr-liveness](registers-agpr-liveness.md) |
| `.amdhsa_next_free_sgpr` | Highest SGPR used + 1 | VCC/flat-scratch reservations add to it |
| `.amdhsa_group_segment_fixed_size` | Static LDS bytes per workgroup | Dynamic LDS passed at launch adds on top |
| `.amdhsa_private_segment_fixed_size` | Scratch bytes per lane | Must be 0 for a spill-free kernel |
| `.amdhsa_user_sgpr_*` | Which user SGPRs CP fills (kernarg ptr, dispatch ptr, ...) | Changes every later SGPR number |
| `.amdhsa_user_sgpr_kernarg_preload_length` | Dwords of kernargs preloaded into SGPRs (gfx942/gfx950 support it) | Shifts the workgroup-id SGPRs |
| `.amdhsa_system_sgpr_workgroup_id_{x,y,z}` | Ask SPI to write WG ids | Off = no id |

The assembler packs these into `compute_pgm_rsrc1/2/3`. On gfx90a, gfx942 and gfx950,
`ACCUM_OFFSET` lives in `compute_pgm_rsrc3[5:0]` as `accum_offset/4 - 1`.
`VGPR` granule in `rsrc1` is 8 registers on these targets.

## 2. Initial register state

SGPRs are numbered **densely from s0** over the enabled fields, in a fixed order:

```
 user SGPRs (set by CP, same for all waves)        system SGPRs (set by SPI, per wave)
 ┌───────────────┬──────────────┬───────────────┐ ┌───────┬───────┬───────┐
 │ ...           │ kernarg ptr  │ preloaded     │ │ WG id │ WG id │ WG id │
 │ (dispatch ptr,│  (2 SGPRs)   │ kernargs (N)  │ │   X   │   Y   │   Z   │
 │  queue ptr..) │              │               │ │       │       │       │
 └───────────────┴──────────────┴───────────────┘ └───────┴───────┴───────┘
  s0 ...                                            s[USER_SGPR_COUNT] ...
```

- With only the kernarg pointer enabled: `s[0:1]` = kernarg ptr, `s2/s3/s4` = WG id X/Y/Z.
- With preload: the WG ids move up to `s[USER_SGPR_COUNT]`. A kernel that hard-codes `s2`
  then reads a kernarg as its block id.
- gfx942 and gfx950 use **packed work-item ids**: `v0[9:0]` = X, `v0[19:10]` = Y,
  `v0[29:20]` = Z. Extract with `v_and_b32` / `v_bfe_u32` before you reuse `v0`.

## 3. Loading kernargs

```asm
s_mov_b32       s29, s4              ; save WG id Z FIRST: the load below lands in s[4:7]
s_load_dwordx4  s[4:7],  s[0:1], 0x0
s_load_dwordx4  s[8:11], s[0:1], 0x10
s_waitcnt       lgkmcnt(0)           ; SMEM can return out of order: only 0 is safe
```

- A scalar load destination is written on return, so copy any live SGPR in the destination
  range before you issue the load.
- Multi-dword SGPR destinations must be aligned (x2 → even, x4 → multiple of 4).
- `s_movk_i32` sign-extends its 16-bit immediate: `s_movk_i32 s0, 0x8000` gives
  `0xFFFF8000`. Use `s_mov_b32` with a literal for values ≥ 0x8000.

## 4. Wave layout prologue

A 256-thread workgroup (4 waves) puts one wave on each SIMD of the CU, which is why most
production asm kernels use it. The standard prologue:

```asm
v_and_b32           v1, 0x3ff, v0    ; tid.x from packed v0
v_lshrrev_b32       v2, 6, v1        ; wave_id (same value in every lane)
v_and_b32           v3, 63, v1       ; lane_id
v_readfirstlane_b32 s20, v2          ; wave_id in an SGPR → scalar branches, no divergence
```

Keep wave-uniform values (wave id, tile base, loop bounds, buffer descriptors) in SGPRs.
Every value moved from a VGPR to an SGPR frees 64 lanes of register file.

## 5. Build, inspect, launch

```
 .s ──clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950──► kernel.co (ET_DYN)
         or: llvm-mc --filetype=obj  ──►  ld.lld -shared  ──► kernel.co
 kernel.co ──llvm-objdump -d --mcpu=gfx950──► disassembly
 kernel.co ──llvm-readelf --notes──► msgpack metadata (.args, vgpr_count, LDS, ...)
```

- The runtime loads a **shared object**. A relocatable `.o` (`-c` only) fails to load.
- The code-object metadata (`amdhsa.kernels`: `.args`, `.kernarg_segment_size`,
  `.group_segment_fixed_size`, `.vgpr_count`, ...) is what the runtime uses to size the
  launch. Write it fully. An empty `.args` list is a common cause of launch failure.
- To edit a binary you did not write: disassemble, rebuild, check the rebuilt `.text` is
  **bit-identical** before you change anything, then swap only `.text` into the original
  file (`llvm-objcopy --update-section .text=...`) so the original metadata survives.
- Launching a raw `.co` from host code: `hipModuleLoad` / `hipModuleGetFunction`, then
  `hipModuleLaunchKernel` with `HIP_LAUNCH_PARAM_BUFFER_POINTER` / `_SIZE` pointing at a
  **host** byte buffer laid out exactly as `.args` says.
- Public ROCm/aiter asm kernels pad each kernarg to a 16-byte slot (pointer + pad,
  int + 12-byte pad). The host struct uses explicit padding members to match. Mirror the
  `.args` offsets, not the C field order.

Sources: [LLVM AMDGPUUsage — kernel descriptor, initial kernel execution state, preloaded kernargs, code object metadata](https://llvm.org/docs/AMDGPUUsage.html) ·
[AMD CDNA3 ISA (SOPK, SMEM)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) ·
[AMD CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) ·
[HIP module API](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___module.html) ·
[ROCm/aiter (`csrc/include/aiter_hip_common.h`, `hsa/`)](https://github.com/ROCm/aiter)
Paths drift; verify before use.

---
Related: [registers-agpr-liveness](registers-agpr-liveness.md) · [waitcnt-and-hazards](waitcnt-and-hazards.md) · [debugging-playbook](debugging-playbook.md) · [gfx942-registers-occupancy](../../hardware/amd-cdna/gfx942-registers-occupancy.md)
