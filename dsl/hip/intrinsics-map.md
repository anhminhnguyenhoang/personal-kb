---
name: intrinsics-map
description: HIP __builtin_amdgcn_* map for CDNA3/CDNA4 — MFMA and scaled MFMA, buffer descriptors, direct-to-LDS, transposed LDS reads, DPP/permute, sched_group_barrier masks, s_waitcnt encoding, inline-asm register classes.
source: https://llvm.org/docs/AMDGPUUsage.html ; https://github.com/llvm/llvm-project/blob/main/clang/include/clang/Basic/BuiltinsAMDGPU.def ; https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/ ; https://github.com/ROCm/amd_matrix_instruction_calculator ; CDNA3 and CDNA4 ISA PDFs
---

# HIP intrinsics map (gfx942 / gfx950)

One table per family. `942` = CDNA3 (MI300), `950` = CDNA4 (MI350).
Every builtin below was compiled with a ROCm 7.x clang for both targets.
Builtin names and signatures drift between LLVM versions. Check `BuiltinsAMDGPU.def` for your toolchain.

```
 HBM ──buffer_load / global_load──► VGPR ──ds_write──► LDS ──ds_read(_tr)──► VGPR ──v_mfma──► VGPR/AGPR
   └────────── buffer_load ... lds / global_load_lds (skip VGPR) ──────────►┘
 cross-lane:  DPP (free, in VALU) · permlane*_swap (VALU, 950) · ds_bpermute/ds_swizzle (LDS crossbar, lgkmcnt)
 ordering:    sched_barrier / sched_group_barrier / iglp_opt (compile-time only) · s_waitcnt · s_barrier · s_setprio
```

## 1. MFMA

Call form: `d = __builtin_amdgcn_mfma_<Cty>_<M>x<N>x<K>[<ABty>](a, b, c, cbsz, abid, blgp)`.
Set `cbsz = abid = blgp = 0` for a plain GEMM (they are broadcast controls).

| Builtin | Arch | A / B per lane | C per lane |
|---|---|---|---|
| `mfma_f32_16x16x16bf16_1k` | 942, 950 | 4 × bf16 (short4) | 4 × f32 |
| `mfma_f32_16x16x32_bf16` / `_f16` | 950 | 8 × bf16 / f16 | 4 × f32 |
| `mfma_f32_32x32x16_bf16` / `_f16` | 950 | 8 × bf16 / f16 | 16 × f32 |
| `mfma_f32_16x16x32_fp8_fp8` (and `_fp8_bf8`, `_bf8_fp8`, `_bf8_bf8`) | 942, 950 | 8 × fp8 (one `long`) | 4 × f32 |
| `mfma_scale_f32_16x16x128_f8f6f4` | 950 | int32x8 (32 fp8 per lane) | 4 × f32 |
| `mfma_scale_f32_32x32x64_f8f6f4` | 950 | int32x8 | 16 × f32 |

Rule of thumb: per-lane elements = tile elements ÷ 64. Get the exact lane-to-element map from the AMD Matrix Instruction Calculator. Do not guess it.

**fp8 format depends on the arch.** On gfx942 the fp8 MFMAs read **FNUZ** encodings. On gfx950 they read **OCP** encodings (e4m3fn / e5m2). The same byte decodes to values a factor of 2 apart, because the exponent bias differs. Quantize with the format of the target arch.

### Scaled MFMA (MX block scale, gfx950)

```cpp
acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
    a, b, acc,
    /*cbsz  = A format*/ 0, /*blgp = B format*/ 0,
    /*opsel_a*/ 0, scale_a,      // opsel and scale alternate per operand
    /*opsel_b*/ 0, scale_b);
```

| Item | Rule |
|---|---|
| Format code (cbsz for A, blgp for B) | 0 fp8 e4m3 · 1 bf8 e5m2 · 2 fp6 e2m3 · 3 bf6 e3m2 · 4 fp4 e2m1 |
| Scale value | E8M0 byte: value = 2^(e − 127). **127 = 1.0. 0 = 2^-127 (about zero). 0xFF = NaN.** (OCP MX) |
| `opsel_*` | Picks which byte of the 32-bit scale register is used, so four scales can share one VGPR |
| Block size | One scale per 32 K-elements (OCP MX) |

**Literal 0 and runtime 0 do different things.** A compile-time constant `0` for both scales makes clang emit the *unscaled* `v_mfma_f32_16x16x128_f8f6f4` (×1.0). A runtime scale register holding `0x00` multiplies by 2^-127 per operand, so the output is zero.
Measured on gfx950 with A = B = fp8 1.0 and K = 128, sweeping a runtime scale byte:

| scale byte | 0 | 1 | 126 | 127 | 128 | 129 |
|---|---|---|---|---|---|---|
| output | 0 | 0 | 32 | 128 | 512 | 2048 |

The same byte scales both A and B here, so each step is ×4. When you want "no scaling" from a runtime value, use 127, not 0.
Verify specifics against the primary source (CDNA4 ISA, MFMA scale section).

### Accumulator register class

On gfx90a and newer, MFMA `C`/`D` can be VGPRs **or** AGPRs. The 512-entry file is unified, so the compiler uses AGPRs only when it runs out of architectural VGPRs.
To force a value into one class, use an inline-asm constraint:

| Constraint | Class | Effect on an accumulator |
|---|---|---|
| `"+v"(acc)` | arch VGPR | keeps acc live here; MFMA writes `v[...]` |
| `"+a"(acc)` | AGPR | MFMA writes `a[...]` directly; costs `v_accvgpr_write` for init |
| `"s"` | SGPR | wave-uniform scalars (descriptors, loop bounds) |

`asm volatile("" : "+v"(acc));` is an optimization fence, not a no-op. It forces `acc` into that register class at that point. Check the ISA for new `v_accvgpr_read/write` after adding one.
HIPCC will not pick AGPRs as MFMA *A/B* inputs; see [hipcc-agpr-pinning](../../papers/hipkittens/01-paper/hipcc-agpr-pinning.md).

## 2. Buffer descriptors and loads

```cpp
__amdgpu_buffer_rsrc_t r = __builtin_amdgcn_make_buffer_rsrc(base, /*stride*/(short)0, num_bytes, flags);
int4  v = __builtin_amdgcn_raw_buffer_load_b128(r, voffset_bytes, soffset, /*aux*/0);
float s = __builtin_amdgcn_raw_buffer_load_b32 (r, voffset_bytes, soffset, 0);
__builtin_amdgcn_raw_buffer_store_b128(v, r, voffset_bytes, soffset, 0);
```

| Item | Rule |
|---|---|
| Descriptor | 128-bit V#, must live in SGPRs. Build it from wave-uniform values. Otherwise the compiler emits a `v_readfirstlane` waterfall loop. |
| `flags` (dword 3) | `0x00020000` is what CK uses on gfx90a/gfx94x/gfx950. `0x00027000` also appears. Use one your library already uses. |
| OOB | With stride 0, `voffset ≥ num_records` → load returns 0, store is dropped. This replaces tail predication. |
| `aux` | cache-policy bits (glc/slc/nt). 0 = default caching. |
| Width | `b32/b64/b96/b128` → `buffer_load_dword{,x2,x3,x4}`. Prefer b128 in hot loops. |

## 3. Direct-to-LDS (skip VGPR staging)

```cpp
__builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, (__attribute__((address_space(3))) void*)lds_base,
                                         size, voffset, soffset, /*imm offset*/0, /*aux*/0);
__builtin_amdgcn_global_load_lds((__attribute__((address_space(1))) void*)gptr,
                                 (__attribute__((address_space(3))) void*)lds_base, size, 0, 0);
```

| `size` (bytes per lane) | gfx942 | gfx950 |
|---|---|---|
| 1, 2, 4 | yes | yes |
| 8 | **compile error** | **compile error** |
| 12, 16 | compile error | yes (`dwordx3`, `dwordx4`) |

The LDS base is wave-uniform (it goes in M0). Lane `i` writes `base + i*size`. Per-lane LDS swizzle is not possible here; put the swizzle into the *global* address. Give each wave its own base (see [memory-hierarchy](../../kernel-optimization/levers/memory-hierarchy.md) lessons).
Completion is tracked by `vmcnt`, not `lgkmcnt`.

## 4. Transposed LDS reads (gfx950 only)

| Builtin | ISA | Element |
|---|---|---|
| `ds_read_tr16_b64_v4i16` / `_v4f16` / `_v4bf16` | `ds_read_b64_tr_b16` | 16-bit |
| `ds_read_tr8_b64_v2i32` | `ds_read_b64_tr_b8` | fp8 / i8 |
| `ds_read_tr6_b96_v3i32` | `ds_read_b96_tr_b6` | fp6 |
| `ds_read_tr4_b64_v2i32` | `ds_read_b64_tr_b4` | fp4 |

The pointer argument must be `address_space(3)` and typed like the return value. Reinterpret the result with `__builtin_bit_cast`. On gfx942 these fail with "needs target feature gfx950-insts".

## 5. Cross-lane

| Builtin | Path | Use |
|---|---|---|
| `__builtin_amdgcn_update_dpp(old, src, ctrl, row_mask, bank_mask, bound_ctrl)` / `mov_dpp` | VALU modifier, no LDS | fixed neighbor shifts inside a 16-lane row |
| `__builtin_amdgcn_permlane16_swap(a, b, fi, bc)` / `permlane32_swap` | VALU, gfx950 | swap halves across 16/32 lanes; returns two values |
| `__builtin_amdgcn_ds_swizzle(v, offset)` | LDS crossbar | fixed pattern inside 32 lanes |
| `__builtin_amdgcn_ds_bpermute(src_lane*4, v)` / `ds_permute` | LDS crossbar | arbitrary gather / scatter; costs `lgkmcnt` |
| `__builtin_amdgcn_readfirstlane(v)` / `readlane(v, i)` | VALU → SGPR | make a value wave-uniform |

DPP `ctrl` codes (gfx9):

| ctrl | Pattern | ctrl | Pattern |
|---|---|---|---|
| `0x00–0xFF` | quad_perm | `0x130 / 0x134` | wave_shl1 / wave_rol1 |
| `0x101–0x10F` | row_shl 1–15 | `0x138 / 0x13C` | wave_shr1 / wave_ror1 |
| `0x111–0x11F` | row_shr 1–15 | `0x140 / 0x141` | row_mirror / row_half_mirror |
| `0x121–0x12F` | row_ror 1–15 | `0x142 / 0x143` | row_bcast15 / row_bcast31 |

`ds_swizzle` bit-mask mode (offset < 0x8000): `and = off[4:0]`, `or = off[9:5]`, `xor = off[14:10]`. XOR butterfly by `k` = `(k << 10) | 0x1F`.

## 6. Scheduling and ordering

| Builtin | Meaning |
|---|---|
| `__builtin_amdgcn_sched_barrier(mask)` | Scheduler fence. `mask` = classes allowed to cross. 0 = nothing crosses. |
| `__builtin_amdgcn_sched_group_barrier(mask, n, sync_id)` | Next group = `n` instructions of class `mask`. Groups with the same `sync_id` are ordered as written. |
| `__builtin_amdgcn_iglp_opt(k)` | Predefined interleave. 0 = DS/MFMA for small GEMM, 1 = same for single-wave small GEMM. |
| `__builtin_amdgcn_s_setprio(p)` | Wave priority 0–3 (`s_setprio`). |
| `__builtin_amdgcn_s_barrier()` | Workgroup barrier. |
| `__builtin_amdgcn_s_waitcnt(imm)` | Raw wait; see encoding below. |

`mask` classes (shared by both barriers):

| Bit | Class | Bit | Class |
|---|---|---|---|
| `0x001` | all non-memory ALU | `0x040` | VMEM write |
| `0x002` | VALU | `0x080` | all DS |
| `0x004` | SALU | `0x100` | DS read |
| `0x008` | MFMA / WMMA | `0x200` | DS write |
| `0x010` | all VMEM | `0x400` | transcendental (exp, log, rcp) |
| `0x020` | VMEM read | | |

**These three are compile-time pseudo-instructions.** In the `.s` file they appear only as comments (`; sched_group_barrier mask(0x8) size(1) SyncID(0)`) on gfx942 and on gfx950. They never reach the binary on any target. To check that a hint worked, read the **instruction order** in the ISA. Counting `s_sched_*` instructions tells you nothing.

### `s_waitcnt` immediate (gfx9 family, incl. gfx942/gfx950)

```
bit: 15 14 | 13 12 | 11 10 9 8 | 7 | 6 5 4 | 3 2 1 0
     vm[5:4]| --   | lgkmcnt   | - | expcnt| vm[3:0]
```

| Want | Immediate |
|---|---|
| wait all | `0x0000` |
| LDS only (`lgkmcnt(0)`, vmcnt 63, expcnt 7) | `0xC07F` |
| `vmcnt(N)` only, N < 16 | `0x0F70 \| N` |

`lgkmcnt` is 4 bits (max 15), so "lgkmcnt 16" cannot be encoded. `asm volatile("s_waitcnt vmcnt(%0)" :: "n"(N))` lets the assembler encode it. The compiler's wait-insertion pass can still merge or strengthen waits, so read the final ISA.

Sources: [LLVM AMDGPUUsage](https://llvm.org/docs/AMDGPUUsage.html) · [clang BuiltinsAMDGPU.def](https://github.com/llvm/llvm-project/blob/main/clang/include/clang/Basic/BuiltinsAMDGPU.def) · [LLVM sched_group_barrier review D128158](https://reviews.llvm.org/D128158) · [AMD lab notes: Matrix Cores (GPUOpen)](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/) · [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) · [CDNA3 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) · [CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) · [OCP MX spec v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) · [HIP low-precision types](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html) · [CK buffer-resource dword](https://github.com/ROCm/composable_kernel/blob/develop/include/ck/ck.hpp)
Paths drift; verify before use.

---
Related: [kernel-patterns](kernel-patterns.md) · [signal-recipes](signal-recipes.md) · [matrix-instructions](../../kernel-optimization/levers/matrix-instructions.md) · [gfx950](../../hardware/amd-cdna/gfx950.md)
