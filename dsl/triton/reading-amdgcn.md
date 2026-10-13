---
name: reading-amdgcn
description: Dump and read the AMDGCN a Triton kernel produces, with a good-vs-bad table for the GEMM/attention inner loop and an occupancy check that sets waves_per_eu.
source: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html ; https://llvm.org/docs/AMDGPUUsage.html ; https://github.com/triton-lang/triton/blob/main/python/triton/knobs.py
---

# Triton on AMD: read the AMDGCN

Autotune time says *which* config won. The ISA says *why*, and it shows silent slow paths:
narrow loads, spills, a wrong MFMA, too many waits.
The general counting method is in [isa-resource-inspection](../../kernel-optimization/levers/isa-resource-inspection.md).
This note is the Triton-specific route to the ISA.

## 1. Get the dump

```bash
AMDGCN_ENABLE_DUMP=1      # final AMDGCN to stdout/stderr
MLIR_ENABLE_DUMP=1        # IR after each pass (TTIR, TTGIR, LLVM)
TRITON_ALWAYS_COMPILE=1   # skip cache, so the dump is this compile
TRITON_PRINT_AUTOTUNING=1 # which config produced it
python run.py > dump.txt 2>&1
```

Without env vars: the compiled kernel object holds the stages.
`kernel.asm["amdgcn"]`, `kernel.asm["ttgir"]`, `kernel.asm["llir"]`; the `hsaco` stage is the code object
(`llvm-objdump -d --arch=amdgcn`).

With autotune, many configs compile. Find the block whose kernel metadata matches the printed winner.

## 2. Resource lines (kernel metadata)

| Field | Meaning | Want |
|---|---|---|
| `.vgpr_count` | VGPRs per lane (incl. AGPRs on gfx942 unified file) | below the next occupancy step |
| `.sgpr_count` | scalar regs | rarely the limit |
| `.group_segment_fixed_size` | LDS bytes per workgroup | fits the per-CU budget × target workgroups |
| `.private_segment_fixed_size` | scratch bytes per lane | **0**. Non-zero = spills to memory. |
| `.max_flat_workgroup_size` | `num_warps * 64` | matches intent |

## 3. Inner-loop checklist

| Look for | Good | Bad → what to try |
|---|---|---|
| global loads | `global_load_dwordx4` / `buffer_load_dwordx4` | `_dword` / `_dwordx2`: prove contiguity (`tl.multiple_of`), check strides |
| masked tail | `buffer_load_*` (range-checked) | `global_load` wrapped in `v_cmp` + `s_and_saveexec`: enable buffer ops |
| LDS | `ds_read_b128` / `ds_write_b128` | `ds_read_b32/b64`: `kpack=2` (gfx942), larger BLOCK_K, check swizzle |
| MFMA | expected shape, dense in the loop | wrong shape: check `matrix_instr_nonkdim`; long gaps: starved pipe |
| accumulator | stays in AGPR/VGPR across iterations | `v_accvgpr_read/write` copies each iteration |
| waits | `s_waitcnt vmcnt(N)` / `lgkmcnt(N)` with N > 0 in the loop | `vmcnt(0)` right after each load = no overlap |
| spills | none | `scratch_store` / `scratch_load` |

`s_waitcnt` counters: `vmcnt` counts vector memory ops in flight; `lgkmcnt` counts LDS, scalar
memory, and message ops. `N` = how many may still be pending when the wave goes on.

## 4. Occupancy check, then `waves_per_eu`

```
1. vgpr = .vgpr_count            e.g. 170
2. round up to 16                 -> 176
3. waves/SIMD = floor(512/176)    -> 2
4. 3 waves need vgpr <= 160      (512/3 = 170.6, round down to step 16)
5. set waves_per_eu=3, recompile, re-read:
     vgpr <= 160 and scratch == 0  -> keep if faster
     scratch > 0                   -> back off
```

Also apply the LDS limit: workgroups/CU ≤ floor(LDS per CU / `.group_segment_fixed_size`).
Occupancy = min of the VGPR limit and the LDS limit. ROCm docs give the formula and point to an
`occ.sh` helper in ROCm/triton.

## 5. Dtype sanity

- fp16/bf16, `nonkdim=16` → `v_mfma_f32_16x16x16_{f16,bf16}`. A `32x32` name means the option was
  32 or the compiler picked it.
- fp8 on gfx942 → `*_fp8_fp8` MFMA. If the dot does not lower, you passed OCP fp8:
  see [low-precision-dtypes](low-precision-dtypes.md).
- MXFP4 on gfx950 → `v_mfma_scale_f32_*_f8f6f4`. Also count `ds_read_u8` and waits around it:
  scale loads can dominate.

## 6. When ISA is bad, go up

Bad ISA → LLVM IR → TTGIR → TTIR. Find the first stage where the pattern appears.
In TTGIR, count `convert_layout` and `local_alloc`/shared buffers; each is LDS traffic.
Then find the pass in the `MLIR_ENABLE_DUMP` output that introduced it.

Sources: [ROCm workload optimization: MLIR and ISA analysis, occupancy](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html) ·
[LLVM AMDGPU usage (s_waitcnt, metadata)](https://llvm.org/docs/AMDGPUUsage.html) ·
[Triton knobs.py (AMDGCN_ENABLE_DUMP, buffer ops)](https://github.com/triton-lang/triton/blob/main/python/triton/knobs.py) ·
[ROCm/triton (occ.sh)](https://github.com/ROCm/triton)

---
Related: [codegen-pipeline](codegen-pipeline.md) · [amd-knobs](amd-knobs.md) · [isa-resource-inspection](../../kernel-optimization/levers/isa-resource-inspection.md)
