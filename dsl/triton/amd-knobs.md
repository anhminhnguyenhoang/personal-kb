---
name: amd-knobs
description: "The AMD-only Triton knobs (matrix_instr_nonkdim, waves_per_eu, kpack, num_stages, schedule_hint, knobs.amd.*): what each changes in codegen, safe start values, and where they must be set."
source: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py ; https://github.com/triton-lang/triton/blob/main/python/triton/knobs.py ; https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# Triton on AMD: the knob set

The Python API is the same as on NVIDIA. The AMD backend adds its own compile options.
They live in `HIPOptions` in `third_party/amd/backend/compiler.py`.

## Rule 0: where to set them

AMD options are compile options. Put them in the `triton.Config` kwargs (or the launch kwargs).
A Python variable with the same name does nothing.

```python
triton.Config(
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8,
     "matrix_instr_nonkdim": 16, "kpack": 2, "waves_per_eu": 2},   # AMD options here
    num_warps=4, num_stages=2)
```

## The map

```
 knob                    stage it acts on          what it changes
 ─────────────────────── ───────────────────────── ───────────────────────────────
 matrix_instr_nonkdim -> TTGIR accelerate-matmul -> MFMA shape (16x16 or 32x32)
 kpack                -> TTGIR accelerate-matmul -> K width per LDS read (gfx942)
 num_stages           -> TTGIR stream pipeliner  -> LDS-staged tiles in flight
 num_warps            -> TTGIR layouts           -> waves per workgroup (x64 lanes)
 waves_per_eu         -> LLVM fn attribute       -> VGPR budget -> occupancy
 schedule_hint        -> LLVM IR pass            -> MFMA / memory interleave
 knobs.amd.* (env)    -> pass on/off switches    -> buffer ops, async copy, ping-pong
```

## Per-knob notes

| Knob | Default | Start value | Effect and trap |
|---|---|---|---|
| `num_warps` | 4 | 4 for GEMM/attention; 2–4 for memory-bound | One warp = one 64-lane wave. 8 copied from NVIDIA configs often spills VGPRs. |
| `num_stages` | 2 | 2 single GEMM; 1 two fused GEMMs (attention); 2 GEMM+elementwise; 1 no GEMM | Extra stages cost LDS. On 64 KB LDS (gfx942) 3–4 stages cut occupancy. gfx950 (160 KB) can afford 3–4; benchmark. |
| `matrix_instr_nonkdim` | 0 (compiler picks) | 16 | 16 → 16x16 MFMA. 32 → 32x32 MFMA, bigger accumulator per wave, needs BLOCK_M/N divisible by 32. AMD docs: 16 usually wins for GEMM on MI300X. |
| `kpack` | 1 | 2 on gfx942 fp16/bf16, BLOCK_K ≥ 64 | Wider dot-operand LDS reads (`ds_read_b128`). Costs VGPRs. Deprecated on gfx950: forced to 1 with a warning. |
| `waves_per_eu` | 0 (off) | 0, then 2–3 if VGPR-limited | Emits `amdgpu-waves-per-eu`. Use only when VGPR use is a little above an occupancy step. Too high → spills. |
| `schedule_hint` | empty | empty | Experimental. Values change between releases (see below). |
| `maxnreg` | none | none | Hard register cap. Rarely better than `waves_per_eu`. |

### waves_per_eu arithmetic

512 VGPRs per SIMD. Allocation step is 16 (gfx942). Max waves = floor(512 / round_up_16(vgpr)).

```
vgpr used : 170 -> rounds to 176 -> 176*3 = 528 > 512 -> 2 waves/SIMD
set waves_per_eu=3 -> LLVM tries to fit 160 -> 160*3 = 480 -> 3 waves/SIMD
check: scratch must stay 0, else back off
```

The full register model (unified VGPR+AGPR file on gfx942) is in
[gfx942-registers-occupancy](../../hardware/amd-cdna/gfx942-registers-occupancy.md).

## Grid-shape constexprs (kernel code, not backend options)

- `GROUP_SIZE_M`: program-ID swizzle for L2 reuse. 8 is a good start.
  For XCD-aware remap see [chiplet-xcd-remap](../../kernel-optimization/levers/chiplet-xcd-remap.md).
- `SPLIT_K`: split the K loop over programs for skinny or decode GEMMs.
  AMD guidance: aim for at least 1024 workgroups on MI300X (304 CUs).
  Cost: C zero-init plus atomics (or a reduce pass).

## Process-wide switches (`triton.knobs.amd`, env vars)

| Env var | Effect | Note |
|---|---|---|
| `AMDGCN_ENABLE_DUMP=1` | print final AMDGCN | see [reading-amdgcn](reading-amdgcn.md) |
| `MLIR_ENABLE_DUMP=1` | print IR after each pass | see layouts, LDS use |
| `AMDGCN_USE_BUFFER_OPS` | lower loads/stores to `buffer_*` | on by default in current upstream; older builds had it off |
| `TRITON_HIP_USE_ASYNC_COPY` | direct global→LDS loads | default on gfx950, off on gfx942 |
| `TRITON_HIP_USE_BLOCK_PINGPONG` | two warp groups alternate MFMA and memory | default on gfx942 (and gfx950 with async copy); needs `num_stages > 1` |
| `TRITON_PRINT_AUTOTUNING=1` | print the winning config | use while tuning |
| `TRITON_ALWAYS_COMPILE=1` | skip the kernel cache | use so dumps match the run |

`OPTIMIZE_EPILOGUE=1` appears in older ROCm guides. Current upstream runs an epilogue-optimize
pass always. Check your build before you rely on the env var.

## Version drift: check before use

The option set moves fast. Examples seen in upstream history:
- `schedule_hint` once took `attention` / `memory-bound-attention`; current main accepts only
  `mfma-schedule` (gfx942/gfx950) and rejects other values.
- Older ROCm forks used `instruction_sched_variant` instead.
- `allowed_dot_input_precisions` now lists `ieee`, `bf16x3`, `bf16x6`.

Always read `HIPOptions` and `knobs.amd` in the build you run.

## Ship the winner, not the autotuner

Autotune on the serving hot path adds first-call latency and varies between runs.
Common practice in vLLM/SGLang: a per-shape JSON table made by an offline sweep, loaded at start.
Tables depend on the Triton/ROCm build. Re-sweep after an upgrade.

Verify specifics against the primary source.

Sources: [compiler.py HIPOptions](https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py) ·
[knobs.py amd_knobs](https://github.com/triton-lang/triton/blob/main/python/triton/knobs.py) ·
[ROCm workload optimization: Triton section](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html) ·
[PyTorch #143286 (AMD knobs in Inductor)](https://github.com/pytorch/pytorch/pull/143286) ·
[vLLM fused_moe per-shape configs](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe)
Paths drift; verify before use.

---
Related: [codegen-pipeline](codegen-pipeline.md) · [reading-amdgcn](reading-amdgcn.md) · [gfx942-registers-occupancy](../../hardware/amd-cdna/gfx942-registers-occupancy.md)
