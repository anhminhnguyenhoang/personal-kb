---
name: scope-and-ceiling
description: When Triton is the right tool on AMD CDNA, NVIDIA habits that hurt on wave64, and the signs that a Triton kernel has hit its compiler-scheduling ceiling so you should switch to HIP, CK, or asm.
source: https://arxiv.org/abs/2511.08083 ; https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html ; https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
---

# Triton on AMD: scope and ceiling

## Where Triton fits

| Use Triton for | Use something else for |
|---|---|
| fusions a library cannot express (epilogues, norm+quant, odd attention variants) | plain dense GEMM: tuned vendor libraries usually win |
| fast prototyping and shape exploration | the last 10–30% of peak on a hot GEMM or attention |
| memory-bound ops: the compiler loses little there | block-scaled GEMM where scale handling dominates |
| skinny / decode GEMM with split-K | |
| `torch.compile` / Inductor output | |

HipKittens (arXiv 2511.08083) reports that compiler-based AMD backends, Triton included, trail
hand-scheduled kernels on CDNA3/CDNA4 GEMM and attention, partly from register-lifetime and
scheduling limits.

## NVIDIA habits to drop

```
 NVIDIA habit                       AMD CDNA reality
 ────────────────────────────────── ─────────────────────────────────────────────
 warp = 32 lanes                    wave = 64 lanes (num_warps=4 -> 256 threads)
 num_warps=8 default                often spills VGPRs; start at 4
 num_stages=3..4 for GEMM           2 single GEMM, 1 fused attention (gfx942)
 228 KB shared mem                  64 KB LDS/CU gfx942, 160 KB gfx950
 OCP fp8 everywhere                 FNUZ fp8 on gfx942, OCP on gfx950
 grid sized for ~132 SMs            304 CUs (MI300X) / 256 (MI355X): aim >= 1024 workgroups
```

## Signs of the ceiling

Tune the knobs first ([amd-knobs](amd-knobs.md)). Then read the ISA ([reading-amdgcn](reading-amdgcn.md)).
You are at the ceiling when **all** of these hold:

1. Resources are clean: no scratch, wide loads, `ds_*_b128`, the right MFMA.
2. Knob sweeps (tiles, warps, stages, `waves_per_eu`, MFMA size) all land in the noise band.
3. The profile is wait-bound: PC samples pile up on `s_waitcnt`, and MFMA busy is far below
   what the roofline allows.
4. In the ISA, MFMAs come in bursts with memory ops grouped between them, not interleaved.

```
 wanted:   mfma  ds_read  mfma  global_load  mfma  ds_read  mfma ...
 ceiling:  global_load x8 ; s_waitcnt ; ds_read x8 ; s_waitcnt ; mfma x16 ; ...
```

Cause: the generic LLVM scheduler does not reliably interleave independent MFMA and memory ops
for this loop. Newer Triton adds experimental MFMA scheduling (`schedule_hint`) and ping-pong; try
them first on your build.

Next step if they do not help: HIP with MFMA builtins and explicit `sched_group_barrier` /
`s_waitcnt` placement, CK, or a tile DSL with schedule control. Method and scheduling theory:
[pipelining-scheduling](../../kernel-optimization/levers/pipelining-scheduling.md),
HipKittens schedules: [schedules](../../papers/hipkittens/01-paper/schedules.md).

## Before you port

- Measure at the system level. A 1.3× kernel that is 2% of end-to-end time is not worth a rewrite.
- Keep the Triton kernel as the correctness reference for the new one.

Verify specifics against the primary source.

Sources: [HipKittens (arXiv 2511.08083)](https://arxiv.org/abs/2511.08083) ·
[ROCm workload optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html) ·
[Triton AMD backend compiler.py (schedule_hint, ping-pong)](https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py) ·
[LLVM AMDGPU usage](https://llvm.org/docs/AMDGPUUsage.html)

---
Related: [amd-knobs](amd-knobs.md) · [reading-amdgcn](reading-amdgcn.md) · [overview-thesis](../../papers/hipkittens/01-paper/overview-thesis.md)
