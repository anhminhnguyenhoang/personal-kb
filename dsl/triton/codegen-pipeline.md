---
name: codegen-pipeline
description: How the AMD Triton backend lowers a kernel (TTIR, TTGIR, TritonAMDGPU, LLVM, AMDGCN), where layouts and convert_layout cost LDS round-trips, and how tl.dot maps to MFMA shapes.
source: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py ; https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html ; https://github.com/ROCm/amd_matrix_instruction_calculator
---

# Triton on AMD: the codegen pipeline

Know the stages. Then you can predict what a knob does and where a bad instruction came from.

## Stages

```
@triton.jit Python
   │
   ▼ TTIR            arch-free tensor ops (tt.load, tt.dot, ...)
   │
   ▼ TTGIR           every tensor gets a LAYOUT: blocked / shared / slice / mfma / dot_op
   │   coalesce · remove-layout-conversions
   │   accelerate-matmul   <- matrix_instr_nonkdim, kpack pick the MFMA layout
   │   optimize-epilogue · optimize-dot-operands
   │   schedule-loops + stream pipeliner   <- num_stages
   │   block ping-pong (num_stages > 1)    · convert-to-buffer-ops
   │
   ▼ LLVM IR         fn attrs: amdgpu-waves-per-eu, amdgpu-flat-work-group-size
   │   optional MFMA scheduler (schedule_hint)
   │
   ▼ AMDGCN          v_mfma_*, ds_read/write_b*, global/buffer_load_dwordx4, s_waitcnt
   ▼ HSACO           loaded by the HIP runtime
```

The pass list is in `make_ttgir` in the backend `compiler.py`. When a dump shows a bad pattern,
use `MLIR_ENABLE_DUMP=1` to find the pass that made it.

## Layouts and `convert_layout`

In TTGIR each tensor has a layout (how elements map to lanes and registers).
If a producer and a consumer disagree, the compiler inserts `convert_layout`.
If data must cross lanes, that becomes an LDS round-trip: `ds_write`, barrier, `ds_read`.

| Convert | When | Can you avoid it? |
|---|---|---|
| blocked → `dot_op` (MFMA operand) | every GEMM operand | No. Make it cheap: wide reads, good swizzle. |
| `mfma` acc → blocked (store) | epilogue | Often. The epilogue pass stores from the MFMA layout. |
| blocked → shared → blocked (transpose) | `tl.trans`, dequant | Yes, by restructuring. ROCm docs show an int4 dequant that hits LDS twice when once is enough. |

Count `convert_layout` ops in the TTGIR dump. Each one that crosses lanes costs LDS traffic and a sync.

## `tl.dot` to MFMA

`tl.dot(a, b, acc)` gets an MFMA layout in accelerate-matmul, then lowers to `v_mfma_*`.
The MFMA is wave-wide: all 64 lanes hold parts of A, B, C.

| Input dtype | nonkdim=16 (gfx942) | nonkdim=32 (gfx942) |
|---|---|---|
| fp16 / bf16 | `v_mfma_f32_16x16x16` | `v_mfma_f32_32x32x8` |
| fp8 (FNUZ on gfx942) | `v_mfma_f32_16x16x32_fp8_fp8` | `v_mfma_f32_32x32x16_fp8_fp8` |
| int8 | `v_mfma_i32_16x16x32_i8` | `v_mfma_i32_32x32x16_i8` |

gfx950 adds K-doubled bf16/fp16 shapes, OCP fp8, and block-scaled `v_mfma_scale_f32_*_f8f6f4`
(see [low-precision-dtypes](low-precision-dtypes.md)). Atom choice in general:
[matrix-instructions](../../kernel-optimization/levers/matrix-instructions.md).

Why 16 is the usual start: a 32x32 tile holds a bigger accumulator per wave. That raises register
pressure and gives the scheduler fewer, larger units to interleave.

## The stream pipeliner (`num_stages` on AMD)

AMD does not use the NVIDIA `cp.async` + mbarrier pipeline.
The AMD pass software-pipelines the K loop: load tile k+1 while MFMAs run on tile k.

```
num_stages=2:   iter k :  [global load k+1 -> regs/LDS]   [ds_read k -> MFMA k]
                iter k+1: [global load k+2 ...]            [ds_read k+1 -> MFMA k+1]
LDS cost ~ (BM*BK + BK*BN) * bytes * num_stages
```

- With async copy (gfx950 default) loads go global → LDS directly. No VGPR staging.
- Block ping-pong splits the workgroup into two groups. One issues MFMA while the other issues
  memory ops. It needs `num_stages > 1`.

Pipeline and ping-pong theory, DSL-neutral: [pipelining-scheduling](../../kernel-optimization/levers/pipelining-scheduling.md).

## Memory ops

- Wide loads: the compiler emits `global_load_dwordx4` only if it can prove 16-byte contiguity and
  alignment. Help it with `tl.multiple_of(ptr, 16)` / `tl.max_contiguous` on offsets.
- Buffer ops: `buffer_load` uses a resource descriptor with hardware range check. Out-of-range
  lanes read 0, so masked tails need no branch. Upstream applies this only to tensors proven
  smaller than 2 GB.

## LLVM stage

The backend adds `amdgpu-waves-per-eu` (from `waves_per_eu`) and the flat workgroup size
(`num_warps * 64`). Then the register allocator decides `.vgpr_count`, `.sgpr_count`, LDS size,
and scratch. Read them in the dump: [reading-amdgcn](reading-amdgcn.md).

Sources: [Triton AMD backend compiler.py](https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py) ·
[ROCm workload optimization (MLIR and ISA analysis)](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html) ·
[AMD CDNA3 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) ·
[AMD matrix instruction calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
Paths drift; verify before use.

---
Related: [amd-knobs](amd-knobs.md) · [reading-amdgcn](reading-amdgcn.md) · [memory-hierarchy](../../kernel-optimization/levers/memory-hierarchy.md)
